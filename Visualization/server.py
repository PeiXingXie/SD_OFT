#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一个“可按绝对路径读取远端文件”的轻量 HTTP 服务：
- 静态文件：默认从 --static-dir 提供（用于打开 HTML）
- API：
  - GET /api/text?path=ABS_PATH        读取文本（UTF-8）
  - GET /api/ls?path=ABS_DIR[&recursive=1][&limit=N]
                                    列目录下的图片文件（返回 JSON）
  - GET /api/bin?path=ABS_PATH         读取二进制（用于图片等），自动猜测 Content-Type

安全：
- 仅允许读取 --allow-prefix 指定前缀下的文件（可多次传入）。
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _guess_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _safe_realpath(p: str) -> str:
    # 统一成真实路径，避免 ../ 绕过
    return os.path.realpath(os.path.expanduser(p))

def _parse_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default

def _parse_int(v: str | None, default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default


class PathAPIHandler(SimpleHTTPRequestHandler):
    server_version = "PathAPIHTTP/0.1"

    # 由 handler_factory 注入
    allow_prefixes: list[str] = []

    _IMAGE_EXTS = {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"
    }

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/text":
            self._handle_api_text(parsed)
            return
        if parsed.path == "/api/ls":
            self._handle_api_ls(parsed)
            return
        if parsed.path == "/api/csv":
            self._handle_api_csv(parsed)
            return
        if parsed.path == "/api/bin":
            self._handle_api_bin(parsed)
            return
        # 静态资源/HTML
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/append":
            self._handle_api_append(parsed)
            return
        self._send_text(HTTPStatus.NOT_FOUND, f"not found: {parsed.path}")

    def _read_path_param(self, parsed) -> str | None:
        q = parse_qs(parsed.query or "", keep_blank_values=True)
        path = (q.get("path", [None])[0] or "").strip()
        if not path:
            return None
        return path

    def _read_query_params(self, parsed) -> dict[str, str | None]:
        q = parse_qs(parsed.query or "", keep_blank_values=True)
        def _get(k: str) -> str | None:
            v = q.get(k, [None])[0]
            if v is None:
                return None
            s = str(v).strip()
            return s
        return {
            "recursive": _get("recursive"),
            "limit": _get("limit"),
        }

    def _is_allowed(self, real: str) -> bool:
        # allow_prefixes 为空时：默认禁止（防止误开全盘）
        if not self.allow_prefixes:
            return False
        for prefix in self.allow_prefixes:
            if real == prefix or real.startswith(prefix + os.sep):
                return True
        return False

    def _send_text(self, code: HTTPStatus, text: str, *, content_type: str = "text/plain; charset=utf-8") -> None:
        data = text.encode("utf-8", errors="replace")
        self.send_response(int(code))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, code: HTTPStatus, data: bytes, *, content_type: str) -> None:
        self.send_response(int(code))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body_bytes(self, max_bytes: int = 1024 * 1024) -> bytes:
        try:
            n = int(self.headers.get("Content-Length") or "0")
        except Exception:
            n = 0
        n = max(0, min(n, max_bytes))
        if n == 0:
            return b""
        return self.rfile.read(n)

    def _handle_api_text(self, parsed) -> None:
        raw_path = self._read_path_param(parsed)
        if not raw_path:
            self._send_text(HTTPStatus.BAD_REQUEST, "missing query param: path")
            return

        real = _safe_realpath(raw_path)
        if not self._is_allowed(real):
            self._send_text(HTTPStatus.FORBIDDEN, f"forbidden: {real}")
            return
        if not os.path.exists(real):
            self._send_text(HTTPStatus.NOT_FOUND, f"not found: {real}")
            return
        if not os.path.isfile(real):
            self._send_text(HTTPStatus.BAD_REQUEST, f"not a file: {real}")
            return

        try:
            with open(real, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, f"read failed: {real}\n{e}")
            return

        self._send_text(HTTPStatus.OK, text, content_type="text/plain; charset=utf-8")

    def _handle_api_ls(self, parsed) -> None:
        raw_path = self._read_path_param(parsed)
        if not raw_path:
            self._send_text(HTTPStatus.BAD_REQUEST, "missing query param: path")
            return

        params = self._read_query_params(parsed)
        recursive = _parse_bool(params.get("recursive"), default=False)
        limit = _parse_int(params.get("limit"), default=20000)
        limit = max(1, min(limit, 200000))

        real = _safe_realpath(raw_path)
        if not self._is_allowed(real):
            self._send_text(HTTPStatus.FORBIDDEN, f"forbidden: {real}")
            return
        if not os.path.exists(real):
            self._send_text(HTTPStatus.NOT_FOUND, f"not found: {real}")
            return
        if not os.path.isdir(real):
            self._send_text(HTTPStatus.BAD_REQUEST, f"not a directory: {real}")
            return

        files: list[str] = []
        try:
            if recursive:
                for root, dirnames, filenames in os.walk(real):
                    # skip hidden dirs
                    dirnames[:] = [d for d in dirnames if d and not d.startswith(".")]
                    for fn in filenames:
                        if not fn or fn.startswith("."):
                            continue
                        ext = os.path.splitext(fn)[1].lower()
                        if ext not in self._IMAGE_EXTS:
                            continue
                        files.append(os.path.join(root, fn))
                        if len(files) >= limit:
                            break
                    if len(files) >= limit:
                        break
            else:
                for fn in os.listdir(real):
                    if not fn or fn.startswith("."):
                        continue
                    full = os.path.join(real, fn)
                    if not os.path.isfile(full):
                        continue
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in self._IMAGE_EXTS:
                        continue
                    files.append(full)
                    if len(files) >= limit:
                        break
        except Exception as e:
            self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, f"list failed: {real}\n{e}")
            return

        files.sort()
        payload = {
            "root": real,
            "recursive": recursive,
            "count": len(files),
            "limit": limit,
            "files": files,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(HTTPStatus.OK, data, content_type="application/json; charset=utf-8")

    def _handle_api_csv(self, parsed) -> None:
        """
        Read a CSV file and return JSON: {"header":[...], "rows":[{col:val,...}, ...]}.
        This avoids front-end CSV parsing issues for large/quoted fields (e.g. JSON strings).
        """
        raw_path = self._read_path_param(parsed)
        if not raw_path:
            self._send_text(HTTPStatus.BAD_REQUEST, "missing query param: path")
            return

        real = _safe_realpath(raw_path)
        if not self._is_allowed(real):
            self._send_text(HTTPStatus.FORBIDDEN, f"forbidden: {real}")
            return
        if not os.path.exists(real):
            self._send_text(HTTPStatus.NOT_FOUND, f"not found: {real}")
            return
        if not os.path.isfile(real):
            self._send_text(HTTPStatus.BAD_REQUEST, f"not a file: {real}")
            return

        try:
            with open(real, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    payload = {"header": [], "rows": []}
                else:
                    header = [str(h or "").strip() for h in header]
                    rows: list[dict[str, str]] = []
                    for r in reader:
                        if not r:
                            continue
                        obj: dict[str, str] = {}
                        for i, k in enumerate(header):
                            if not k:
                                continue
                            obj[k] = "" if i >= len(r) or r[i] is None else str(r[i])
                        rows.append(obj)
                    payload = {"header": header, "rows": rows}
        except Exception as e:
            self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, f"read failed: {real}\n{e}")
            return

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(HTTPStatus.OK, data, content_type="application/json; charset=utf-8")

    def _handle_api_bin(self, parsed) -> None:
        raw_path = self._read_path_param(parsed)
        if not raw_path:
            self._send_text(HTTPStatus.BAD_REQUEST, "missing query param: path")
            return

        real = _safe_realpath(raw_path)
        if not self._is_allowed(real):
            self._send_text(HTTPStatus.FORBIDDEN, f"forbidden: {real}")
            return
        if not os.path.exists(real):
            self._send_text(HTTPStatus.NOT_FOUND, f"not found: {real}")
            return
        if not os.path.isfile(real):
            self._send_text(HTTPStatus.BAD_REQUEST, f"not a file: {real}")
            return

        try:
            with open(real, "rb") as f:
                data = f.read()
        except Exception as e:
            self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, f"read failed: {real}\n{e}")
            return

        self._send_bytes(HTTPStatus.OK, data, content_type=_guess_type(real))

    def _handle_api_append(self, parsed) -> None:
        """
        Append one line to a text/CSV file.
        - POST /api/append?path=ABS_PATH
        - body: JSON {"line": "..."} (UTF-8)
        Security:
        - only allowed under allow-prefix
        - only allow writing to files ending with "_selected.csv"
        """
        raw_path = self._read_path_param(parsed)
        if not raw_path:
            self._send_text(HTTPStatus.BAD_REQUEST, "missing query param: path")
            return

        real = _safe_realpath(raw_path)
        if not self._is_allowed(real):
            self._send_text(HTTPStatus.FORBIDDEN, f"forbidden: {real}")
            return

        # Hard guard: only allow writing to _selected.csv
        if not real.endswith("_selected.csv"):
            self._send_text(HTTPStatus.BAD_REQUEST, f"only _selected.csv is writable: {real}")
            return

        parent = os.path.dirname(real)
        if not parent or not os.path.isdir(parent):
            self._send_text(HTTPStatus.BAD_REQUEST, f"parent dir not found: {parent}")
            return

        body = self._read_body_bytes(max_bytes=1024 * 1024)
        line = ""
        if body:
            ctype = (self.headers.get("Content-Type") or "").lower()
            try:
                if "application/json" in ctype:
                    obj = json.loads(body.decode("utf-8", errors="replace") or "{}")
                    line = str(obj.get("line", "")).strip("\r\n")
                else:
                    line = body.decode("utf-8", errors="replace").strip("\r\n")
            except Exception:
                line = ""

        if not line:
            self._send_text(HTTPStatus.BAD_REQUEST, "missing body line")
            return

        # Normalize: always append as one CSV cell (no commas); caller provides filename.
        # Ensure one trailing newline.
        to_write = (line + "\n").encode("utf-8", errors="replace")
        try:
            with open(real, "ab") as f:
                f.write(to_write)
        except Exception as e:
            self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, f"append failed: {real}\n{e}")
            return

        payload = {"ok": True, "path": real, "bytes": len(to_write)}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(HTTPStatus.OK, data, content_type="application/json; charset=utf-8")


def build_handler(static_dir: str, allow_prefixes: list[str]):
    static_dir = str(Path(static_dir).resolve())
    allow_real = [_safe_realpath(p) for p in allow_prefixes]

    def handler_factory(*args, **kwargs):
        # 重要：SimpleHTTPRequestHandler 会在 __init__ 里立刻进入 handle()，
        # 所以 allow_prefixes 必须在实例初始化前就可用。
        class _H(PathAPIHandler):
            allow_prefixes = allow_real

        return _H(*args, directory=static_dir, **kwargs)

    return handler_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--static-dir", default=os.path.dirname(__file__))
    parser.add_argument(
        "--allow-prefix",
        action="append",
        default=[],
        help="允许读取的路径前缀（可重复传入），例如 ./ 或 Dataset/",
    )
    args = parser.parse_args()

    httpd = ThreadingHTTPServer(
        (args.bind, args.port),
        build_handler(args.static_dir, args.allow_prefix),
    )

    print(
        f"Serving on http://{args.bind}:{args.port}/\n"
        f"- static-dir: {Path(args.static_dir).resolve()}\n"
        f"- allow-prefixes:\n  - " + "\n  - ".join([_safe_realpath(p) for p in args.allow_prefix] or ["(none; ALL FORBIDDEN)"])
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()

