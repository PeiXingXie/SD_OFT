from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sample_expand.config import AndesAuthConfig, load_image_gen_config
from sample_expand.factory import build_client


def _data_url_to_bytes(data_url: str) -> bytes:
    if not isinstance(data_url, str) or "base64," not in data_url:
        raise ValueError("invalid data url")
    b64 = data_url.split("base64,", 1)[1]
    return base64.b64decode(b64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path (see configs/*.yaml)")
    ap.add_argument("--api-app-id", default="", help="override auth.app_id (instead of ENV/YAML)")
    ap.add_argument("--api-secret-key", default="", help="override auth.secret_key (instead of ENV/YAML)")
    ap.add_argument("--api-verbose", action="store_true", help="print API attempt/retry status")
    ap.add_argument("--quiet", action="store_true", help="suppress status prints (except errors)")
    ap.add_argument("--prompt", required=True, help="text prompt/caption")
    ap.add_argument("--out", required=True, help="output image path, e.g. out.png")
    ap.add_argument("--dump-json", default="", help="optional path to dump response json")
    args = ap.parse_args()

    cfg = load_image_gen_config(args.config)
    if (args.api_app_id and not args.api_secret_key) or (args.api_secret_key and not args.api_app_id):
        raise SystemExit("Both --api-app-id and --api-secret-key must be provided together.")
    if args.api_app_id and args.api_secret_key:
        cfg = replace(
            cfg,
            auth=AndesAuthConfig(app_id=str(args.api_app_id).strip(), secret_key=str(args.api_secret_key).strip()),
        )
    client = build_client(cfg)
    if not args.quiet:
        print("[RUN] calling api...")
    cb = (lambda s: print(s, flush=True)) if (args.api_verbose and not args.quiet) else None
    res = client.generate(args.prompt, status_cb=cb)

    if not res.ok:
        print(f"[FAILED] {res.error}")
        if res.raw_http:
            print(f"[RAW_HTTP] {json.dumps(res.raw_http, ensure_ascii=False)[:2000]}")
        return 2

    if not res.image_base64:
        print("[FAILED] missing image_base64 in result")
        return 3

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(_data_url_to_bytes(res.image_base64))
    if not args.quiet:
        print(f"[OK] saved: {out_path} (attempts={res.attempts}, elapsed={res.elapsed_sec:.3f}s)")

    if isinstance(args.dump_json, str) and args.dump_json.strip():
        dump_path = Path(args.dump_json).expanduser().resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(res.response or {}, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"[OK] response json dumped: {dump_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

