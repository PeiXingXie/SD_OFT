"""
Client for sd3_5_medium_daemon_server.py (JSON Lines over TCP).

Example:
  python BaseModel/SD-3-5-Medium/sd3_5_medium_daemon_client.py \
    --prompt "A capybara holding a sign that reads Hello World" \
    --out outputs/capybara.png
"""

from __future__ import annotations

import argparse
import json
import socket
from typing import Any, Dict, Optional


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=6319)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--prompt", type=str, required=False)
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num_images_per_prompt", type=int, default=None)
    p.add_argument("--out", type=str, required=False)
    p.add_argument("--ping", action="store_true", help="Health check the daemon.")
    return p


def _send(host: str, port: int, timeout: float, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(data)
        # Read until newline
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    if not buf:
        return {"ok": False, "error": "Empty response"}
    line = buf.split(b"\n", 1)[0]
    return json.loads(line.decode("utf-8"))


def main() -> None:
    args = build_argparser().parse_args()

    if args.ping:
        resp = _send(args.host, args.port, args.timeout, {"cmd": "ping"})
        print(json.dumps(resp, ensure_ascii=False))
        return

    if not args.prompt or not args.out:
        raise SystemExit("Need --prompt and --out (or use --ping).")

    payload: Dict[str, Any] = {
        "cmd": "generate",
        "prompt": args.prompt,
        "out": args.out,
        "steps": args.steps,
        "guidance": args.guidance,
    }
    if args.negative_prompt is not None:
        payload["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        payload["seed"] = int(args.seed)
    if args.height is not None:
        payload["height"] = int(args.height)
    if args.width is not None:
        payload["width"] = int(args.width)
    if args.num_images_per_prompt is not None:
        payload["num_images_per_prompt"] = int(args.num_images_per_prompt)

    resp = _send(args.host, args.port, args.timeout, payload)
    print(json.dumps(resp, ensure_ascii=False))

    if not resp.get("ok", False):
        raise SystemExit(resp.get("error", "Unknown error"))


if __name__ == "__main__":
    main()

