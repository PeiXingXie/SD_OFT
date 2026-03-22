"""
GPU burn / keep-alive loop for SD1.5 daemon.
"""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_PORT = 6315


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--timeout", type=float, default=3600.0)

    p.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars")
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=None, help="If set, base seed; each iter uses seed+iter.")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num_images_per_prompt", type=int, default=None)

    p.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "test" / "gpu_burn.png"),
        help="Overwritten each iteration by default to avoid disk growth.",
    )
    p.add_argument("--repeat", type=int, default=0, help="0 means run forever.")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--print_every", type=int, default=1)
    return p


def _send(host: str, port: int, timeout: float, payload: Dict[str, Any]) -> Dict[str, Any]:
    import json

    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(data)
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

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    i = 0
    while True:
        if args.repeat and i >= args.repeat:
            break

        seed: Optional[int] = None
        if args.seed is not None:
            seed = int(args.seed) + i

        payload: Dict[str, Any] = {
            "cmd": "generate",
            "prompt": args.prompt,
            "out": str(out_path),
            "steps": int(args.steps),
            "guidance": float(args.guidance),
        }
        if args.negative_prompt is not None:
            payload["negative_prompt"] = args.negative_prompt
        if seed is not None:
            payload["seed"] = seed
        if args.height is not None:
            payload["height"] = int(args.height)
        if args.width is not None:
            payload["width"] = int(args.width)
        if args.num_images_per_prompt is not None:
            payload["num_images_per_prompt"] = int(args.num_images_per_prompt)

        t0 = time.time()
        resp = _send(args.host, args.port, args.timeout, payload)
        dt = time.time() - t0

        if not resp.get("ok", False):
            raise SystemExit(resp.get("error", "Unknown error"))

        if args.print_every > 0 and (i % args.print_every == 0):
            print(f"[iter={i}] dt={dt:.2f}s out={out_path}")

        i += 1
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()

