"""
SD1.5 local daemon server (keep model alive).

Protocol (JSON Lines):
  Request:
    - {"cmd":"ping"}
    - {"cmd":"generate","prompt":"...","out":"/abs/path.png","steps":30,"guidance":7.5,
       "seed":123,"negative_prompt":"...","height":512,"width":512,"num_images_per_prompt":1}
  Response:
    - {"ok": true, "result": {...}}
    - {"ok": false, "error": "message"}
"""

from __future__ import annotations

import argparse
import json
import socketserver
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from sd1_5 import DEFAULT_SD15_DIR, get_sd15


DEFAULT_PORT = 6315
_GEN_LOCK = threading.Lock()


def _json_dumps(obj: Dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


class SD15RequestHandler(socketserver.StreamRequestHandler):
    model = None

    def handle(self) -> None:
        raw = self.rfile.readline()
        if not raw:
            return
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            self.wfile.write(_json_dumps({"ok": False, "error": "Invalid JSON"}))
            return

        cmd = req.get("cmd")
        if cmd == "ping":
            self.wfile.write(_json_dumps({"ok": True, "result": {"msg": "pong"}}))
            return

        if cmd != "generate":
            self.wfile.write(_json_dumps({"ok": False, "error": f"Unknown cmd: {cmd}"}))
            return

        prompt = req.get("prompt")
        out = req.get("out")
        if not isinstance(prompt, str) or not prompt.strip():
            self.wfile.write(_json_dumps({"ok": False, "error": "Missing/empty prompt"}))
            return
        if not isinstance(out, str) or not out.strip():
            self.wfile.write(_json_dumps({"ok": False, "error": "Missing/empty out"}))
            return

        negative_prompt = req.get("negative_prompt", None)
        steps = int(req.get("steps", 30))
        guidance = float(req.get("guidance", 7.5))
        seed: Optional[int] = req.get("seed", None)
        if seed is not None:
            seed = int(seed)

        extra_kwargs: Dict[str, Any] = {}
        for k in ("height", "width", "num_images_per_prompt"):
            if k in req and req[k] is not None:
                extra_kwargs[k] = int(req[k])

        try:
            out_path = Path(out).expanduser().resolve()
            with _GEN_LOCK:
                saved = self.model.generate_to_file(
                    prompt,
                    out_path,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed,
                    **extra_kwargs,
                )
            self.wfile.write(
                _json_dumps(
                    {
                        "ok": True,
                        "result": {
                            "out": str(saved),
                            "steps": steps,
                            "guidance": guidance,
                            "seed": seed,
                            **extra_kwargs,
                        },
                    }
                )
            )
        except Exception as e:
            self.wfile.write(_json_dumps({"ok": False, "error": str(e)}))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--model_dir", type=str, default=DEFAULT_SD15_DIR)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    torch_dtype = None if args.torch_dtype == "auto" else args.torch_dtype
    model = get_sd15(
        args.model_dir,
        device=args.device,
        torch_dtype=torch_dtype,
    )
    SD15RequestHandler.model = model

    with socketserver.TCPServer((args.host, args.port), SD15RequestHandler) as srv:
        srv.allow_reuse_address = True
        print(
            f"SD1.5 daemon listening on {args.host}:{args.port} "
            f"(device={model.device}, model_dir={model.model_dir})"
        )
        srv.serve_forever()


if __name__ == "__main__":
    main()

