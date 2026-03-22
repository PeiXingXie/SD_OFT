"""
SD3.5 Medium local daemon server.

Goal: keep the model instance alive in a long-running process, so you can call it
multiple times from shell commands without re-loading weights each time.

Protocol (JSON Lines):
  Request (one line JSON):
    - {"cmd":"ping"}
    - {"cmd":"generate","prompt": "...", "out": "/abs/path.png", "steps":40, "guidance":4.5,
       "seed":123, "negative_prompt":"..."}
  Response (one line JSON):
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

from sd3_5_medium import DEFAULT_SD35_MEDIUM_DIR, get_sd35_medium


_GEN_LOCK = threading.Lock()


def _json_dumps(obj: Dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


class SD35RequestHandler(socketserver.StreamRequestHandler):
    # Set by main() after model is loaded:
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
        steps = int(req.get("steps", 40))
        guidance = float(req.get("guidance", 4.5))
        seed: Optional[int] = req.get("seed", None)
        if seed is not None:
            seed = int(seed)

        extra_kwargs: Dict[str, Any] = {}
        # Allow a small set of common generation kwargs for compute/load control.
        for k in ("height", "width", "num_images_per_prompt"):
            if k in req and req[k] is not None:
                extra_kwargs[k] = int(req[k])

        try:
            out_path = Path(out).expanduser().resolve()
            # Serialize access to GPU to avoid overlapping runs by default.
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
                        },
                    }
                )
            )
        except Exception as e:
            self.wfile.write(_json_dumps({"ok": False, "error": str(e)}))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=6319)
    p.add_argument("--model_dir", type=str, default=DEFAULT_SD35_MEDIUM_DIR)
    p.add_argument("--device", type=str, default=None, help='e.g. "cuda", "cuda:0", "cpu"')
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype override (default: auto).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    torch_dtype = None if args.torch_dtype == "auto" else args.torch_dtype
    model = get_sd35_medium(
        args.model_dir,
        device=args.device,
        torch_dtype=torch_dtype,
    )

    # Bind the loaded model to handler class.
    SD35RequestHandler.model = model

    with socketserver.TCPServer((args.host, args.port), SD35RequestHandler) as srv:
        srv.allow_reuse_address = True
        print(
            f"SD3.5 Medium daemon listening on {args.host}:{args.port} "
            f"(device={model.device}, model_dir={model.model_dir})"
        )
        srv.serve_forever()


if __name__ == "__main__":
    main()

