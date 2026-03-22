from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import replace
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sample_expand.config import AndesAuthConfig, load_image_gen_config  # noqa: E402
from sample_expand.factory import build_client  # noqa: E402


def _data_url_to_bytes(data_url: str) -> bytes:
    if not isinstance(data_url, str) or "base64," not in data_url:
        raise ValueError("invalid data url")
    b64 = data_url.split("base64,", 1)[1]
    return base64.b64decode(b64)


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(stem: str) -> str:
    stem = str(stem or "").strip()
    stem = stem.replace("/", "_").replace("\\", "_")
    stem = _SAFE_CHARS.sub("_", stem)
    stem = stem.strip("._-")
    return stem or "untitled"


def should_skip_caption(caption: str) -> bool:
    c = str(caption or "").strip()
    if not c:
        return True
    if c.lower() == "error":
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path (see configs/*.yaml)")
    ap.add_argument("--api-app-id", default="", help="override auth.app_id (instead of ENV/YAML)")
    ap.add_argument("--api-secret-key", default="", help="override auth.secret_key (instead of ENV/YAML)")
    ap.add_argument("--api-verbose", action="store_true", help="print API attempt/retry status for each request")
    ap.add_argument("--quiet", action="store_true", help="suppress per-sample status prints (still prints final summary)")
    ap.add_argument("--log-every", type=int, default=1, help="print status every N in-range samples, default: 1")
    ap.add_argument("--csv", required=True, help="input CSV path")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--ext", default=".png", help="output suffix/extension, default: .png")
    ap.add_argument("--start", type=int, default=0, help="start sample index (0-based, inclusive)")
    ap.add_argument("--end", type=int, default=-1, help="end sample index (0-based, exclusive). -1 means no limit")
    ap.add_argument(
        "--range-on",
        choices=["raw", "valid"],
        default="raw",
        help="apply --start/--end on raw CSV row index (raw) or on valid sample index after skip filters (valid). default: raw",
    )
    ap.add_argument("--id-col", default="id", help="id column name, default: id")
    ap.add_argument("--caption-col", default="caption", help="caption column name, default: caption")
    ap.add_argument(
        "--error-col",
        default="__error",
        help="error column name; if exists and non-empty -> skip row. default: __error",
    )
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    ap.add_argument("--sleep-sec", type=float, default=0.0, help="sleep between requests (seconds)")
    ap.add_argument("--report", default="", help="optional path to write a CSV report")
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

    in_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = str(args.ext or ".png").strip()
    if not ext.startswith("."):
        ext = "." + ext

    report_fp = None
    report_writer: csv.DictWriter | None = None
    if isinstance(args.report, str) and args.report.strip():
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # stream write: overwrite but flush per row so partial results survive interruptions
        report_fp = report_path.open("w", encoding="utf-8", newline="")
        report_writer = csv.DictWriter(
            report_fp,
            fieldnames=["csv_idx", "valid_idx", "id", "status", "reason", "out_path", "image_url"],
        )
        report_writer.writeheader()
        report_fp.flush()

    def write_report(row: dict[str, str]) -> None:
        if report_writer is None or report_fp is None:
            return
        report_writer.writerow(row)
        report_fp.flush()

    n_total = 0
    n_out_of_range = 0
    n_skip = 0
    n_exists = 0
    n_ok = 0
    n_fail = 0
    n_in_range = 0
    valid_idx = -1  # increments only after passing skip-filters (id/error/caption)

    with in_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            n_total += 1
            rid = str((row.get(args.id_col) if row else "") or "").strip()
            caption = str((row.get(args.caption_col) if row else "") or "").strip()

            if not rid:
                n_skip += 1
                # raw-range filtering applies before we even know validity; valid-range ignores missing id rows anyway
                if args.range_on == "raw":
                    if idx < int(args.start or 0) or (int(args.end) >= 0 and idx >= int(args.end)):
                        n_out_of_range += 1
                        continue
                    n_in_range += 1
                log_this = (not args.quiet) and int(args.log_every) > 0 and (n_in_range % int(args.log_every) == 0)
                if log_this:
                    print(f"[SKIP] idx={idx} id=<missing> reason=missing_id", flush=True)
                write_report(
                    {
                        "csv_idx": str(idx),
                        "valid_idx": "",
                        "id": "",
                        "status": "skip",
                        "reason": "missing id",
                        "out_path": "",
                        "image_url": "",
                    }
                )
                continue

            if args.error_col and isinstance(row, dict) and args.error_col in row:
                err_val = str(row.get(args.error_col) or "").strip()
                if err_val:
                    n_skip += 1
                    if args.range_on == "raw":
                        if idx < int(args.start or 0) or (int(args.end) >= 0 and idx >= int(args.end)):
                            n_out_of_range += 1
                            continue
                        n_in_range += 1
                    log_this = (not args.quiet) and int(args.log_every) > 0 and (n_in_range % int(args.log_every) == 0)
                    if log_this:
                        print(f"[SKIP] idx={idx} id={rid} reason={args.error_col}_not_empty", flush=True)
                    write_report(
                        {
                            "csv_idx": str(idx),
                            "valid_idx": "",
                            "id": rid,
                            "status": "skip",
                            "reason": f"{args.error_col} not empty",
                            "out_path": "",
                            "image_url": "",
                        }
                    )
                    continue

            if should_skip_caption(caption):
                n_skip += 1
                if args.range_on == "raw":
                    if idx < int(args.start or 0) or (int(args.end) >= 0 and idx >= int(args.end)):
                        n_out_of_range += 1
                        continue
                    n_in_range += 1
                log_this = (not args.quiet) and int(args.log_every) > 0 and (n_in_range % int(args.log_every) == 0)
                if log_this:
                    print(f"[SKIP] idx={idx} id={rid} reason=empty_or_error_caption", flush=True)
                write_report(
                    {
                        "csv_idx": str(idx),
                        "valid_idx": "",
                        "id": rid,
                        "status": "skip",
                        "reason": "empty/error caption",
                        "out_path": "",
                        "image_url": "",
                    }
                )
                continue

            # sample is "valid" after skip-filters
            valid_idx += 1

            # range selection: raw uses csv idx; valid uses valid_idx (after filters)
            range_idx = idx if args.range_on == "raw" else valid_idx
            if range_idx < int(args.start or 0) or (int(args.end) >= 0 and range_idx >= int(args.end)):
                n_out_of_range += 1
                continue

            n_in_range += 1
            log_this = (not args.quiet) and int(args.log_every) > 0 and (n_in_range % int(args.log_every) == 0)

            out_path = out_dir / f"{safe_filename(rid)}{ext}"
            if out_path.exists() and not args.overwrite:
                n_exists += 1
                if log_this:
                    print(f"[EXISTS] idx={idx} id={rid} out={out_path}", flush=True)
                write_report(
                    {
                        "csv_idx": str(idx),
                        "valid_idx": str(valid_idx),
                        "id": rid,
                        "status": "exists",
                        "reason": "",
                        "out_path": str(out_path),
                        "image_url": "",
                    }
                )
                continue

            if log_this:
                print(f"[RUN] idx={idx} id={rid} -> {out_path}", flush=True)
            cb = None
            if args.api_verbose and log_this:
                cb = lambda s, _idx=idx, _rid=rid: print(f"[{_idx}:{_rid}] {s}", flush=True)  # noqa: E731
            res = client.generate(caption, status_cb=cb)
            if not res.ok or not res.image_base64:
                n_fail += 1
                reason = str(res.error or "unknown error")
                if log_this:
                    print(
                        f"[FAIL] idx={idx} id={rid} attempts={res.attempts} elapsed={res.elapsed_sec:.3f}s reason={reason}",
                        flush=True,
                    )
                write_report(
                    {
                        "csv_idx": str(idx),
                        "valid_idx": str(valid_idx),
                        "id": rid,
                        "status": "fail",
                        "reason": reason,
                        "out_path": str(out_path),
                        "image_url": str(res.image_url or ""),
                    }
                )
                continue

            try:
                out_path.write_bytes(_data_url_to_bytes(res.image_base64))
            except Exception as e:
                n_fail += 1
                if log_this:
                    print(f"[FAIL] idx={idx} id={rid} attempts={res.attempts} reason=write_failed:{e}", flush=True)
                write_report(
                    {
                        "csv_idx": str(idx),
                        "valid_idx": str(valid_idx),
                        "id": rid,
                        "status": "fail",
                        "reason": f"write_failed: {e}",
                        "out_path": str(out_path),
                        "image_url": str(res.image_url or ""),
                    }
                )
                continue

            n_ok += 1
            if log_this:
                print(
                    f"[OK] idx={idx} id={rid} attempts={res.attempts} elapsed={res.elapsed_sec:.3f}s out={out_path}",
                    flush=True,
                )
            write_report(
                {
                    "csv_idx": str(idx),
                    "valid_idx": str(valid_idx),
                    "id": rid,
                    "status": "ok",
                    "reason": "",
                    "out_path": str(out_path),
                    "image_url": str(res.image_url or ""),
                }
            )

            if args.sleep_sec and args.sleep_sec > 0:
                time.sleep(float(args.sleep_sec))

    print(
        json.dumps(
            {
                "total": n_total,
                "out_of_range": n_out_of_range,
                "skip": n_skip,
                "exists": n_exists,
                "ok": n_ok,
                "fail": n_fail,
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
        )
    )
    if report_fp is not None:
        try:
            report_fp.close()
        except Exception:
            pass
        print(f"[OK] report saved: {Path(args.report).expanduser().resolve()}")

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

