"""
Hard-logic scoring for Pointillism StyleMatch rubric.

This module computes the final 0–100 style score deterministically from the model's
BASE SCORES JSON output produced by `prompt/StyleMatch.py`.

Expected model JSON (base only):
{
  "regions": {
    "foreground": [{"A":0-3,"C":0-3,"T":0-3,"E":0-3, ...}, ...],
    "midground": [...],
    "background": [...]
  },
  "global_scores": {"V":0-3,"K":0-3,"P":0-3}
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


Zone = Literal["foreground", "midground", "background"]


DEFAULT_ZONE_WEIGHTS: dict[Zone, float] = {"foreground": 0.45, "midground": 0.35, "background": 0.20}


@dataclass(frozen=True)
class StyleMatchResult:
    # computed local means
    zone_feature_means: dict[Zone, dict[str, float]]
    image_level_local_scores: dict[str, float]  # A_img, C_img, T_img, E_img
    global_scores: dict[str, int]  # V,K,P
    raw_final_score: float
    final_score_after_caps: float
    applied_caps: list[str]


def parse_style_match_base_json(x: Any) -> dict[str, Any]:
    """
    Parse model output into the expected base payload dict.

    Accepts:
    - dict payload already
    - JSON string (optionally with extra text; first {...} block will be extracted)
    """

    if isinstance(x, dict):
        return x
    if x is None:
        raise ValueError("style_match payload is empty")
    s = str(x).strip()
    if not s:
        raise ValueError("style_match payload is empty string")
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        obj = json.loads(s[i : j + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("failed to parse style_match JSON object")


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _zone_present(regions: Any) -> bool:
    # present if it is a non-empty list
    return isinstance(regions, list) and any(isinstance(r, dict) for r in regions)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _coerce_0_3(v: Any) -> float | None:
    f = _to_float(v)
    if f is None:
        return None
    return _clamp(f, 0.0, 3.0)


def _coerce_int_0_3(v: Any) -> int | None:
    i = _to_int(v)
    if i is None:
        return None
    if i < 0:
        return 0
    if i > 3:
        return 3
    return i


def compute_style_match_score(payload: dict[str, Any], *, zone_weights: dict[Zone, float] | None = None) -> StyleMatchResult:
    """
    Compute final style score with:
    - zone weights foreground=0.45, midground=0.35, background=0.20
    - if a zone is absent (no regions), redistribute its weight proportionally over remaining zones
    - formula:
        S_style = (20*A_img + 15*C_img + 15*T_img + 10*E_img + 15*V + 15*K + 10*P) / 3
    - caps:
        if A_img < 1.0 => final <= 45
        if A_img < 1.5 and C_img < 1.5 => final <= 65
        if P == 0 => final <= 55
    """

    zone_weights = dict(zone_weights or DEFAULT_ZONE_WEIGHTS)
    regions_by_zone = (payload or {}).get("regions", {}) or {}
    global_scores_in = (payload or {}).get("global_scores", {}) or {}

    # Parse global scores
    V = _coerce_int_0_3(global_scores_in.get("V"))
    K = _coerce_int_0_3(global_scores_in.get("K"))
    P = _coerce_int_0_3(global_scores_in.get("P"))
    if V is None or K is None or P is None:
        raise ValueError("global_scores must include integer keys V, K, P")

    # Determine zone presence and redistribute weights if needed
    present: list[Zone] = []
    for z in ["foreground", "midground", "background"]:
        rz = regions_by_zone.get(z, None)
        if _zone_present(rz):
            present.append(z)  # type: ignore[arg-type]
    if not present:
        raise ValueError("regions must include at least one zone with non-empty region list")

    missing = [z for z in ["foreground", "midground", "background"] if z not in present]  # type: ignore[list-item]
    if missing:
        missing_w = sum(float(zone_weights.get(z, 0.0)) for z in missing)  # type: ignore[arg-type]
        present_w = sum(float(zone_weights.get(z, 0.0)) for z in present)
        if present_w <= 0:
            raise ValueError("invalid zone weights; present zone weights sum to 0")
        for z in present:
            zone_weights[z] = float(zone_weights.get(z, 0.0)) + missing_w * (float(zone_weights.get(z, 0.0)) / present_w)
        for z in missing:
            zone_weights[z] = 0.0  # type: ignore[index]

    # Compute zone feature means (A,C,T,E) by averaging across regions within zone
    feats = ["A", "C", "T", "E"]
    zone_means: dict[Zone, dict[str, float]] = {"foreground": {}, "midground": {}, "background": {}}
    for z in ["foreground", "midground", "background"]:
        rz = regions_by_zone.get(z, []) or []
        for f in feats:
            vals: list[float] = []
            if isinstance(rz, list):
                for r in rz:
                    if not isinstance(r, dict):
                        continue
                    v = _coerce_0_3(r.get(f))
                    if v is None:
                        continue
                    vals.append(v)
            m = _mean(vals)
            # If zone is absent or feature missing, treat mean as 0.0 for aggregation
            zone_means[z][f] = float(m) if m is not None else 0.0  # type: ignore[index]

    # Aggregate zones into image-level local scores
    A_img = sum(zone_weights[z] * zone_means[z]["A"] for z in ["foreground", "midground", "background"])  # type: ignore[index]
    C_img = sum(zone_weights[z] * zone_means[z]["C"] for z in ["foreground", "midground", "background"])  # type: ignore[index]
    T_img = sum(zone_weights[z] * zone_means[z]["T"] for z in ["foreground", "midground", "background"])  # type: ignore[index]
    E_img = sum(zone_weights[z] * zone_means[z]["E"] for z in ["foreground", "midground", "background"])  # type: ignore[index]

    raw = (20.0 * A_img + 15.0 * C_img + 15.0 * T_img + 10.0 * E_img + 15.0 * V + 15.0 * K + 10.0 * P) / 3.0

    final = float(raw)
    caps: list[str] = []
    if A_img < 1.0:
        final = min(final, 45.0)
        caps.append("cap:A_img<1.0<=45")
    if (A_img < 1.5) and (C_img < 1.5):
        final = min(final, 65.0)
        caps.append("cap:A_img<1.5&C_img<1.5<=65")
    if P == 0:
        final = min(final, 55.0)
        caps.append("cap:P==0<=55")

    return StyleMatchResult(
        zone_feature_means=zone_means,
        image_level_local_scores={"A_img": A_img, "C_img": C_img, "T_img": T_img, "E_img": E_img},
        global_scores={"V": V, "K": K, "P": P},
        raw_final_score=float(raw),
        final_score_after_caps=float(final),
        applied_caps=caps,
    )


def compute_style_band(score_0_100: float) -> str:
    """
    Optional helper: map score to a coarse label.
    """

    s = float(score_0_100)
    if s < 20:
        return "very_weak"
    if s < 40:
        return "weak"
    if s < 60:
        return "moderate"
    if s < 80:
        return "strong"
    return "very_strong"


def build_style_match_stats_dict(res: StyleMatchResult) -> dict[str, Any]:
    """
    Build a JSON-serializable dict for saving "统计评分".
    """

    return {
        "zone_feature_means": res.zone_feature_means,
        "image_level_local_scores": res.image_level_local_scores,
        "global_scores": res.global_scores,
        "raw_final_score": res.raw_final_score,
        "final_score_after_caps": res.final_score_after_caps,
        "applied_caps": list(res.applied_caps),
        "style_band": compute_style_band(res.final_score_after_caps),
    }

