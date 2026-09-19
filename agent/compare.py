"""Compare two saved Dryft run reports without calling the API."""

import argparse
import json
import math
from pathlib import Path


RUNS_DIR = Path(__file__).resolve().parent / "runs"
METRICS = (
    ("TTFT", "ttftMs", "ms", "lower"),
    ("TPOT", "tpotMs", "ms", "lower"),
    ("TPS", "tokensPerSecond", "tok/s", "higher"),
)


def _positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _load(run_id: str, directory: Path) -> dict:
    path = directory / f"{run_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"saved run not found: {path}")
    detail = json.loads(path.read_text())
    if detail.get("id") != run_id:
        raise ValueError(f"run ID in {path} does not match {run_id}")
    return detail


def _public_shapes(detail: dict) -> dict:
    return {shape["id"]: shape for shape in (detail.get("result") or {}).get("shapes", [])
            if shape.get("public") is True and shape.get("id")}


def _metric(shape: dict, key: str):
    if key == "tokensPerSecond":
        return shape.get(key)
    return (shape.get("modelMetrics") or {}).get(key)


def compare(candidate: dict, baseline: dict) -> str:
    lines = []
    for label, detail in (("candidate", candidate), ("baseline", baseline)):
        result = detail.get("result") or {}
        score = result.get("score")
        ranked_score = (f"{score:.6f} tok/s" if result.get("ranked") is True
                        and detail.get("state") == "succeeded" and _positive(score)
                        else "unavailable")
        lines.append(f"{label}: {detail.get('id')}  state={detail.get('state')}  "
                     f"commitSha={detail.get('commitSha') or 'unknown'}  "
                     f"ranked={result.get('ranked')}  score={ranked_score}")
    c_score = (candidate.get("result") or {}).get("score")
    b_score = (baseline.get("result") or {}).get("score")
    if (candidate.get("state") == baseline.get("state") == "succeeded"
            and (candidate.get("result") or {}).get("ranked") is True
            and (baseline.get("result") or {}).get("ranked") is True
            and _positive(c_score) and _positive(b_score)):
        lines.append(f"ranked score: {(c_score / b_score - 1) * 100:+.2f}% (higher is favorable)")
    else:
        lines.append("ranked score: unavailable; no score delta for unranked or failed runs")

    current = _public_shapes(candidate)
    reference = _public_shapes(baseline)
    for shape_id in sorted(current.keys() | reference.keys()):
        lines.append(f"{shape_id}:")
        c_shape, b_shape = current.get(shape_id), reference.get(shape_id)
        if c_shape is None or b_shape is None:
            lines.append("  missing matched public workload; deltas unavailable")
            continue
        for label, key, unit, direction in METRICS:
            c_value, b_value = _metric(c_shape, key), _metric(b_shape, key)
            if not (_positive(c_value) and _positive(b_value)):
                lines.append(f"  {label}: unavailable (missing, zero, or nonfinite metric)")
                continue
            delta = ((b_value - c_value) / b_value if direction == "lower"
                     else (c_value - b_value) / b_value) * 100
            lines.append(f"  {label}: {c_value:.3f} vs {b_value:.3f} {unit}; "
                         f"advantage {delta:+.2f}% (positive favors candidate)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_run_id")
    parser.add_argument("baseline_run_id")
    args = parser.parse_args()
    try:
        candidate = _load(args.candidate_run_id, RUNS_DIR)
        baseline = _load(args.baseline_run_id, RUNS_DIR)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(compare(candidate, baseline))


if __name__ == "__main__":
    main()
