"""Record and assess runs from a repository-connected Dryft submission."""

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from client import Dryft
from package import package

LATENCY_GATE = 1.10
ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"
HISTORY_DIR = Path(__file__).resolve().parent / "runs"
REGISTRY_FILE = Path(__file__).resolve().parent / "candidates" / "registry.json"


def _latency_ratios(shape: dict) -> dict[str, float | None]:
    metrics = shape.get("modelMetrics") or {}
    pairs = {
        "ttft": (metrics.get("ttftMs"), metrics.get("referenceTtftMs")),
        "tpot": (metrics.get("tpotMs"), metrics.get("referenceTpotMs")),
    }
    return {
        name: mine / native if isinstance(mine, (int, float))
        and isinstance(native, (int, float))
        and math.isfinite(mine) and math.isfinite(native)
        and mine > 0 and native > 0 else None
        for name, (mine, native) in pairs.items()
    }


def report(detail: dict) -> bool:
    """Print workload measurements and require a clean, gate-safe result."""
    state = detail.get("state")
    result = detail.get("result") or {}
    shapes = result.get("shapes") or []

    print(f"run {detail.get('id')}: {state}")
    if result.get("score") is not None:
        print(f"score {result['score']:.1f} tok/s (geometric mean)")
    if not shapes:
        print("  no workload results")

    passed = state == "succeeded" and bool(shapes)
    for shape in shapes:
        metrics = shape.get("modelMetrics") or {}
        status = shape.get("caseStatus", "?")
        columns = [f"{shape.get('id', '?'):<10}", f"{status:<13}"]
        if shape.get("metricMs") and metrics.get("referenceMs"):
            speedup = metrics["referenceMs"] / shape["metricMs"]
            columns.append(f"{shape['metricMs']:8.1f} ms  {speedup:5.2f}x native")
        ratios = _latency_ratios(shape)
        for name, ratio in ratios.items():
            if ratio is None:
                columns.append(f"{name} missing")
                passed = False
            else:
                flag = "  OVER GATE" if ratio > LATENCY_GATE else ""
                columns.append(f"{name} {ratio:4.2f}x{flag}")
                if ratio > LATENCY_GATE:
                    passed = False
        if shape.get("tokensPerSecond") is not None:
            columns.append(f"{shape['tokensPerSecond']:7.1f} tok/s")
        print("  " + "  ".join(columns))
        if shape.get("caseMessage"):
            print(f"    {shape['caseMessage']}")
        if status not in ("passed", "succeeded"):
            passed = False

    for label, value in (
        ("failure", result.get("failureMessage") or result.get("failureCode")),
        ("error", detail.get("errorMessage") or detail.get("errorCode")),
        ("not ranked", result.get("rankingReason")),
    ):
        if value:
            print(f"  {label}: {value}")
    return passed


def _engine_revision(engine_dir: Path) -> dict:
    """Identify exact source bytes, including edits not committed to Git."""
    archive = package(engine_dir)
    repo = engine_dir.parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=False,
    )
    diff = subprocess.run(
        ["git", "status", "--porcelain", "--", str(engine_dir)],
        cwd=repo, text=True, capture_output=True, check=False,
    )
    return {
        "gitCommit": revision.stdout.strip() if revision.returncode == 0 else None,
        "engineDirty": bool(diff.stdout.strip()) if diff.returncode == 0 else None,
        "archiveSha256": hashlib.sha256(archive).hexdigest(),
        "archiveBytes": len(archive),
    }


def save_run(detail: dict, submission_id: str, mode: str,
             local_checkout_snapshot: dict,
             history_dir: Path = HISTORY_DIR) -> Path:
    """Persist full API detail and append a compact experiment index."""
    history_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(detail["id"])
    full_path = history_dir / f"{run_id}.json"
    full_path.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n")
    entry = {
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "runId": run_id,
        "submissionId": submission_id,
        "mode": mode,
        "state": detail.get("state"),
        "scoreTokensPerSecond": (detail.get("result") or {}).get("score"),
        "commitSha": detail.get("commitSha"),
        "ranked": (detail.get("result") or {}).get("ranked"),
        "detailFile": full_path.name,
        "localCheckoutSnapshot": local_checkout_snapshot,
        "submissionSourceVerified": False,
    }
    with (history_dir / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
    return full_path


def attempt(client: Dryft, submission_id: str, mode: str, timeout: float,
            engine_dir: Path = ENGINE_DIR, history_dir: Path = HISTORY_DIR) -> bool:
    """Run a submission already created by the connected GitHub repository."""
    if not submission_id:
        raise ValueError("pass the submission ID produced by a connected repository push")
    revision = _engine_revision(engine_dir)
    print(f"local engine {revision['archiveSha256']} ({revision['archiveBytes']} bytes)")
    started = client.start_run(submission_id, mode=mode)
    run_id = started["id"]
    print(f"submission {submission_id}, {mode} run {run_id}; waiting")
    detail = client.wait(run_id, timeout=timeout)
    path = save_run(detail, submission_id, mode, revision, history_dir)
    print(f"saved {path}")
    return report(detail)


def _run_details(history: list[dict]) -> list[dict]:
    if not history and HISTORY_DIR.exists():
        history = [json.loads(path.read_text())
                   for path in HISTORY_DIR.glob("*.json")]
    details = []
    for entry in history:
        if "result" in entry:
            details.append(entry)
        elif entry.get("detailFile"):
            path = HISTORY_DIR / entry["detailFile"]
            if path.exists():
                details.append(json.loads(path.read_text()))
    return details


def plan_next_edit(history: list[dict]) -> str:
    """Suggest one decision from measured runs; never submit or edit source."""
    candidates = json.loads(REGISTRY_FILE.read_text())["candidates"]
    details = _run_details(history)
    by_id = {str(detail.get("id")): detail for detail in details}
    terminal = {"succeeded", "failed", "timed_out", "canceled", "infra_error"}
    ranked = [detail for detail in details
              if detail.get("state") == "succeeded"
              and detail.get("commitSha")
              and (detail.get("result") or {}).get("ranked") is True
              and isinstance((detail.get("result") or {}).get("score"), (int, float))
              and math.isfinite((detail.get("result") or {})["score"])
              and (detail.get("result") or {})["score"] > 0]
    best = max(ranked, key=lambda d: d["result"]["score"], default=None)
    best_label = (f"best ranked {best['result']['score']:.2f} tok/s "
                  f"at {best['commitSha'][:8]}" if best else "no ranked result yet")

    for candidate in candidates:
        run_id = candidate.get("runId")
        if not run_id:
            return f"Stage {candidate['id']} from {candidate['path']}; {best_label}."
        detail = by_id.get(run_id)
        if detail is None or detail.get("state") not in terminal:
            return f"Wait for existing run {run_id} ({candidate['id']}); {best_label}."
        result = detail.get("result") or {}
        code = result.get("failureCode") or detail.get("errorCode")
        if detail.get("state") == "infra_error" or code in {
            "harness_error", "infrastructure_error", "infra_error"
        }:
            return f"Retry infrastructure failure for {candidate['id']} run {run_id}; {best_label}."
        if code == "incorrect_output":
            return f"Rollback {candidate['id']} after incorrect output; {best_label}."
        if code in {"latency_limit", "unstable_timing", "memory_limit", "timeout"}:
            return f"Review {candidate['id']} {code} before another experiment; {best_label}."
        if detail.get("state") != "succeeded" or result.get("ranked") is not True:
            return f"Review {candidate['id']} run {run_id} ({code or detail.get('state')}); {best_label}."
    return f"All registered candidates measured; {best_label}."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for and save an existing Dryft run without starting another"
    )
    parser.add_argument("run_id", help="existing run ID")
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--mode", choices=("official", "public"), default="official")
    parser.add_argument("--timeout", type=float, default=3000)
    args = parser.parse_args()
    snapshot = _engine_revision(ENGINE_DIR)
    detail = Dryft().wait(args.run_id, timeout=args.timeout)
    path = save_run(detail, args.submission_id, args.mode, snapshot)
    print(f"saved {path}")
    if not report(detail):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
