"""Record and assess runs from a repository-connected Dryft submission."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from client import Dryft
from package import package

LATENCY_GATE = 1.10
ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"
HISTORY_DIR = Path(__file__).resolve().parent / "runs"


def _latency_ratios(shape: dict) -> dict[str, float | None]:
    metrics = shape.get("modelMetrics") or {}
    pairs = {
        "ttft": (metrics.get("ttftMs"), metrics.get("referenceTtftMs")),
        "tpot": (metrics.get("tpotMs"), metrics.get("referenceTpotMs")),
    }
    return {
        name: mine / native if isinstance(mine, (int, float))
        and isinstance(native, (int, float)) and native > 0 else None
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
        if status in ("failed", "error", "timed_out"):
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
             engine_dir: Path = ENGINE_DIR, history_dir: Path = HISTORY_DIR) -> Path:
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
        "detailFile": full_path.name,
        **_engine_revision(engine_dir),
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
    path = save_run(detail, submission_id, mode, engine_dir, history_dir)
    print(f"saved {path}")
    return report(detail)


def plan_next_edit(history: list[dict]) -> str:
    raise NotImplementedError("this is the part you write")
