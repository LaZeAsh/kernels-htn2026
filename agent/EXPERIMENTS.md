# Experiment log

## 2026-09-19: first engine candidate

- Baseline archived at `/tmp/dryft-baseline/engine` before the first edit.
- Commit `1dfba6c` contains the first candidate: bundled Triton RMSNorm on
  Qwen's norms plus direct decoder-layer dispatch. It keeps native SDPA and
  `DynamicCache` behavior.
- Repository-created submission: `f747a3b6-de62-42d1-a176-503f8bfff340`.
- Official run: `defeefdc-f71b-4718-80f5-e8a94ae43ebb` succeeded. Score
  216.94 tokens/s, rank 40. Public-shaped workload throughputs reported within
  the official run were 44.6, 154.2, and 720.9 tokens/s. TTFT ratios were
  0.89, 0.80, 0.78; TPOT ratios were 0.77, 0.67, 0.70. Full report is stored
  under `agent/runs/` (ignored by Git).
- Local checks: Python syntax compilation and `git diff --check` passed.
  Local Transformers and CUDA were unavailable, so there was no local model
  execution or token comparison.
- The v1 source is archived under `agent/candidates/fused_native/`. A second
  candidate with fixed KV storage, explicit decode masking, and a captured
  single-token CUDA graph was promoted from `agent/candidates/graphed/` to
  `engine/`. It has not been run remotely yet.

## Platform observations on 2026-09-19

- The bundled CLI needs `DRYFT_API=https://htn.dryft.ai`; its default endpoint
  returned HTTP 403.
- Direct archive upload (`POST /submissions`) returned HTTP 405. Submissions
  come from the connected GitHub repository.
- Public runs returned HTTP 422. The live API accepts official mode only,
  with six hidden workloads. Do not treat the earlier public workload table
  as observed live performance.
- Repository `autoDeploy=false`: a push alone did not create a submission.
  The connected app only exposed `kernels-htn2026` from the available repos.
- None of these API observations required or stored a token in this log.

## Next step

Wait for the existing v1 run by ID; do not start a duplicate. Save its full
report and inspect token correctness, TTFT and TPOT ratios, throughput, memory,
and spread. Promote the graph candidate only after v1 is understood, then run
one measured change at a time. The graph candidate currently has only static
syntax checks; CUDA capture, memory use, and BF16 token agreement need remote
validation.

```sh
DRYFT_API=https://htn.dryft.ai python3 agent/loop.py \
  defeefdc-f71b-4718-80f5-e8a94ae43ebb \
  --submission-id f747a3b6-de62-42d1-a176-503f8bfff340
```
