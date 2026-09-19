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

## Staged V3: packed projections

`agent/candidates/packed/` combines each layer's Q, K, V projection weights
into one `[6144, 2560]` BF16 parameter and gate/up weights into one
`[19456, 2560]` BF16 parameter. Original projection modules are deleted after
packing, so steady-state parameter memory is unchanged. The adapter performs
one BF16 linear for each pack, splits the BF16 result, then follows pinned
Transformers 4.51.3 Qwen3 norm, RoPE, cache update, SDPA interface, output
projection, activation and residual order. The graph and fixed-cache behavior
is inherited from V2. This is a projection experiment; attention computation
remains the native SDPA adapter.

Static syntax, diff and package checks pass (four source files, 4,326-byte
archive). It has no CUDA correctness or speed result. Packing can change GEMM
reduction order despite preserving BF16 output boundaries, so token validation
is required before use. The V2 engine remains live pending its run.

## Staged V4: direct decode GQA

`agent/candidates/gqa/` inherits V3's packed projections and fixed cache. Its
single-token decode attention reads K/V directly as `[B,8,C,128]` and maps
query head `h` to KV head `h//4`. It masks all cache slots after the device
position, uses FP32 score and online-softmax accumulation, and stores BF16
output. Prefill still uses native causal SDPA. Batches of at least four use
one program per query head; batch one at capacity at least 1024 uses four
fixed split-K programs and a stable FP32 combination. Split choice depends
only on shape, so capture addresses and timing paths are prompt independent.

Wrapper checks BF16/CUDA shapes and cache contiguity. Static syntax/diff/package
checks pass (five source files, 5,686-byte archive). No Triton compilation,
CUDA run, performance or teacher-forced numerical check has been performed
locally. The kernel changes SDPA reduction order and softmax implementation;
this remains a candidate until an official correctness result is obtained.

## V2 run in progress

The CUDA graph candidate was pushed as commit `d664f2a`, creating submission
`57acbad3-afe6-4091-bdfc-d8e4e4ba9516`. Official run
`1e88deb4-e94c-4e51-b029-ec8aa04561ee` succeeded: 346.677 tokens/s and rank 39, versus V1 at 216.94 tokens/s.
Public-shaped TPOT was 7.425, 15.365, and 16.332 ms; TTFT was 28.689,
163.121, and 151.36 ms (native ratios 1.03, 0.80, 0.79). All tokens passed.
The report records source commit `d664f2a35dea36a86a13a529b81b393e3941a7a2`. The explicit
candidate registry is `agent/candidates/registry.json`; `plan_next_edit` reads
terminal run details and proposes the next staged experiment only after this
run finishes. Infrastructure errors call for retry, wrong tokens call for
rollback, and latency or memory failures call for review. It keeps the best
ranked official score using the run's own `commitSha`.

## V3 promotion

After V2 passed, the packed-projection candidate was copied to `engine/` for
its own official run. The V2 graph source remains at
`agent/candidates/graphed/`. V4 direct GQA remains staged and unmeasured.

## Staged V5: decode elementwise fusion

`agent/candidates/fused_decode/` inherits V4 packed projections and direct
GQA. Decode now fuses per-head Q/K RMSNorm, RoPE and the K/V cache write in one
Triton kernel. It normalizes FP32, casts to BF16 before multiplying the gain,
rounds that result to BF16, rounds each RoPE product to BF16, then adds and
rounds to BF16. FMA fusion is disabled. The V value is copied unchanged from
the BF16 packed projection. A second decode-only kernel computes SiLU in FP32,
rounds it to BF16, multiplies the BF16 up projection, and rounds the product
to BF16. Prefill, including a one-token prompt, uses the original packed
projection, norm, RoPE, cache-update, native SDPA and activation paths.

The V5 archive has six source files and is 6,862 bytes. Syntax, diff and
package checks pass. No CUDA compiler, numerical comparison or benchmark run
was available locally; neither V4 nor V5 should be treated as validated.

## V3 failed; native-projection GQA staged

V3 packed projections at commit `790d02580d70dcfae8b550c64dbd62800d929a9f`
failed official run `f449f653-e897-4aaf-bdb3-161852793b18` with
`incorrect_output` on a hidden workload. All public-shaped cases passed and
TPOT improved only slightly (7.046, 15.182, 16.093 ms versus V2's 7.425,
15.365, 16.332 ms). The hidden report does not localize the bad token, so no
specific operation is proven responsible. Best ranked result remains V2 at
346.677 tokens/s. The failed source is preserved at `agent/candidates/packed/`.

A new `agent/candidates/gqa_native/` starts from passing V2. It retains the
original separate Q/K/V and gate/up projections and native causal SDPA
prefill. Only cached decode attention changes to direct full-context GQA. It
was copied into `engine/` for the next official run. Static syntax, diff and
package checks pass (five source files, 5,397-byte archive). No GPU result is
available yet. The earlier packed V4/V5 candidates remain staged but are not
on this correctness baseline.

## V4 measuring; native-projection fusion staged

V4 native-projection direct GQA was pushed as commit `5d5fedb`, creating
submission `6d95e6b0-791e-4953-9e92-a2a73326ca8e` and official run
`8544b784-1c73-41b6-ba40-8552fb99eae8`. It was validating at this entry;
no outcome is assumed.

`agent/candidates/fused_native_decode/` stages a follow-up based on V4. It
passes the three separate native BF16 Q/K/V projection tensors to the fused
Q/K normalization, RoPE and KV-write kernel. SwiGLU consumes separate native
gate and up projection tensors. No packed GEMM weights or source modules are
changed; prefill, including one-token prompts, follows the native projection,
normalization, RoPE, SDPA and MLP order. The stage has six source files and a
6,761-byte package, with syntax/diff/package checks passing. It will remain
unpromoted until V4's correctness and latency are known.

## Staged V6: grouped Tensor Core decode attention

`agent/candidates/grouped_tc/` starts from native-projection V4, not from V5
fusion or the failed packed candidate. Prefill remains native causal SDPA. For
decode, one Triton program handles one KV head and its four query heads,
loading each K/V tile once for those heads. A `[16,128]` padded BF16 Q tile
multiplies a `[128,64]` K tile with FP32 scores. Online softmax statistics stay
FP32; probabilities round to BF16 for the Tensor Core PV dot; partial PV and
split reduction accumulate FP32 before the BF16 output store. Every valid key
is included, with no sparse or approximate context selection.

Split count is fixed by batch and prompt shape: target about 128 CTAs across
`B*8*split`, capped by prompt-length 64-token tiles. Splits cover contiguous,
nonoverlapping key ranges; the last split includes decode growth. Empty blocks
are skipped on the device. Pure-Python partition checks passed for public and
small-prompt shapes. Syntax, diff and package checks pass (five source files,
5,692 bytes). Triton 3.1 compilation, output agreement and speed remain
unverified; V4 remains live while its run completes.

## V4 passed; V5 promoted

V4 native-projection direct GQA passed official run
`8544b784-1c73-41b6-ba40-8552fb99eae8`, scoring 449.552 tokens/s at
rank 39. Public-shaped TPOT was 7.654, 12.797 and 9.395 ms; TTFT was 28.12,
163.91 and 151.55 ms. This is the best ranked result so far. Its source
remains at `agent/candidates/gqa_native/`.

The native-projection elementwise fusion candidate was copied from
`agent/candidates/fused_native_decode/` to `engine/` for its own official run.
It adds decode-only Q/K normalization, RoPE and KV-write fusion plus SwiGLU
fusion, with separate native projection modules. Static syntax, diff and
package checks pass (six source files, 6,774 bytes). Triton compilation and
greedy-token correctness remain unverified until the run. The grouped Tensor
Core candidate remains staged separately.
