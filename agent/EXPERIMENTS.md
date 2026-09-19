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

## V5 measuring; fused MLP staged

V5 native-projection decode fusion was pushed as commit `eb125bb`, producing
submission `bc5a01ed-9c49-4ed2-bf8f-81a699ec142d` and official run
`65e4885e-140d-48b7-a744-2eb5a59c82a4`. It was measuring when recorded.

`agent/candidates/fused_mlp/` stages a separate decode-only MLP experiment on
top of V5. It keeps separate original gate and up BF16 weight modules and
computes both projections from each input tile in one Triton program, without
writing gate/up activations. Two FP32 Tensor Core accumulators round to BF16
projection outputs, then SiLU rounds to BF16 before multiplication with BF16
up output; the BF16 product enters native down_proj. It applies only to
batches up to 16. Prefill and larger-batch paths stay on V5 code. Fixed tiles
are BM16/BN64/BK64 with four warps and two stages. This changes GEMM
reduction order and therefore has higher correctness risk than elementwise
fusion; V3's hidden incorrect output reinforces the need for a remote token
check without proving this candidate wrong. Syntax, diff and package checks
pass (seven source files, 7,495 bytes); no CUDA compile or speed data exists.

The separate residual-plus-RMSNorm fusion stage at
`agent/candidates/residual_fused/` is recorded in the registry as the next
lower-risk experiment. It is owned by another coding agent; this log does not
claim to validate it.

## V5 passed; residual fusion promoted

V5 native-projection decode fusion passed official run
`65e4885e-140d-48b7-a744-2eb5a59c82a4` at commit `eb125bb`, scoring
502.036739 tokens/s (11.7% above V4's 449.552). This is the best ranked
result so far. The passing V5 source remains at
`agent/candidates/fused_native_decode/`.

The residual-plus-RMSNorm fusion candidate was promoted from
`agent/candidates/residual_fused/` to `engine/` for its own official run. Static
syntax, diff and package checks pass (seven source files, 7,590-byte archive).
The staged fused MLP candidate remains unmeasured and has higher numerical
risk because it changes projection GEMM reduction order.

## Residual run; grouped KV sharing on passing V5 staged

Residual fusion was pushed as commit
`b0e72d6c6773a6e4875c794c970ee1c817aa52ba`, producing submission
`a6e768d8-bab6-4fad-a2ab-9c5b0fda4dc0` and official run
`f7f18cf8-8fad-476d-9cda-385f2affb373`. It was validating when recorded.

`agent/candidates/grouped_tc_fused/` starts from passing V5, retaining its
native separate projections, fused Q/K norm/RoPE/KV writes, and fused SwiGLU.
Only its scalar-head GQA decode kernel is replaced with grouped Tensor Core
GQA. A program loads each K/V tile once for four query heads. Prefill remains
native SDPA. Split ranges and masks are unchanged from the standalone grouped
candidate and cover all initialized keys with a device-side current position.
The pinned Triton 3.1 source defines the used `tl.dot(..., out_dtype=...)`,
reshape/split/join and scalar-control APIs; GPU compilation is still required.
Syntax, diff, partition and package checks pass (six source files, 6,999
bytes). The BF16 probability cast and changed reduction order require
teacher-forced token validation. The live residual engine remains untouched.

## Grouped Tensor Core candidate promoted

`agent/candidates/grouped_tc_fused/` was copied to `engine/` for an independent
official run while the residual fusion submission remains immutable and
measuring. Its comparison baseline is passing V5; it changes only decode GQA
to load each KV tile once for four query heads. The residual stage remains at
`agent/candidates/residual_fused/` with its original commit, submission and
run metadata. Live package validation passes (six source files, 6,999 bytes),
as do Python syntax and diff checks. GPU compilation and correctness are still
unknown until this candidate's own run.

The grouped Tensor Core candidate was committed as
`477526dfb1340d7e8f185d2171310ed9c73e39e8`, creating submission
`66b198b9-84c4-40fa-8f66-c347da512fcd` and official run
`fb2a006e-e72b-4ea4-8e39-534c9b45b850` (queued at 20:18:47 UTC).
The residual fusion run was still measuring. No grouped candidate result is
claimed here.

## Residual passed; prefill fusion on residual baseline promoted

Residual-plus-RMSNorm fusion passed official run
`f7f18cf8-8fad-476d-9cda-385f2affb373` at commit
`b0e72d6c6773a6e4875c794c970ee1c817aa52ba`, scoring 517.746928
tokens/s and rank 38. This is the best ranked result so far.

`agent/candidates/prefill_residual_fused/` combines the passing residual
engine and residual norm kernel with the staged prefill Q/K norm, RoPE and KV
write fusion plus prefill SwiGLU. The decode engine loop, residual kernel, GQA
kernel and RMSNorm kernel are byte-for-byte identical to the passing residual
candidate; its attention decode branch keeps the same operations. Thus the
new experiment targets prefill only. It was copied to `engine/` while the
separate grouped Tensor Core run remained measuring. Syntax, diff and package
checks pass (seven source files, 8,116 bytes). Prefill numerical agreement and
performance still require an official run.

## Staged fused MLP on passing residual baseline

`agent/candidates/fused_mlp_residual/` begins from the passing residual source,
not from pending prefill or grouped Tensor Core variants. For decode batches up
to 16, one Triton kernel reads each hidden-state tile once, performs separate
BF16 gate and up weight dots with FP32 accumulators, rounds each projection to
BF16, then computes BF16-rounded SiLU and product for the unchanged native
down projection. It writes no gate/up activation tensors and concatenates no
weights. Prefill and larger-batch behavior stay on the passing residual path.
The residual engine, residual norm, decode fusion and GQA kernels are bytewise
identical to the passing snapshot. Fixed tiles are BM16/BN64/BK64, four warps,
two stages. Pinned Triton 3.1 source supports the BF16 `tl.dot` with FP32
accumulator. Syntax, diff and package checks pass (eight source files, 8,309
bytes). Custom GEMM summation differs from native and is a higher numerical
risk; no GPU compile, token or performance result exists. The earlier V5-based
fused MLP stage is withdrawn in the registry in favor of this independent
residual-baseline experiment.

## Prefill submission ingestion gap

The prefill-plus-residual engine was pushed to GitHub main as commit
`a9f4756d6cc4d476df7c620fd16e41884aa6a6b6`, but the Dryft platform had
not created a submission after approximately three minutes. The repository UI
showed auto deployment enabled and a check delivery for the earlier
`477526d` push. This is an ingestion observation, not an engine run or
performance result. The registry records the prefill commit as
`awaiting_ingest` with no submission or run ID. A later legitimate metadata
push may retrigger the repository hook; match any resulting submission's
source commit before attributing it to this engine.

The metadata retry was ingested as source commit
`9d73378b889ac565fa9adcb30a9305c892f396a1`, creating submission
`2c713878-4fb0-4a23-a710-01bdb93fd7b0` and official run
`3d453ddc-1f50-478a-ad72-efdf5725f9ed` (queued at 20:28:10 UTC). The
original prefill engine push was `a9f4756d6cc4d476df7c620fd16e41884aa6a6b6`;
the run belongs to the retry commit. No result was available when recorded.

## Fused MLP on residual baseline promoted

`agent/candidates/fused_mlp_residual/` was copied to `engine/` for an
independent official run. The live engine's decoder loop, residual norm,
Q/K fusion and scalar GQA files are bytewise identical to the passing
residual source. Prefill remains native, so this tests only the custom
decode gate/up GEMM and SwiGLU path for batches up to 16. The grouped Tensor
Core and prefill submissions are separate immutable runs. Syntax, diff and
package checks pass (eight source files, 8,309 bytes). The custom GEMM
reduction order remains a high numerical risk until its own GPU result.
