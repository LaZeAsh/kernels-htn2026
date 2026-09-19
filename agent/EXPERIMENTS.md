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

## Grouped GQA passed; grouped GQA plus residual staged

Grouped Tensor Core GQA on the V5 baseline passed official run
`fb2a006e-e72b-4ea4-8e39-534c9b45b850` at commit
`477526dfb1340d7e8f185d2171310ed9c73e39e8`, scoring 709.857148
tokens/s at rank 33. It is the best ranked result so far. The prefill run
`3d453ddc-1f50-478a-ad72-efdf5725f9ed` was measuring, and the separate
fused MLP run `18b4c0e9-56ec-48fc-9a66-cc25c98475a8` was queued from
commit `83b962a410e16123fa66df618aaa60b0154055e3` and submission
`4807f5eb-5aee-4529-8b1c-872fb75369ba`.

`agent/candidates/combined_tc_residual/` combines two passing changes:
residual fusion from V6 and grouped GQA from V7. The engine loop and residual
kernel are bytewise the passing V6 files; attention and grouped GQA kernel
are bytewise the passing V7 files. Other shared kernels match both sources.
It contains no pending prefill or fused MLP changes. Syntax, diff and package
checks pass (seven source files, 7,816 bytes). Their interaction still needs
an independent GPU correctness and speed result. It remains staged while the
pending runs complete; the live engine is untouched.

## Prefill passed; grouped GQA plus prefill/residual staged

Prefill fusion on the residual baseline passed official run
`3d453ddc-1f50-478a-ad72-efdf5725f9ed` at actual source commit
`9d73378b889ac565fa9adcb30a9305c892f396a1`, scoring 537.320345
tokens/s. Public-shaped TTFT was 15.984, 131.567 and 118.139 ms versus the
residual baseline's 20.613, 162.26 and 151.397 ms; TPOT was 6.301, 11.329
and 7.863 ms. Grouped Tensor Core GQA remains the best ranked result at
709.857148 tokens/s. The fused MLP run
`18b4c0e9-56ec-48fc-9a66-cc25c98475a8` began measuring at 20:36:12 UTC.

`agent/candidates/combined_tc_prefill_residual/` starts from passing prefill
plus residual fusion. Its engine loop, fused prefill/SiLU kernel and residual
norm kernel are bytewise identical to that source. The only attention adapter
change replaces scalar GQA in decode with passing grouped Tensor Core GQA;
the prefill branch is unchanged. The grouped kernel is bytewise the passing
V7 kernel. Syntax, diff and package checks pass (seven source files, 8,342
bytes); the combination still needs its own correctness and speed run.

The separate `agent/candidates/qkv_split_fused/` stage from another coding
agent is recorded as unmeasured with high numerical risk. Root reviewed its
basic indexing; it has no GPU correctness result. Neither stage was promoted
while the MLP run was pending.

## Candidate registry audit while MLP measures

The original V4-based grouped Tensor Core stage is withdrawn because the
V5-based grouped Tensor Core candidate passed. The unsubmitted grouped-plus-
residual stage is withdrawn because the passing prefill change has been added
in `combined_tc_prefill_residual`. Earlier measured V1/V2 entries remain
recorded as passed, with their run IDs and scores preserved in full run JSON.
The decision helper now waits for the current fused MLP run; if that run
succeeds and ranks, its next staged suggestion is
`combined_tc_prefill_residual`. The QKV split stage remains a later,
unmeasured, high-risk option. No engine source changed in this audit.

## Fused MLP result and next combined candidate

The fused MLP on the passing residual baseline succeeded and ranked in official
run `18b4c0e9-56ec-48fc-9a66-cc25c98475a8` at commit
`83b962a410e16123fa66df618aaa60b0154055e3`, scoring 514.843935
tokens/s. Correctness passed, but the score is below the residual baseline's
517.746928 tokens/s. Public TPOT was 6.399, 11.492 and 8.001 ms versus
6.410, 11.381 and 7.913 ms. The candidate is rejected for promotion due to
no convincing speed gain; its run and source remain recorded.

`combined_tc_prefill_residual` is now the live engine for the next official
run. It combines the independently passing grouped Tensor Core GQA and
prefill/residual changes; their interaction has not yet been measured. The
custom MLP kernel is absent from the live package. QKV split remains an
unmeasured, higher-risk stage.

## Combined run validating; QKV split next

The combined grouped GQA, prefill and residual candidate was accepted as
submission `a048e10e-1516-455d-8df7-64708351daa2`, official run
`b4768620-111c-41ae-8330-8984b2e7dc15`, source commit
`587f4410734c23806f81d7de44013c141ee9f373`. It began validation at
20:44:32 UTC. Its source remains archived at
`agent/candidates/combined_tc_prefill_residual/`.

The live engine now uses the separate `qkv_split_fused` candidate based on the
passing 709.857148 tokens/s grouped GQA source. It changes the decode Q/K/V
projection path, keeping native prefill and the passing grouped attention
kernel; no result is attributed to it yet. The separate `mlp_split_fused`
variant is registered as a later unmeasured experiment on the same passing
base. It is not merged with QKV split.

## QKV split queued; MLP split promoted separately

QKV split was accepted as submission `9f529391-880f-4259-9bb1-72391c70b685`,
official run `27d7715d-f2ac-48a3-9275-db1c25e0845e`, actual source commit
`575d1f9174b25de287fadb2ff839fd71abee6073`. It was queued when
recorded; no correctness or score is attributed to it yet. The combined
prefill/residual plus grouped attention run remains measuring. Both snapshots
are retained under `agent/candidates/`.

The live engine is now the independent `mlp_split_fused` candidate on the
passing 709.857148 tokens/s grouped attention baseline. It changes the
decode gate/up projection schedule while preserving native prefill and native
Q/K/V projections. It does not include the queued QKV or combined changes.
The custom reduction remains unmeasured and numerically higher risk.

## MLP split submission queued

The independent MLP split candidate was accepted from actual source commit
`efa8cea5061e84154011b5ee2f796c35ab130560` as submission
`1413e9db-7f80-4e8d-b3f7-dbf50eb2bccc`, official run
`51f47141-bdad-4c85-8666-a72590bc6a64`. It was queued when recorded;
no correctness or performance result is attributed yet. The live engine source
is unchanged. The combined run is measuring and the QKV split run is queued.

## Combined candidate failed hidden correctness

The grouped GQA plus prefill/residual candidate failed official run
`b4768620-111c-41ae-8330-8984b2e7dc15` at source commit
`587f4410734c23806f81d7de44013c141ee9f373` with `incorrect_output`.
All three public workloads passed, with TTFT 15.814, 130.190 and 121.320 ms
and TPOT 4.917, 5.854 and 5.849 ms. The hidden workload failed before a
ranked score was reported. Preserve the report and do not retry this unchanged
combination. Grouped GQA alone remains best at 709.857148 tokens/s.

Static source comparison found the expected integration only: the passing
prefill/residual attention adapter substitutes the passing grouped GQA decode
call, and the grouped kernel is bytewise identical to the passing V7 source.
The prefill, engine loop and residual kernels match passing prefill/residual
source. No obvious source integration bug was found; the hidden numerical
interaction is unresolved. `combined_tc_residual` is reactivated as the next
unmeasured isolation candidate without prefill fusion. `native_gqa_prefill`
was staged from the failed combined candidate, so it is withdrawn pending a
rebase. QKV split and MLP split remain independent queued experiments.

## Decision helper ordering and passing-base prefill stage

The decision helper now checks accepted pending runs and platform ingestion
before suggesting any unmeasured stage, regardless of registry order. Terminal
failure review still precedes new staging. A synthetic case with an earlier
unmeasured candidate and later pending run returned the pending-run wait;
current history also waits for QKV split rather than suggesting residual
combination. This change does not submit or edit an engine.

`agent/candidates/native_gqa_prefill_base/` is registered as a separate
unmeasured stage based on passing grouped GQA V7. The earlier
`native_gqa_prefill` stage based on the failed combined candidate remains
withdrawn. Neither prefill stage is live.

## QKV split passed; precise PV isolation promoted

QKV split passed and ranked in official run
`27d7715d-f2ac-48a3-9275-db1c25e0845e` at source commit
`575d1f9174b25de287fadb2ff839fd71abee6073`, scoring 745.470297
tokens/s, the new best. Public TPOT was 4.716, 5.363 and 5.492 ms.
The separate MLP split submission is queued and should proceed independently.

`combined_precise_pv` is now live as a targeted retry of the failed combined
prefill/residual plus grouped GQA candidate. Bytewise source comparison shows
only one kernel change: the PV operation uses a high BF16 probability term and
an additional BF16 residual term in a second Tensor Core dot. This tests
whether probability rounding caused the hidden correctness failure. The
prefill, residual, attention adapter and all other source are identical to
the rejected candidate. Its correctness and speed are unmeasured; the second
PV dot costs work. `combined_tc_residual` remains staged as the next isolation
candidate if this fix fails, with no prefill fusion. The independent
`autotuned_mlp_linear` stage on passing grouped GQA V7 is also unmeasured.

## Precise PV queued

The precise PV candidate was accepted from actual source commit
`30026bd98b01fbef288b9f050742003e76716d5c` as submission
`f18285a6-f5e7-48ea-8e24-5aa59658976d`, official run
`4d4bd460-6fe8-4f7b-bff2-a2d4833cdee4`. It was queued when recorded;
no correctness or score is attributed yet. MLP split began measuring at
20:58:56.905 UTC. Live engine source is unchanged.

## Precise PV passed; MLP split failed; more GQA splits live

The precise PV candidate passed official run
`4d4bd460-6fe8-4f7b-bff2-a2d4833cdee4` at actual source commit
`30026bd98b01fbef288b9f050742003e76716d5c`, scoring 720.684275
tokens/s. The result supports the hypothesis that the two-term BF16
probability PV calculation improves numerical agreement for the hidden
workloads; it does not establish which arithmetic difference caused the
earlier failure. It remains below the QKV split best of 745.470297 tokens/s.
The full official report is saved under `agent/runs/`.

MLP split failed official run `51f47141-bdad-4c85-8666-a72590bc6a64` with
`incorrect_output` on a hidden workload. All three public workloads passed.
Its source commit was `efa8cea5061e84154011b5ee2f796c35ab130560`;
retain the report and reject promotion or unchanged retry.

The live engine is now `gqa_more_splits` based on passing QKV split. Source
comparison shows exactly one line changes: the deterministic grouped GQA
split target rises from 128 to 512. This tests additional decode parallelism
using the same full-context formula with a changed reduction partition and
order. QKV projections and prefill remain unchanged. The
`output_proj_fused` stage is separately registered as unmeasured on the QKV
baseline. `combined_tc_residual` remains archived as a later isolation stage.

## GQA scheduling queued; precise PV plus QKV promoted

The increased grouped GQA split count was accepted at actual source commit
`5fc732acf07aeb55b2b4df2096dfbb7e1059c930` as submission
`009185bb-8ec4-4fb0-a007-38782af55a0b`, official run
`4450c9d9-aff3-45c3-b2eb-73634e665af3`. It was queued when recorded;
no correctness or score is attributed yet.

The live engine is now `combined_precise_qkv`, combining the passing precise
PV candidate (720.684275 tokens/s) with the passing QKV split candidate
(745.470297 tokens/s). Relative to precise PV, only the decode attention
adapter dispatch and passing QKV split kernel were added. The precise PV
kernel and prefill/residual engine path are bytewise unchanged. Their
interaction remains unmeasured, so neither passing result is attributed to
this combined engine. The GQA scheduling variant is kept separate.

## Combined queued; native prefill rebase and output projection review

The precise PV plus QKV candidate was accepted at actual source commit
`4e52c3831ad2bd45b5605304effbdcf7b756efd0` as submission
`e503080a-2c33-48b9-ba11-372774648bd4`, official run
`fa4fba3d-2bae-4b98-a285-06ce21f5bb90`. It was queued when recorded,
with no correctness or score yet. `native_gqa_prefill_qkv` is registered as an
unmeasured native GQA prefill experiment based on the passing 745.470297
QKV source. The older 709.857148-base stage is withdrawn as superseded.

Static review of `output_proj_fused` found aligned split sizes for both
4096-column attention output and 9728-column MLP intermediate: each divides
into four 64-wide K tiles. The 40 output tiles cover 2560 columns, row masks
cover batches 1 through 16, and the wrapper falls back to the native layer
for prefill, sequence length other than one, or batch above 16. The reducer
casts the projected sum to BF16 before adding the BF16 residual, matching the
native operation boundary. The four-way FP32 reduction changes sum order,
so hidden token correctness remains a real risk despite these checks.

The stage currently changes both `o_proj` and `down_proj` in each decode
layer. Two separate variants (attention output only, then MLP down only) would
provide more actionable correctness and latency evidence if either path is
tested. This is a recommendation; no stage source or live engine changed.

## Output projection paths isolated

Two independent stages are archived from `output_proj_fused`, both based on
passing QKV split. `output_o_fused` uses the custom split-K projection and
BF16 residual add only for attention `o_proj`; its MLP path calls the passing
native `mlp.forward` and adds the BF16 output to the residual. Conversely,
`output_down_fused` keeps native attention `o_proj` and residual addition and
uses the custom projection only for MLP `down_proj`. Both preserve native
prefill through the original layer forward and use the same decode batch
fallback above 16. The original both-projection stage is retained but
withdrawn from the next experiment sequence for clearer failure isolation.
Neither separate stage has GPU correctness or speed evidence.

## Native GQA prefill on QKV baseline promoted

The precise PV plus QKV combination remains queued under submission
`e503080a-2c33-48b9-ba11-372774648bd4`, run
`fa4fba3d-2bae-4b98-a285-06ce21f5bb90`, source commit
`4e52c3831ad2bd45b5605304effbdcf7b756efd0`. No result is attributed
to it yet.

The live engine is now `native_gqa_prefill_qkv`, based on the passing QKV
split result at 745.470297 tokens/s. Its decode source is bytewise the
passing baseline. The prefill adapter calls PyTorch scaled dot product
attention with `enable_gqa=True`, and prefill is run under a Flash-only SDPA
context. The original native projections, Q/K norm, RoPE and cache updates
remain. This changes prefill attention arithmetic and must pass its own
hidden correctness and latency gates. `cublaslt_decode_qkv` is separately
registered as an unmeasured decode projection experiment.

## Native prefill queued and staged priority

The native GQA prefill on QKV baseline was accepted at actual source commit
`df8e307e46ae33c0c9f246415b492ae70e510a3f`, submission
`f6935a08-6098-4563-afa1-f9ed68f87bdd`, official run
`94c75a6f-2519-4151-aecf-a03c2b636703`. It was queued when recorded;
no correctness or score is attributed. The earlier grouped GQA plus residual
isolation stage is withdrawn from scheduling because the precise PV repair
passed hidden correctness; its source remains archived.

After pending runs and terminal failures are handled, the decision helper
uses explicit staged priorities: cuBLASLt decode QKV first, then MLP down
projection only, then attention output projection only, then autotuned MLP
linear. Priority does not bypass waiting for already accepted runs or their
failure review. No engine source changed.

## Combined precise PV plus QKV leads; cuBLASLt scheduling live

The precise PV plus QKV combination passed official run
`fa4fba3d-2bae-4b98-a285-06ce21f5bb90` at source commit
`4e52c3831ad2bd45b5605304effbdcf7b756efd0`, scoring 769.789834
tokens/s, the new best. The increased grouped GQA split count also passed
run `4450c9d9-aff3-45c3-b2eb-73634e665af3` at commit
`5fc732acf07aeb55b2b4df2096dfbb7e1059c930`, scoring 745.890289
tokens/s, only a small gain over its 745.470297 QKV baseline. Independent
isolation tests continue from the original QKV baseline. Native GQA prefill
is measuring.

The live engine is now `cublaslt_decode_qkv`. Relative to the passing QKV
baseline, its only source change sets PyTorch's preferred BLAS library to
cuBLASLt during decode warmup and CUDA graph capture, restoring the previous
preference afterward. This is a scheduling experiment for native Linear GEMMs
in the captured decode step; it may affect more than Q/K/V projections.
No correctness or speed result is attributed to this stage yet.

## cuBLASLt queued; isolated MLP down projection live

The cuBLASLt graph capture scheduling candidate was accepted at source commit
`3eab3f69e265bd351f691aa3997a8630183ffd08`, submission
`9e0fcaf9-2686-47af-9c58-6b3e5d4fe8fa`, official run
`3c47d87f-2030-4c0e-962c-4c03c01abcd2`. It was queued when recorded;
no score or correctness is attributed yet.

The live engine is now `output_down_fused`, an independent candidate on the
passing QKV baseline. Only decode MLP `down_proj` and its residual add use
the custom split-K projection. Attention `o_proj` remains the native Linear,
and prefill follows the native layer path. The passing combined precise PV
plus QKV snapshot remains archived separately at 769.789834 tokens/s.

## Native GQA prefill failed; output down queued

The native GQA prefill candidate failed official run
`94c75a6f-2519-4151-aecf-a03c2b636703` with `incorrect_output` on a
hidden workload. All public workloads passed at 194.590, 408.422 and
2500.318 tokens/s. The staged source and full run report remain archived;
do not retry it unchanged. Its separate two-term PV repair is in preparation
and has no result yet.

The isolated output down projection candidate was accepted at actual source
commit `4a46c5ab4de8c8f498275d24464b1ea58c983641`, submission
`a0f7d5c7-3daa-47b0-a54d-64d0bfa23761`, official run
`32b6cfd5-b34c-4295-bdd8-7ac129b4618b`. It was queued when recorded.
The cuBLASLt scheduling run began measuring at 21:38:23 UTC. The separate
`transposed_mlp_qkv` and `parallel_mlp_qkv` stages are registered unmeasured
on the passing QKV baseline. Live engine source is unchanged.

## Native prefill plus precise PV repair promoted

The live engine is now `native_gqa_prefill_precise_qkv`, a targeted follow-up
to the hidden-failed native GQA prefill run. Relative to that archived source,
only `kernels/grouped_tc.py` differs: it is bytewise the two-term BF16
probability PV kernel from the passing precise PV plus QKV combination.
The direct native GQA prefill adapter, Flash-only context, QKV projection,
and other source remain unchanged. This tests whether the passing PV change
restores hidden correctness in the native prefill variant; no result is yet
attributed. The queued output down projection candidate remains archived
with its existing run IDs. `transposed_mlp_qkv` remains staged separately.

## Precision repair queued; next staged priority

The native GQA prefill plus precise PV candidate was accepted from actual
source commit `0ddc22194bf691233885fc8c317c02cdb0db10be`, submission
`2704f25c-4d9a-431f-963a-e2a4c9097cfa`, official run
`43987bca-d75b-4ea6-97e3-1e7a84cff71f`. It was queued when recorded;
no result is attributed. Live engine source remains unchanged.

After accepted runs are handled, staged priority is parallel MLP on the
passing QKV baseline, transposed MLP, attention output projection only,
then compiled MLP linear. The already accepted cuBLASLt and output down
candidates keep their run metadata and do not compete for staging priority.

## Autotuned fused MLP staged

`autotuned_fused_mlp_precise_qkv` is registered as an unmeasured stage on the
passing precise PV plus QKV baseline at 769.789834 tokens/s. The reviewed
source changes decode SwiGLU for batches up to 16, retaining the original
BF16 projection and activation boundaries. Four fixed Triton tile choices
vary BN (64 or 128) and BK (64 or 128) with split four and two stages.
One batch-specific choice is shared across layers after a warmup benchmark.
GPU correctness, performance, and the load plus warmup budget are unmeasured.
Its staged priority is 25, after parallel and transposed MLP and before
attention output projection and compiled MLP. Live engine and pending runs
remain unchanged.

## cuBLASLt scheduling passed

The cuBLASLt graph capture scheduling candidate passed official run
`3c47d87f-2030-4c0e-962c-4c03c01abcd2` at source commit
`3eab3f69e265bd351f691aa3997a8630183ffd08`, scoring 775.994986
tokens/s, the new best. This is about 4.10% above its passing QKV baseline
at 745.470297 tokens/s. The full report is saved under `agent/runs/`.
The staged source remains archived; live engine source was not changed by
recording this result.

## Precise PV plus QKV with cuBLASLt promoted

The live engine is now `combined_precise_qkv_cublaslt`. Relative to the
passing precise PV plus QKV candidate (769.789834 tokens/s), its only source
change sets PyTorch's preferred BLAS library to cuBLASLt during decode graph
warmup and capture, then restores the prior setting. That capture preference
passed independently on the QKV baseline at 775.994986 tokens/s. Their
interaction remains unmeasured. The native prefill plus precise PV repair
remains queued under run `43987bca-d75b-4ea6-97e3-1e7a84cff71f` and
submission `2704f25c-4d9a-431f-963a-e2a4c9097cfa`.

`output_down_norm_precise_qkv` is registered unmeasured at low priority 50,
behind the isolated down projection result. Its custom down projection and
norm take two kernel launches, the same count as the passing 769.789834
baseline's separate down projection and norm. No launch-count improvement is
claimed. The comparison CLI and documentation remain outside the engine.

## Combined Lt queued; public decode comparison

The precise PV plus QKV with cuBLASLt candidate was accepted from source
commit `f94a42d448b9e702aa2b92f9902ad166e740fe12` as submission
`364b7917-6191-4291-9312-b8fde89218e1`, official run
`d5421685-4526-4ef7-862d-7c32abaa8c13`. It was queued when recorded;
no result is attributed yet. Live engine source remains unchanged.

The passing cuBLASLt scheduling run (775.994986 tokens/s) compared with its
QKV baseline (745.470297 tokens/s) shows public TPOT improvements of 2.85%,
2.07% and 2.69%: 4.581 vs 4.716 ms, 5.252 vs 5.363 ms, and 5.344 vs
5.492 ms. Public TTFT also varied, but prefill source was unchanged, so that
variation is not credited to the decode-only scheduling change. The full
comparison can be reproduced with `agent/compare.py` using run IDs
`3c47d87f-2030-4c0e-962c-4c03c01abcd2` and
`27d7715d-f2ac-48a3-9275-db1c25e0845e`.

## Output down passed; parallel MLP rebased on cuBLASLt

The isolated output down projection candidate passed official run
`32b6cfd5-b34c-4295-bdd8-7ac129b4618b` at source commit
`4a46c5ab4de8c8f498275d24464b1ea58c983641`, scoring 759.568012
tokens/s. This is 1.89% above its 745.470297 QKV baseline and below the
775.994986 cuBLASLt best. Public TPOT improved 0.32%, 1.45% and 1.18%
relative to QKV. It remains a passing isolated memory scheduling change.

The live engine is now `parallel_mlp_cublaslt`, based on the passing cuBLASLt
QKV candidate at 775.994986 tokens/s. Relative to that source, only the
auxiliary stream decode MLP path differs. Relative to the earlier parallel
MLP stage, only the passing cuBLASLt preference around graph warmup and
capture was added. Native prefill and the B>16 fallback remain. The old
parallel stage is archived but withdrawn as superseded. CUDA graph stream
interaction has no official correctness or speed result yet.

## Parallel Lt queued; transposed MLP and LM head staged

The parallel MLP plus cuBLASLt candidate was accepted at actual source
commit `61d4f7aa8fc64deff8049f618590aff0798a545a`, submission
`d2b01ba9-f8be-486a-af7d-c4089cdfb0d8`, official run
`550ccc14-8940-4090-96dd-e0b4b775d7e6`. It was queued when recorded;
no score or correctness is attributed yet. Live source is unchanged.

`transposed_mlp_cublaslt` is staged at priority 20 on the passing cuBLASLt
QKV baseline; the older transposed MLP stage on the QKV baseline is archived
and withdrawn as superseded. `fused_lm_head_cublaslt` is staged at priority
30, using full-vocabulary logits, a BF16 logit boundary and two-level argmax
with the original minimum-index tie behavior. Its autotune and custom dot
reduction are unmeasured. Later priorities are attention output projection
only (40), compiled MLP linear (50), and down projection plus norm (60).
Neither new stage is live.

## Native prefill precision repair passed

The native GQA prefill plus precise PV repair passed official run
`43987bca-d75b-4ea6-97e3-1e7a84cff71f` at source commit
`0ddc22194bf691233885fc8c317c02cdb0db10be`, scoring 774.449251
tokens/s. This is below the cuBLASLt best at 775.994986 tokens/s. The
successful hidden correctness result supports the probability-rounding
hypothesis for the earlier failed native prefill variant, but does not prove
its causal mechanism.

A read-only comparison against the failed native prefill run has no ranked
score delta because that run was unranked. Public TPOT was 4.647 vs 4.622 ms,
5.354 vs 5.254 ms, and 5.486 vs 5.346 ms; the repair was 0.54%, 1.89% and
2.63% slower in those public cases. The comparison does not establish a
hidden speed delta. The lossless MLP stage remains unregistered pending the
requested source review. Live engine and accepted run metadata are unchanged.

## Combined Lt leads; parallel Lt slower; transposed MLP live

The precise PV plus QKV with cuBLASLt candidate passed official run
`d5421685-4526-4ef7-862d-7c32abaa8c13` at source commit
`f94a42d448b9e702aa2b92f9902ad166e740fe12`, scoring 776.343294
tokens/s, a marginal new best. The parallel MLP plus cuBLASLt candidate also
passed correctness in run `550ccc14-8940-4090-96dd-e0b4b775d7e6` at commit
`61d4f7aa8fc64deff8049f618590aff0798a545a`, scoring 757.773532
tokens/s. It is rejected for promotion because it is slower than its
775.994986 cuBLASLt baseline. Full run JSON files are saved under
`agent/runs/`, with actual source commits matching the registry.

The live engine is now `transposed_mlp_cublaslt`, based on the passing
cuBLASLt QKV source. Only decode gate/up physical weight layout and Linear
calls differ; native weights remain for prefill. It adds approximately
3.34 GiB of BF16 gate/up copies across the model. Official correctness,
memory and speed for this stage remain unmeasured.

## Flash GQA prefill on current best promoted

The transposed MLP plus cuBLASLt candidate was accepted at actual source
commit `6ebb9c4eed588daf0906b4b1ba7b6977631bcf4c`, submission
`1d06a140-919c-4d88-8318-27dca99e6c94`, official run
`0ed08d2b-ad1b-4993-a015-bcc4030c4d19`. It was validating when
recorded; no result is attributed yet.

The live engine is now `fused_prefill_gqa_precise_qkv_cublaslt`, based on the
passing 776.343294 tokens/s precise PV plus QKV with cuBLASLt candidate.
Only the prefill attention call uses direct grouped GQA Flash SDPA instead
of the Transformers interface, with a Flash-only context around prefill.
Its decode source and fused prefill projections remain unchanged. Earlier
native GQA prefill plus precise PV passed independently, but this composition
still needs official correctness and speed measurement.

`lossless_mlp_precise_qkv` is registered unmeasured at priority 80. Luna's
static source review found no bit or stride blocker; a GPU bit self-test,
all-weight validation, and official correctness remain unrun. It is staged
separately and not part of the live engine.

## Flash prefill queued; autotuned MLP on current best live

The Flash grouped GQA prefill candidate was accepted from actual source
commit `c4bf8b495be921fa3444a04877fa22a90b33428f`, submission
`30653b0f-997d-43bd-a292-e5d2c6a53ff4`, official run
`5025ba67-5819-441f-bd93-bb8f032de6da`. It was queued when recorded;
no result is attributed yet.

The live engine is now `autotuned_fused_mlp_precise_qkv_cublaslt` on the
passing 776.343294 tokens/s precise PV/QKV/cuBLASLt baseline. Relative to
that source, only decode SwiGLU uses the staged split-K gate/up projection
kernel for batches up to 16. Its four fixed tile choices and BF16 output
boundaries are unchanged from the earlier staged version. The older non-Lt
MLP stage is retained but withdrawn as superseded.

`fused_lm_head_precise_qkv_cublaslt` is separately registered unmeasured at
priority 30 on the same passing baseline. The older LM head stage on the
745.470297 baseline is retained but withdrawn. Neither newly composed path
has an official GPU result yet.
