---
title: "Quantization in LLM Serving: llama.cpp, Attention Backends, and Engines"
date: 2026-09-20
excerpt: "A source-code-oriented exploration of how serving stacks load quantized models, select kernels, and handle weights, activations, and the KV cache."
tags: [LLM Serving, Quantization, Research Notes]
toc: true
published: false
---

> Working draft. The llama.cpp case study, selected vLLM backend integrations, SGLang's KV-cache formats and execution paths, Transformers' quantized cache, MNN's CPU/Metal paths, and selected ExecuTorch export paths are covered below. Standalone backend-library comparisons and our own experiments remain in progress. “TBD” means not yet investigated, rather than unsupported.

## Why look at the implementation?

A low-bit format does not determine how inference computes with it. This post traces the path from stored codes and reconstruction parameters to cache writes, kernel selection, and attention arithmetic. We begin with llama.cpp as an integrated case study, then separate the roles of attention backends, framework adapters, and serving engines.

## Scope and comparison setup

The initial scope is **weight quantization, activation quantization, and KV cache quantization**. For each stack, follow one representative execution path before expanding to other methods or backends.

| Stack | Release / commit | Hardware and backend | Model / checkpoint | Selected quantization path |
| --- | --- | --- | --- | --- |
| llama.cpp | [`a894dae939d4`][llama-blocks] | Generic CPU and selected CUDA paths | Source analysis; no model run | Weights, temporary activations, and persistent KV cache |
| vLLM | [`17e50b9b7610`][vllm-revision] | Selected FlashAttention, FlashInfer, and Triton paths | Source analysis; no model run | Quantization interfaces, KV-cache scales, and attention reads |
| SGLang | [`2f5c9ac43d76`][sg-revision] | Ordinary MHA/GQA pools; selected CUDA adapters | Source analysis; no model run | FP8/MXFP8/FP4 algorithms, storage, attention paths, and allocation |
| Transformers | [`96331a9f93b7`][hf-revision] | Ordinary `QuantizedCache`; Llama attention adapters | Source analysis; no model run | Quanto/HQQ encoding, residual storage, and dense attention inputs |
| MNN | [`67450001627f`][mnn-revision] | Ordinary CPU and Metal attention | Source analysis; no model run | INT8/TQ3/TQ4, growing buffers, and backend-specific arithmetic |
| ExecuTorch | [`12668f8b81c39`][et-revision] | Llama CPU export; selected CUDA, MLX, and Qualcomm paths | Source analysis; no model run | Dynamic/static INT8, TQ4, persistent tensors, and export-time specialization |
| FlashInfer | [`v0.6.18`][fi-xqa-api] | Selected XQA CUDA kernels | Source analysis; no model run | API dtypes versus internal FP8/FP16/BF16 operands |

**Remaining scope:** Pin the remaining external backend-library versions alongside their framework integrations, then specify workloads and hardware for experiments. The FlashInfer kernel check is versioned separately from vLLM's routing analysis; it does not identify the binary installed in a particular deployment. Source analysis alone does not establish performance or accuracy.

## A vocabulary for reading the code

Three precision choices must be distinguished: **storage format** describes the cached or saved codes and metadata; **operand dtype** describes the values supplied to a dot product; **accumulation dtype** describes its running sum. These can differ within one kernel. FP8 is itself floating-point, so “quantized” and “floating-point” are not opposing categories.

**Reconstruction (dequantization)** recovers approximate values from codes and metadata. It may produce a temporary tensor, occur on tiles inside a kernel, or be implemented by applying scale/offset corrections to intermediate results. “Dequantization” alone does not identify which approach is used.

A quantization algorithm chooses codes and parameters; checkpoint and cache formats specify their representation; backend kernels determine their use. Parameter granularity specifies which values share a scale or offset. Offline calibration, load-time conversion, and runtime quantization are separate stages; weight, activation, and KV-cache precision are separate configuration choices.

## Reading order

1. **llama.cpp:** follow block formats, temporary activation quantization, persistent KV storage, and attention dispatch within one codebase.
2. **Backend profiles:** study FlashAttention, Triton, FlashInfer/TRT-LLM, and TurboQuant separately, each through its cache types, storage layout, and attention paths.
3. **vLLM dispatch:** trace how GPU generation, model shape, cache format, and requested features select an adapter and its kernel path.
4. **SGLang:** connect cache algorithms to physical storage and attention paths, then compare allocation and prefix reuse with vLLM.
5. **Transformers:** examine compression behind a dense-tensor attention interface and its memory/computation trade-offs.
6. **MNN:** compare CPU and Metal cache encodings, growing buffers, and integrated attention kernels.
7. **ExecuTorch:** distinguish exported cache state from delegate-specific formats and attention implementations.

## llama.cpp: quantization as a block format

llama.cpp provides a useful starting point because its tensor formats, KV-cache handling, and CPU/CUDA execution paths can be followed within the same project. We first establish that complete path before separating responsibilities across libraries and serving engines.

GGUF stores the model; GGML defines the tensor encodings within it. The classic integer and K-quant families encode weights on **locally uniform grids**, sharing reconstruction parameters across groups. K-quants additionally compress those parameters. The [legacy guide][legacy-guide] and [K-quant guide][k-guide] provide the intuition; the definitions below follow llama.cpp commit `a894dae939d4` (2026-09-20).

### Legacy formats: one scale per block

The five classic weight formats below group **32 weights** together. For Q4/Q5, `_0` reconstructs a centered integer code, while `_1` adds a stored FP16 offset:

$$
\hat w_i =
\begin{cases}
d\,(q_i-2^{b-1}), & \mathrm{Q}b\_0,\\
d\,q_i+m, & \mathrm{Q}b\_1,
\end{cases}
\qquad b\in\{4,5\}.
$$

Here $q_i$ is the unsigned packed code; Q8_0 instead stores signed INT8 values and uses $\hat w_i=dq_i$. Q4_0 uses all 16 codes, decoding to integers $[-8,7]$; its reference quantizer can use a signed scale. The stored offset in Q4_1/Q5_1 is a floating-point value, not an explicit integer zero-point. [Reconstruction code][legacy-dequant].

| Format | FP16 parameters | Packed integer fields | Bytes / block; bits / weight |
| --- | --- | --- | --- |
| Q4_0 | `d` | `qs[16]` | 18; **4.5** |
| Q4_1 | `d, m` | `qs[16]` | 20; **5.0** |
| Q5_0 | `d` | `qs[16] + qh[4]` | 22; **5.5** |
| Q5_1 | `d, m` | `qs[16] + qh[4]` | 24; **6.0** |
| Q8_0 | `d` | signed `qs[32]` | 34; **8.5** |

Array lengths above count bytes. Q5 stores four low bits in `qs` and the fifth bit in `qh`. Sizes follow the [`block_q*` definitions and static assertions][llama-legacy-blocks]; bits per weight include metadata, computed as $8\,\mathrm{sizeof(block)}/32$.

### IQ4_NL: four-bit indices into a fixed codebook

IQ4_NL replaces the uniform integer grid with a **nonuniform scalar codebook**. Each block stores one FP16 scale $d$ and 32 four-bit indices in 16 bytes: 18 bytes total, or **4.5 bits per value**. An index $u_i\in\{0,\ldots,15\}$ selects a reconstruction level rather than directly representing its magnitude:

$$
\hat x_i=d\,C[u_i].
$$

```text
C = [-127, -104, -83, -65, -49, -35, -22, -10,
        1,   13,  25,  38,  53,  69,  89, 113]
```

The encoder fits the scale and selects codebook indices; decoding unpacks each index, looks up its level, and multiplies by $d$. The table is shared across blocks, so only the scale and indices are stored; there is no per-block codebook or zero point. [Structure and table][iq4-nl-format]; [encoding and reconstruction][iq4-nl-code].

**Why these levels?** The original author applied K-means to model weights after block scaling and observed similar centroids across tensor rows. Statistics collected from several models were fitted with a cubic polynomial, scaled to the INT8 range, and rounded to obtain this table. Its nonuniform spacing gives finer resolution near zero and coarser resolution in the tails. The fixed INT8 lookup table also supports efficient CPU SIMD decoding, avoiding evaluation of the polynomial. Thus the levels are empirically derived from weights, rather than fitted anew to every cache vector or prescribed by an FP4 exponent/mantissa encoding. [Original design rationale, PR #5590][iq4-nl-origin].

Hadamard rotation is a separate attention-graph operation: it can transform K/V before IQ4_NL encoding, but is not part of the IQ4_NL format or encoder. Its use for KV caching therefore also requires support from the selected attention backend.

### K-quants: quantize the scales as well

The Q2_K–Q6_K weight formats use super-blocks of **256 weights**, split into groups of 16 or 32. Each group has an integer scale code and, for affine formats, an integer minimum code. Shared FP16 factors reconstruct these local parameters:

$$
\hat w_{g,i}=(D s_g)q_{g,i}-D_{\min}m_g.
$$

This is the reconstruction for Q2_K, Q4_K, and Q5_K. Q3_K and Q6_K omit the minimum term and use signed decoded weights and scale codes. Thus the hierarchy compresses metadata while retaining local quantization grids. [K-quant guide][k-guide]; [Q4_K reconstruction][k-dequant].

| Format | Groups × weights | Local parameter precision | Bytes / super-block; bits / weight |
| --- | --- | --- | --- |
| Q2_K | 16 × 16 | 4-bit scale + 4-bit minimum | 84; **2.625** |
| Q3_K | 16 × 16 | 6-bit signed scale | 110; **3.4375** |
| Q4_K | 8 × 32 | 6-bit scale + 6-bit minimum | 144; **4.5** |
| Q5_K | 8 × 32 | 6-bit scale + 6-bit minimum | 176; **5.5** |
| Q6_K | 16 × 16 | 8-bit signed scale | 210; **6.5625** |

These are exact block costs, derived from [`ggml-common.h`][llama-k-blocks], excluding GGUF headers and tensor alignment.

**Why 6-bit metadata?** Q4_K keeps weights at 4 bits and represents each group scale/minimum with a 6-bit code (64 levels). This is a format-specific precision/storage choice. The guide's blanket description of block constants as INT8 is inaccurate: `uint8_t scales[12]` is a byte array containing **sixteen packed 6-bit values**, not twelve independent 8-bit constants. The table shows the actual widths across formats; Q3_K and Q6_K also use only one FP16 super-block factor because they have no minimum term. [Original K-quant design][k-design].

For example, `block_q4_K` has FP16 `d` and `dmin`, **12 bytes** of packed `scales`, and **128 bytes** of `qs`. The 12 bytes hold eight scale/minimum pairs: $8\times(6+6)=96$ bits. Total storage is $4+12+128=144$ bytes. Eight Q4_1 blocks cover the same 256 weights in 160 bytes. Q4_K therefore retains affine reconstruction at 4.5 rather than 5 bits per weight, at the cost of quantizing its local parameters. [Layout][llama-k-blocks]; [unpacking][k-unpack].

Using 8-bit rather than 6-bit group parameters would increase this metadata from 12 to 16 bytes, giving 148 bytes per super-block (4.625 bits per weight). The 6-bit choice saves four bytes; the layout alone does not establish that it is universally optimal.

### Three representative data structures

The following field-equivalent definitions expand array sizes and omit union aliases and portability macros for clarity. `ggml_half` stores a 16-bit floating-point value. [Original definitions][llama-blocks].

```c
// Legacy Q4_0: 32 weights, 18 bytes.
typedef struct {
    ggml_half d;        // One scale shared by all 32 weights.
    uint8_t qs[16];     // 32 packed 4-bit codes.
} block_q4_0;

// Legacy Q4_1: 32 weights, 20 bytes.
typedef struct {
    ggml_half d;        // Block scale.
    ggml_half m;        // Block minimum (additive offset).
    uint8_t qs[16];     // 32 packed 4-bit codes.
} block_q4_1;

// K-quant Q4_K: 256 weights in 8 groups, 144 bytes.
typedef struct {
    ggml_half d;        // Super-block factor for group scales.
    ggml_half dmin;     // Super-block factor for group minima.
    uint8_t scales[12]; // 8 pairs of 6-bit scale/minimum codes.
    uint8_t qs[128];    // 256 packed 4-bit weight codes.
} block_q4_K;
```

Q4_0 directly stores its block scale. Q4_1 additionally stores an offset, reconstructing each weight as $\hat w_i=dq_i+m$ for $q_i\in\{0,\ldots,15\}$; its 20-byte block costs 5 bits per weight. Q4_K reconstructs each group's scale and minimum from packed integer metadata and two shared FP16 factors.

### Tensor presets and runtime representations

`Q4_K` is a tensor encoding; **Q4_K_S and Q4_K_M are model-conversion presets**. Both select Q4_K as their base type, but conversion rules can assign other types to selected tensors. For example, Q4_K_M can promote selected attention-value tensors to Q6_K. Consequently, a model's average storage cost need not equal the block-level 4.5 bits per weight. `_S/_M/_L` describe precision-allocation choices, not separate C block structures. [Preset mapping][llama-presets]; [tensor selection][llama-selection].

### Activations: temporary quantization for a kernel

In the generic CPU matrix-multiplication path, the weight type selects a companion activation type through `vec_dot_type`. Floating-point activations are converted into a work buffer using `from_float`, consumed by the matching dot-product kernel, and produce FP32 output. Quantization is local to this operation; it does not make every intermediate tensor an INT8 tensor. [CPU conversion path][cpu-mulmat].

| Weight type | Temporary activation type on this CPU path |
| --- | --- |
| Q4_0, Q5_0, Q8_0 | Q8_0 |
| Q4_1, Q5_1 | Q8_1 |
| Q2_K–Q6_K | Q8_K |

These pairings come from the [CPU type-trait table][cpu-types]. Scales are computed from the **current activation blocks**, rather than loaded from a calibration checkpoint. For example, the reference Q8_0 quantizer uses $d=\max_i|x_i|/127$ for each 32-value block, then rounds $x_i/d$ into signed INT8 codes. [Quantization routine][online-q8].

Q8_1 additionally stores $d\sum_i q_i$; Q8_K stores an FP32 scale, 256 INT8 codes, and sums over 16-value groups. These sums support offset corrections in affine dot products. Q8_K is an intermediate computation format, not a selectable KV-cache format below. [Structures][llama-blocks].

The pairing is backend-dependent: CUDA's quantized matrix-vector path creates **Q8_1** activations even for Q4_K weights. Other kernels can follow different paths; the CPU table is not a universal activation-format rule. [CUDA conversion][cuda-mmvq].

### KV cache: from token identity to byte address

For ordinary non-MLA attention, llama.cpp manages a **fixed-capacity slot space backed by separate K and V tensors for each participating layer**. Each allocated tensor is contiguous in CPU or GPU memory, including space for empty slots. A slot is a shared index into these tensors; one token's cache across all layers is distributed across them. Tensor regions may share backend buffers or reside on different devices. [Allocation][llama-cache-allocation].

A **stream** in llama.cpp's kv cache management is a storage partition with its own slot metadata and allocation cursor, unrelated to a CUDA execution stream. Unified mode maps all sequences to stream 0; non-unified mode creates `n_seq_max` streams and maps sequence IDs to their corresponding streams. Each stream has a configured slot capacity $T$. [Stream mapping][llama-cache-streams].

**Worked example.** Assume two layers, one stream, two KV heads, 64 components per head, and Q8_0 storage for both K and V. We show the first three slots, using the non-transposed layout and zero-based indices:

```text
Token identity                   Assigned slot in stream 0
Sequence A, logical position 5 →  0
Sequence B, logical position 9 →  1
Sequence A, logical position 6 →  2

Separate contiguous tensor regions (first three entries shown):
Layer 0 K: [K for slot 0][K for slot 1][K for slot 2] ...
Layer 0 V: [V for slot 0][V for slot 1][V for slot 2] ...
Layer 1 K: [K for slot 0][K for slot 1][K for slot 2] ...
Layer 1 V: [V for slot 0][V for slot 1][V for slot 2] ...
```

The metadata maps token identity to a slot; position 9 does **not** imply slot 9. Attention masks use the recorded positions and sequence membership. [Cell metadata][llama-cache-cells]; [mask construction][llama-cache-mask].

Use the following **explanatory indexing notation**, rather than literal C++ array syntax:

```text
K[layer][stream][slot][head][component]
V[layer][stream][slot][head][component]

K[1][0][1][1][37]
  = layer 1's K, stream 0, slot 1, head 1, component 37
  = one cached component for sequence B at position 9
```

Layer and K/V select a tensor. Within it, storage nests as **stream → slot → head → quantization block**. Each Q8_0 block contains `[FP16 scale | 32 INT8 codes]`, occupying 34 bytes. Thus a 64-component head uses two blocks (68 bytes), and a token's entry in one K or V tensor uses two heads (136 bytes). Blocks group adjacent components of one head, not tokens. [Block definition][llama-blocks]; [tensor strides][llama-cache-views].

The byte offsets below are relative to the selected tensor's base. Reading across each row and then downward follows increasing addresses:

| Slot | Head 0, block 0 | Head 0, block 1 | Head 1, block 0 | Head 1, block 1 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 34 | 68 | 102 |
| 1 | 136 | 170 | 204 | 238 |
| 2 | 272 | 306 | 340 | 374 |

For the example access, component 37 belongs to block 1, code index 5:

```text
K[1][0][1][1][37]:
  block offset = slot 1 × 136 + head 1 × 68 + block 1 × 34 = 238
  scale address = K_layer1_base + 238         (2-byte FP16)
  code address  = K_layer1_base + 238 + 2 + 5 = K_layer1_base + 245
  reconstructed component = scale × signed_INT8_code
```

`V[1][0][1][1][37]` uses the same offsets in this example, but starts from **`V_layer1_base`**. It is in a separate tensor.

**Dimension notation does not change the layout.** Our indexing lists outermost to innermost: `[stream, slot, head, component]`. GGML lists the fastest-varying dimension first: the allocation has dimensions `[dH, T, S]`, and the attention view separates heads as `[d, H, n_kv, S_view]`. Here $d$ is head dimension, $H$ is KV-head count, and $S$ is stream count; the view can cover a subset of streams and a padded slot extent `n_kv` within capacity $T$. These describe the same storage, using different dimension conventions. [Allocation][llama-cache-allocation]; [views][llama-cache-views]; [attention extent][llama-cache-extent].

The exception is **transposed V**, whose physical nesting is `[stream][head][component][slot]`: one token's components are then separated in memory. Quantized V requires FlashAttention at this revision and uses the non-transposed layout illustrated above. [V strides][llama-cache-views]; [validation][cache-validation].

**Removal and reuse change occupancy, not allocation.** Removing sequence B from the example leaves a hole; a new token can reuse it without moving A's surviving entries:

```text
Physical slot          0       1       2
Before removal       [ A/5 ] [ B/9 ] [ A/6 ]
Remove B             [ A/5 ] [free ] [ A/6 ]
Insert A/7           [ A/5 ] [ A/7 ] [ A/6 ]
```

“Free” means reusable: memory remains allocated and may retain stale bytes. A cell shared by several sequences becomes free only when none still references it. [Removal][llama-cache-removal]; [cell release][llama-cache-cell-release].

Each stream's cursor scans for empty or sliding-window-reclaimable cells, skipping live entries and wrapping at the end. For example, it can select the last slot, wrap, skip an occupied slot 0, and select free slot 1. After assignment, the cursor becomes `last_selected_slot + 1`; removal or a hole-filling heuristic can move the search start earlier. Insufficient reusable slots cause allocation failure, not automatic growth or unconditional eviction. [Slot search][llama-cache-search]; [cursor update][llama-cache-cursor].

### KV cache: quantize on insertion, retain across tokens

The [server documentation][server-cache] exposes independent `--cache-type-k` and `--cache-type-v` options, both defaulting to **F16**. At this revision, accepted quantized types are **Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, and IQ4_NL**; floating-point alternatives are F32, F16, and BF16. The [argument parser][cache-types] excludes Q4_K, Q8_K, and Q8_1. A Q4_K weight model therefore does not imply a Q4_K cache.

For the ordinary attention-cache path, online data passes through three stages:

1. **Write:** newly computed K/V rows enter `cpy_k`/`cpy_v`, which construct `ggml_set_rows` operations targeting the selected cache tensors. The CPU implementation calls the destination type's `from_float` converter; CUDA has corresponding block-quantizing row-write kernels. Codes and locally computed scales/offsets are written together. [Cache insertion][cache-write]; [CPU writer][cpu-cache-write]; [CUDA writer][cuda-cache-write].
2. **Retain:** blocks span adjacent vector components within a token's row, not a sequence of 32 tokens. Ordinary appends quantize the new rows without refitting old rows' scales. Thus cache quantization requires no offline calibration of those generated values. [Row layout and indexing][cache-write].
3. **Read:** in the generic CPU FlashAttention path, Q is temporarily converted to K's companion dot-product type; quantized K participates in the dot product. Quantized V is converted a row at a time to FP32 for the weighted accumulation. The persistent cache stays compressed. [CPU attention][cpu-attention].

There are version- and backend-specific qualifications. This revision conditionally applies Hadamard rotations before cache quantization: Q/K rotate together, and V's rotation is undone after attention. Quantized-cache head dimensions must meet block-alignment requirements; quantized V requires FlashAttention. [Rotation selection][cache-rotation]; [graph ordering][cache-graph]; [validation][cache-validation].

**Does attention dequantize the cache or quantize Q?** Both occur, depending on the selected kernel and the attention stage. For quantized K/V at this revision:

| Execution path | Score computation, $QK^T$ | Value accumulation, $PV$ |
| --- | --- | --- |
| Generic CPU FlashAttention | Convert Q to K's companion quantized type and call the matching dot product | Dequantize V rows to FP32 and accumulate with floating-point softmax weights |
| CUDA FlashAttention vector kernel, with a compiled quantized-type pairing | Quantize Q into a temporary Q8_1-style representation; unpack K codes and use integer dot products with block-scale/offset corrections | Dequantize V inside the kernel and accumulate in floating point |
| CUDA FP16 MMA kernel | Convert quantized K to an FP16 temporary buffer; use FP16 Q/K operands for tensor-core MMA | Convert quantized V to an FP16 temporary buffer and use floating-point MMA |

The CPU behavior is automatic **within that implementation**, not a backend-independent rule. CUDA's vector kernel explicitly calls `quantize_q8_1_to_shared`; its Q4_0/Q4_1 score routines use `ggml_cuda_dp4a`, an integer vector dot product rather than low-bit tensor-core MMA. Softmax remains floating point. [CPU implementation][cpu-attention]; [CUDA query conversion][cuda-fa-query]; [integer score routines][cuda-fa-dot]; [value dequantization][cuda-fa-value].

CUDA selects these paths automatically; the decision tree below gives the conditions. Crucially, the FP16 path launches K/V conversion **before** attention and materializes temporary device buffers; it does not merely dequantize tiles inside the attention kernel. Missing compiled vector type pairings also fall back to FP16 conversion. [Conversion and buffers][cuda-fa-convert]; [MMA launch][cuda-fa-mma]; [build options][cuda-fa-build].

**Design interpretation:** packed codes with per-block scales/offsets are not directly usable as ordinary integer-MMA operands: unpacking, tile layout, and scale corrections must be handled explicitly. FP16 reconstruction enables conventional tensor-core kernels. Nevertheless, for few-query decoding, compressing repeated cache reads and reusing a temporarily quantized Q can favor integer vector dot products. Thus block-wise storage alone does not establish which route is faster; the tradeoff depends on query count, cache length, hardware, and conversion overhead. No performance comparison is claimed here.

For example, this requests Q8_0 cache storage independently of a Q4_K_M weight checkpoint:

```sh
llama-server -m model-Q4_K_M.gguf \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
```

This is a configuration example, not a benchmark. Q8_0 cache blocks occupy 34 bytes per 32 values (8.5 bits/value), versus 64 bytes for F16; total memory also includes cache padding and runtime buffers. The central distinction is **temporary activation quantization for computation versus persistent KV quantization for storage and repeated attention reads**.

### Attention dispatch: from graph to kernel

The following trees trace revision `a894dae939d4`. **Graph selection, kernel-family selection, and type conversion are separate decisions.** A prefill/decode label alone does not select a kernel.

```text
Ordinary attention graph (after configuration validation)
├─ FlashAttention disabled, or explicit KQ bias tensor present
│  └─ Separate QK matmul → softmax → V matmul
└─ FlashAttention enabled and no explicit KQ bias tensor
   └─ FLASH_ATTN_EXT, assigned by the backend scheduler
      ├─ Generic CPU, with either K or V quantized
      │  └─ Row-wise implementation: K's type selects Q's dot-product type
      └─ CUDA backend
         └─ Support checks → kernel-family tree → type-conversion tree
```

Quantized V with `--flash-attn off` is rejected; `auto` is promoted to `on` for quantized V. Otherwise, automatic FlashAttention resolution can disable it when the probe is assigned to a different device from its layer. A CUDA `NONE` result means that backend rejects the operation, **not** that the selector itself calls CPU attention or unfused attention. [Configuration][cache-validation]; [automatic resolution][fa-auto]; [graph construction][fa-graph]. On the generic CPU, both the tiled and split-KV optimizations require matching F16/F32 K/V, so quantized-cache inputs take the row-wise implementation. [CPU dispatch][cpu-fa-select].

For CUDA, define the following directly from tensor dimensions and source predicates:

| Symbol | Meaning |
| --- | --- |
| `d`, `dv` | Q/K head dimension `Q.ne[0] = K.ne[0]`, and V head dimension `V.ne[0]` |
| `n`, `s`, `t` | Query positions `Q.ne[1]`, sequence dimension `Q.ne[3]`, and supplied KV extent `K.ne[1]` (including padding) |
| `r` | GQA ratio `Q.ne[2] / K.ne[2]` |
| `B` | At least one of K or V is quantized |
| `G` | `r >= 2`, mask present, `max_bias == 0`, `t % 256 == 0`, and strides `nb[1..3]` of each non-quantized Q/K/V/mask tensor divisible by 16 |
| `W` | Vector eligibility: `d ∈ {64,128,256}` and `t % 256 == 0` |
| `e` | Largest power-of-two divisor of `r`, capped at 32 for `d=320`, 16 for `d∈{192,576}`, otherwise 8 |
| `A` | Device compute capability at least Ada Lovelace |

Here `n` is **not** the command-line batch limit or `n × r`. The source uses `n × e` only in the branches shown below, and computes `e` independently of `G`. MMA availability also depends on the compiled GPU architectures, not just the device name. [Dimension and eligibility logic][cuda-fa-gates]; [capability helpers][cuda-fa-capabilities].

<details markdown="1">
<summary>CUDA support checks, before kernel selection</summary>

FlashAttention must be compiled in. K/V must each be one of F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, or Q8_0; a supplied mask must have `ne[2] == 1`. The head shapes must satisfy:

| `d` | Additional requirements |
| --- | --- |
| 40, 64, 72, 80, 96, 112, 128, 256 | `dv == d` |
| 192 | `dv == 128`, `G`, and `r % 8 == 0` |
| 320 | `dv == 256`, `G`, and `r % 32 == 0` |
| 512 | `dv == 512` and `G` |
| 576 | `dv == 512` and `G` |
| Any other dimension | Unsupported |

Failure returns `NONE`. In particular, **IQ4_NL is accepted by the cache-option parser but rejected by this CUDA FlashAttention selector**. Its K/V type allowlist omits IQ4_NL, so selection stops before VEC/MMA/TILE dispatch. The F16 conversion fallback for an uncompiled vector type pair occurs later and does not bypass this check. Enabling all supported CUDA FlashAttention quantizations therefore does not add IQ4_NL support. [Support checks][cuda-fa-gates]; [conversion fallback][cuda-attention]; [build options][cuda-fa-build].

This is an implementation coverage limit, not a fundamental inability of CUDA to evaluate a codebook. IQ4_NL needs its lookup mapping when reconstructing values or forming dot products; CUDA already implements that mapping for other operations. The inspected FlashAttention path does not expose the format. Its exclusion alone does not establish a performance-based reason for omitting it. [CUDA IQ4_NL reconstruction][iq4-nl-cuda].

</details>

Branches below are evaluated **in order**; within a branch, the first matching condition wins. `MMA` means `MMA_F16`, and `TILE` means the generic floating-point tile kernel. The AMD branches belong to the shared implementation used by HIP. [Complete selector][cuda-fa-select].

```text
CUDA FlashAttention support checks
├─ Fail → NONE
└─ Pass
   ├─ Turing-or-newer MMA available AND d not in {40,72}
   │  ├─ W AND B AND (A ? n <= 2 : n == 1) → VEC
   │  ├─ W AND NOT B AND F                 → VEC
   │  ├─ W AND NOT G AND n == 1            → VEC
   │  └─ Otherwise                        → MMA
   ├─ Else, Volta MMA available AND d not in {40,72}
   │  ├─ W AND n*e <= 2                   → VEC
   │  ├─ n*e <= 16                        → TILE
   │  └─ Otherwise                        → MMA
   ├─ Else, AMD MFMA available AND d not in {40,72}
   │  AND any of the following:
   │     d <= 64  AND n*e > 8
   │     d <= 128 AND n*e > 16
   │     d <= 256 AND n*e > 64             → MMA
   ├─ Else, AMD WMMA available AND G AND d <= 256
   │  AND d not in {40,72}
   │  AND n*e > (d <= 128 ? 8 : 16)        → MMA
   └─ Otherwise
      ├─ W AND B AND n <= 2               → VEC
      ├─ W AND NOT B AND n == 1 AND NOT G → VEC
      └─ Otherwise                       → TILE
```

`F` is the floating-cache fast path: `A && n == 1 && s == 1 && !(r > 4 && t >= 8192) && !sparse_decode`. The source defines `sparse_decode` as `G && r > 4 && may_use_sparse(d,dv,1,8) && shall_use_sparse(cc,dst,1)`; those helpers check the compiled shape and sparse-mask workload. This condition is irrelevant when `B` is true. [Sparse shape predicate][cuda-fa-sparse-shape]; [sparse workload predicate][cuda-fa-sparse-work].

**Conceptual interpretation.** The dispatcher chooses between a specialized vector kernel for very few queries, matrix hardware for workloads with sufficient parallelism and K/V reuse, and a general floating-point tile kernel. These are fixed selection heuristics, not runtime benchmarking.

| Path | Role in the decision tree |
| --- | --- |
| `VEC` | Handle eligible, very small query workloads; compiled quantized-K variants can consume compressed K using a temporarily quantized Q |
| `MMA` | Exploit matrix hardware and reuse K/V across queries or grouped heads; quantized K/V first require FP16 conversion in this revision |
| `TILE` | Handle workloads for which MMA is unavailable or not selected, including intermediate workloads on Volta |

The architecture branches express different thresholds for making matrix hardware worthwhile:

- **Modern NVIDIA:** prefer MMA, except for eligible small workloads. Quantized caches select VEC for one query position, or up to two on Ada and newer. Floating-point caches have additional GQA, cache-length, and sparse-attention conditions that can retain MMA even for one query.
- **Volta:** use VEC when `n × e <= 2` and vector eligibility holds; otherwise use TILE up to `n × e == 16`, then MMA. Grouped query heads can therefore move single-position decoding to TILE or MMA.
- **AMD:** MFMA requires `n × e > 8`, `> 16`, or `> 64` for head dimensions up to 64, 65–128, or 129–256, respectively, subject to the exclusions above. WMMA requires GQA optimization and its own threshold. When these conditions fail, execution continues to the final VEC/TILE choice.
- **Final fallback:** choose VEC for eligible tiny workloads; otherwise TILE. `W` is a hard shape/alignment requirement, not a performance estimate. [Selection rules and source comments][cuda-fa-select].

The likely performance rationale is that few-query decoding offers limited matrix parallelism: quantizing Q once and repeatedly reading compressed K can be economical. More queries or grouped heads increase opportunities to reuse K/V, potentially amortizing FP16 conversion and making MMA preferable. This is an interpretation of the heuristics, not a measured performance result. **Query count, reuse, hardware, and kernel eligibility determine the route; “decode” and “prefill” alone do not.**

Finally, selecting `VEC` does **not** guarantee integer arithmetic:

```text
Selected kernel family
├─ MMA or TILE
│  └─ Convert non-F16 K/V to temporary F16 buffers before attention
└─ VEC
   ├─ No compiled instance for (d, K type, V type)
   │  └─ Use F16/F16 VEC, converting non-F16 K/V first
   └─ Compiled instance exists
      ├─ K quantized → quantize Q; integer score dot products
      ├─ K floating  → floating-point score dot products
      └─ Independently: quantized V → dequantize inside accumulation
```

The last three lines describe score and value handling within the same kernel. **Quantized V alone does not cause Q quantization**: Q's conversion is selected by K's type. The persistent cache remains in its configured type, even when a temporary FP16 copy is needed. [Vector lookup/fallback][cuda-attention]; [buffer selection][cuda-fa-buffers]; [query conversion][cuda-fa-query]; [value accumulation][cuda-fa-value]. For example, with supported Q8_0/Q8_0, `d=128`, padded KV extent, and a matching compiled vector instance, an Ada GPU uses `VEC` for `n=1` or `2`, and `MMA` for `n>=3` when Turing-or-newer MMA is available in the build.

### What the evaluation establishes

[Kurt's study][quant-study] evaluates Llama-3.1-8B-Instruct using llama.cpp `b7600` on two Xeon Platinum 8488C CPUs. Q4_K_S reduces file size by **70.83%** and scores **69.17** on the unweighted five-benchmark mean, versus **69.47** for F16; Q4_0 gives a similar size reduction (**71.03%**) but a lower mean (**67.98**). This supports comparing format design at similar storage budgets. It does not establish a universal winner: the same study finds non-monotonic throughput across formats, and its results concern one model and CPU setup (Tables 2–3).

**Next:** Validate the inspected execution paths with kernel traces and measure the separate effects of weight and KV-cache quantization.

## Reading the backend profiles

The llama.cpp case follows formats, cache management, and kernels in one project. For modular stacks, we will instead examine **one backend at a time**, in the same order: **cache types → storage layout → attention paths → integration constraints**. The final vLLM section then assembles those constraints into its selection process.

All four profiles describe the integrations inspected at vLLM `17e50b9b7610`; they are not universal support claims about the underlying libraries. An advertised cache type is only an initial allowlist: hardware, shapes, attention features, and kernel availability can impose further restrictions. Library capabilities that vLLM does not expose are identified separately.

### Components and integration boundaries

vLLM's registered adapters implement its attention interface and invoke either external libraries or kernels within vLLM. An adapter, a kernel library, and a quantization method are different kinds of components. [Registry][vllm-attention-registry].

| Name | Role and relationship |
| --- | --- |
| **FlashAttention** | An attention algorithm family and its kernel implementations. vLLM's `FLASH_ATTN` adapter calls a packaged FlashAttention implementation; its version and GPU support matter. [Project][flashattention-project]; [vLLM dependency pin][vllm-fa-dependency] |
| **TensorRT-LLM (TRT-LLM)** | NVIDIA's inference framework/runtime, containing reusable attention kernels. In the vLLM routes inspected here, those kernels are exposed through FlashInfer APIs; TensorRT-LLM does not replace vLLM's serving engine. [Project][trtllm-project]; [kernel API][flashinfer-trtllm-api] |
| **FlashInfer** | An inference kernel library with several internal backends, including FA2/FA3 and TRT-LLM implementations. Thus vLLM can select `FLASHINFER`, which in turn invokes one of several attention implementations. [Project][flashinfer-project] |
| **Triton** | A GPU programming language and compiler. vLLM's `TRITON_ATTN` is a particular attention implementation written in Triton, rather than a separate serving engine or quantization algorithm. [Language][triton-language]; [kernel][vllm-kv-compute] |
| **TurboQuant** | A quantization method. vLLM implements selected variants through a dedicated `TURBOQUANT` attention backend, including Triton and alternative FlyDSL paths. The method, adapter, and implementation language are distinct. [Variants][vllm-turbo-config]; [implementation selection][vllm-turbo-packing-select] |

Here **Triton means the GPU language/compiler**, not NVIDIA Triton Inference Server. FlashInfer can expose FlashAttention-derived or TRT-LLM-derived kernels; selecting the FlashInfer adapter therefore does not identify a unique kernel family.

### vLLM's cache interface: logical indices, physical order, and packing

The backend profiles distinguish **logical indices** (which cached value), **physical axis order** (where it resides), and **content packing** (how its codes and metadata are encoded). A **page** groups tokens for allocation and addressing; a **quantization group** contains numerical components sharing reconstruction parameters.

The responsibilities below describe the inspected vLLM integration. The adapter, cache manager, layout resolver, and allocator are parts of vLLM; the kernels may reside in vLLM or an external library.

| Component | Responsibility |
| --- | --- |
| vLLM cache manager | Assign and reuse physical pages; maintain request block tables and token-to-slot mappings |
| Backend adapter | Declare supported layouts and formats; prepare the tensor views, strides, and metadata required by its kernels |
| vLLM layout resolver | Select one axis-order policy compatible with the selected adapters and configuration |
| vLLM cache allocator | Allocate backing memory and construct per-layer views with strides matching that policy and the cache specification |
| Cache-write and attention kernels | Compute addresses and write/read codes, values, and metadata according to the strides and format-specific packing |

[Page management][vllm-block-metadata]; [adapter contract][vllm-backend-layout]; [resolver][vllm-layout-resolver]; [allocation][vllm-cache-allocate]; [example writer][vllm-kv-write] and [reader][vllm-kv-compute].

```text
Initialization:
  adapter constraints → vLLM layout selection → allocation and strided views
Execution:
  request's block/slot mapping + adapter-prepared views → kernel address calculation
```

**The logical indexing is implemented by vLLM.** Its byte-level storage abstraction uses `[L,B,H,N,C]`: layers, physical blocks/pages, head slots, stored states (normally tokens), and content bytes. Each layer receives a `[B,H,N,C]` view. The shape names coordinates; **strides map those coordinates to byte addresses**. Adapters can split, reinterpret, or transpose these views to match their kernel interfaces. [View construction][vllm-cache-views]; [layout definitions][vllm-cache-layout].

**A layer's page contains both K and V.** In the inspected FlashAttention and FlashInfer FP16/BF16 and conventional FP8 paths, each token/head record stores a complete K vector followed by a complete V vector:

```text
R(t,h) = [ K(t,h): D components ][ V(t,h): D components ]
```

K/V are paired at vector granularity, rather than alternating individual components as `k0,v0,k1,v1,…`. The page therefore contains many such K/V pairs, covering its token slots and the layer's local KV heads. [FlashAttention views][vllm-fa-layout]; [FlashInfer views][vllm-fi-payload].

For formats with a contiguous record per token/head, let `R(t,h)` denote that record. **This is explanatory notation, not a universal vLLM record type.** The same records can have different physical orders; the following shows three tokens and two heads within one page:

```text
Token-major: [token][head][record bytes]
  [R(0,0)][R(0,1)] [R(1,0)][R(1,1)] [R(2,0)][R(2,1)]

Head-major: [head][token][record bytes]
  [R(0,0)][R(1,0)][R(2,0)] [R(0,1)][R(1,1)][R(2,1)]
```

Both expose the same logical `cache[page,head,token,content_byte]` indexing. With an illustrative **16-byte record**, the densely packed page above has these byte strides and offsets:

| Physical order | Token stride | Head stride | Offset of `cache[0,0,1,0]` |
| --- | ---: | ---: | ---: |
| Token-major (`LBNHC`) | 32 | 16 | 32 |
| Head-major (`LBHNC`) | 16 | 48 | 16 |

The layout names additionally place layers and pages outermost. For this byte view, addressing is `base + page × page_stride + head × head_stride + token × token_stride + content_byte`. Thus **layout selection determines storage order; strides encode its connection to logical indexing**. The adapter supplies suitable views and arguments, while the kernel executes address arithmetic, using explicit strides or a supported layout-specific implementation. Merely transposing a view does not move bytes; rearranging populated storage while preserving its logical contents requires data movement. [Layout enum][vllm-cache-layout]; [views][vllm-cache-views]; [example kernel addressing][vllm-kv-compute].

**Separate K/V tensor arguments can reference the same page storage.** For example, FlashAttention splits the content dimension after dtype reinterpretation:

```python
# Logical shape: [pages, heads, tokens, 2 * head_size]
key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
# Each view: [pages, tokens, heads, head_size]
```

These are views without copying: their starting offsets select K or V, and their strides skip the other vector's bytes. They do not allocate separate K-only and V-only pages. [Actual split][vllm-fa-layout].

**Adapters constrain the choice; vLLM makes the final selection.** It intersects their supported orders; `None` expresses no constraint or preference through this interface. Incompatible declarations cause an error. The detailed selection policy appears in the vLLM section below. [Resolver][vllm-layout-resolver].

**A shared axis order does not make cache formats interchangeable.** The ordering enum does not fully specify K/V field offsets, scale placement, or code packing. For example, conventional FP8 passes scales separately, Triton's per-token/head formats store inline metadata, and FlashInfer's NVFP4 path separates K/V head groups and data/scale regions within a page rather than using the ordinary vector-pair record above. Each profile therefore specifies both axis order and format-specific packing. [NVFP4 views][vllm-nvfp4-views].

### Attention arithmetic over compressed caches

The profiles distinguish two common computation patterns: **FP8 dot-product operands with scale corrections**, and **reconstruction into higher-precision operands**. Reconstruction can happen inside attention or in a temporary buffer. FP8 cache storage alone establishes neither the operand dtype nor the accumulation dtype.

## FlashAttention: FP16/BF16 and conventional FP8

### FlashAttention: cache types

The vLLM `FLASH_ATTN` adapter accepts `auto`, `float16`, `bfloat16`, `fp8`, and `fp8_e4m3`; `fp8` means E4M3. It does not accept E5M2, per-token/head FP8/INT8/INT4, NVFP4, or TurboQuant records. Those formats require a different compatible adapter. [Allowlist][vllm-fa-cache-allowlist].

Conventional FP8 reconstructs $\hat x=sx_q$, with $x_q=\operatorname{cast}_{\mathrm{FP8}}(x/s)$. Per-layer K/V scales are loaded from checkpoints and default to 1.0 when absent. Calibrated per-head scales are another supported pathway, documented with llm-compressor and FlashAttention; these scales are shared across tokens, unlike dynamic per-token/head quantization. [Calibration][vllm-kv-doc]; [scale loading][vllm-kv-scales]; [per-head support][vllm-kv-heads].

### FlashAttention: storage layout

The adapter splits the per-layer cache into logical `[block, token, head, component]` K and V views. Split/transpose operations retain the backing storage and expose its strides. FP8 scales are passed separately from the cache payload. The layout declaration is inherited `None`: this imposes no ordering preference through vLLM's layout interface. With no other restriction, override, or connector preference, vLLM selects token-major `LBNHC`; `None` does not mean that the cache has no layout. [Views and scales][vllm-fa-layout]; [layout interface][vllm-backend-layout].

### FlashAttention: attention paths

| Implementation | Input and computation |
| --- | --- |
| FA2 CUDA | FP16/BF16 with matching Q/K/V dtypes. FP8 KV is unsupported. [Checks][fa2-input-dtypes] |
| FA3 CUDA | Matching FP16, BF16, or E4M3 FP8 Q/K/V. For supported FP8 KV, **vLLM quantizes Q before the call**; FA3 selects its FP8 specialization from the supplied dtype. It does not quantize a BF16 query itself. [Checks][fa3-input-dtypes]; [dispatch][fa3-dtype-dispatch] |
| FA4 through vLLM's ordinary adapter | Supported conventional FP8 KV likewise uses FP8 Q. The adapter passes Q/K/V scales to the library. [Wrapper][vllm-fa4-wrapper] |
| Additional FA4 library capability | The pinned library has `fp8_kv_dequant=True`: paged E4M3 KV with FP16/BF16 Q on SM90, subject to shape/page constraints. It reconstructs KV inside the kernel for FP16 computation with FP32 accumulation. **The inspected vLLM wrapper does not enable this option.** [Library][fa4-kv-dequant]; [wrapper][vllm-fa4-wrapper] |

The adapter's query capability is an internal declaration, not a user option:

```python
def flash_attn_supports_quant_query_input() -> bool:
    return not current_platform.is_xpu()
```

It is `True` on CUDA even with a BF16 cache. Q conversion additionally requires an eligible cache type; BF16 KV does not trigger it. Thus the accepted E4M3 route supplies FP8 dot-product operands, while scales, softmax, accumulation, and output retain their own precision rules. [Capability assignment][vllm-fa-query-capability]; [conversion conditions][vllm-query-quant].

### FlashAttention: constraints used by vLLM

The CUDA adapter requires SM80 or later and a head dimension divisible by eight. Its initial head-size check accepts up to 256, or up to 512 when FA4 is available; version-specific checks can narrow that set. Kernel block sizes are generally multiples of 16, with a dedicated FA4 head-dimension-256 path requiring 128-token kernel pages. [Shape and block checks][vllm-fa-shape-gates].

For FP8 KV, the final version/hardware gate is **FA3 on SM90 or FA4 on SM100**. ALiBi, head dimensions, batch invariance, and other features can alter version selection before this gate. Failure does not trigger a dequantized fallback within the same adapter: explicit incompatible selection errors, while automatic selection can choose another backend. The exact vLLM selection sequence is collected below. [Version policy][vllm-fa-version-policy]; [FP8 gate][vllm-fa-fp8-gate].

## Triton: conventional FP8 and per-token/head quantization

### Triton: cache types

`TRITON_ATTN` implements attention kernels within vLLM using the Triton compiler. Its allowlist includes `auto`, FP16/BF16, conventional FP8 (`fp8`, E4M3, E5M2), and `int8_per_token_head`, `fp8_per_token_head`, and `int4_per_token_head`. An enum/allowlist entry does not establish every end-to-end route: the shared CUDA Q-conversion path explicitly asserts E4M3 for conventional FP8. The direct FP8 example below is therefore scoped to E4M3. [Allowlist][vllm-triton-cache-types]; [Q conversion][vllm-query-quant].

For dynamic INT8/FP8, each incoming K or V head vector gets its own scale $s=\max(\max_i|x_i|/q_{\max},10^{-6})$. The writer stores the codes and an FP32 scale at the assigned slot; entries remain compressed across subsequent tokens. This differs from scales calibrated once per layer/head. INT4 instead uses asymmetric parameters after rotation. [Dynamic writer][vllm-kv-write].

### Triton: storage layout

The adapter inherits layout preference `None` and reads actual tensor strides. Its per-token/head records can use either token-major or head-major pages; absent other requirements, vLLM chooses `LBNHC`. Each record keeps K/V codes and metadata inline, while conventional FP8 uses separate scale arguments. [Layout interface][vllm-backend-layout]; [scale views][vllm-cache-scale-views].

For Triton's per-token/head formats, content size is computed by this excerpt from `customize_spec()`:

```python
hs_k, hs_v = spec.head_size, spec.head_size_v
if mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
    hs_k, hs_v = hs_k // 2, hs_v // 2
scale_bytes = get_dtype_size(torch.float32)
content = (hs_k + hs_v) * get_dtype_size(spec.dtype) + 2 * scale_bytes
return replace(spec, state_content_bytes=content)
```

The resulting records for head dimension 128 are illustrated below. Each row describes **one token and one KV head**, including both K and V; sizes exclude page padding.

```text
INT8 / FP8 per token/head — 264 bytes
┌──────────────┬─────────────┬──────────────┬─────────────┐
│ K: 128 codes │ K scale: 4B │ V: 128 codes │ V scale: 4B │
│     128B     │    FP32     │     128B     │    FP32     │
└──────────────┴─────────────┴──────────────┴─────────────┘

INT4 per token/head — 136 bytes
┌──────────────┬─────────────┬──────────────┬─────────────┐
│ K: 128 codes │ K metadata  │ V: 128 codes │ V metadata  │
│ packed: 64B  │     4B      │ packed: 64B  │     4B      │
└──────────────┴─────────────┴──────────────┴─────────────┘
INT4 metadata: one 32-bit word combining a truncated scale and zero point.
```

The scale tensors passed to kernels are **views into these inline bytes**, sharing the payload allocation. INT4 readers extract the zero point and clear its bits before interpreting the scale. [Content-size code][vllm-cache-content]; [scale views][vllm-cache-scale-views]; [INT4 decoding][vllm-int4-read].

**INT4 metadata packing.** This is a packing choice in `int4_per_token_head`, not an intrinsic property of INT4. The writer replaces the lowest four fraction bits of an FP32 scale with an integer zero point. The reader separates them using bit masks; the combined word must not be used directly as a scale. [Writer][vllm-int4-pack]; [reader][vllm-int4-read].

```text
Bit 31      30 … 23       22 … 4                3 … 0
┌──────────┬────────────┬─────────────────────┬─────────────┐
│ sign: 1b │ exponent:8b│ scale fraction: 19b │ zero pt: 4b │
└──────────┴────────────┴─────────────────────┴─────────────┘
Read scale: retain bits 31…4 and set bits 3…0 to zero.
Read zero point: retain bits 3…0 as an unsigned integer.
```

The quantizer explicitly rounds and clamps $z$ to $[0,15]$, so four bits encode every allowed zero point. Here $z$ is the code representing zero, while the reconstructed value is $\hat x=\tilde s(u-z)$; the real-valued offset $-\tilde s z$ is not itself restricted to four-bit integers.

The scale retains **28 representation bits: one sign, eight exponent, and 19 fraction bits**. For normal FP32 numbers, the implicit leading one gives 20 significant binary digits, down from 24. Clearing four fraction bits introduces relative scale error at most $15\times2^{-23}\approx1.79\times10^{-6}$ (about 0.00018%), derived from FP32 spacing. The writer's minimum scale of $10^{-6}$ keeps finite scales in the normal range. Codes are computed using the original scale; reconstruction uses the slightly truncated scale. This saves a separate zero-point field at a precision cost generally much smaller than four-bit code rounding; it does not establish overall model accuracy.

**Comparison with llama.cpp.** Both `int8_per_token_head` and `fp8_per_token_head` share one dynamically computed scale across each token's entire K head vector, and a separate scale across V, with no stored zero point. They differ in quantization levels: INT8 uses a uniform integer grid, while FP8 uses nonuniform floating-point levels. In llama.cpp's legacy Q8_0/Q4_0/Q4_1 formats, reconstruction parameters instead cover groups of **32 adjacent components**. For a 128-component vector, this gives four independently scaled blocks rather than one head-wide scale. Q4_1 additionally stores an FP16 additive offset; it is not an explicitly stored integer zero point. [vLLM quantizer][vllm-kv-write]; [GGML structures][llama-blocks].

```text
One 128-component K vector (V is handled independently):

vLLM INT8 / FP8 per token/head:
  [             128 codes             ] + one FP32 scale

llama.cpp Q8_0:
  [32 codes + s₀][32 codes + s₁][32 codes + s₂][32 codes + s₃]
  Each s is FP16; block boundaries follow components, not tokens.
```

This comparison applies specifically to the per-token/head modes. Conventional FP8 cache paths can instead use per-layer or calibrated per-head scales shared across tokens, as described in the FlashAttention profile. vLLM's INT4 path is asymmetric; TurboQuant and NVFP4 use other parameterizations. Likewise, llama.cpp's K-quants use a separate hierarchy, described earlier, and are not selectable KV-cache types at the inspected revision.

### Triton: attention paths

| Cache mode | Q preparation and attention arithmetic |
| --- | --- |
| Conventional E4M3 FP8 on supported CUDA | vLLM quantizes Q to FP8. The kernel retains FP8 K/V operands for `tl.dot` and applies scales separately. With non-FP8 Q, the alternative kernel branch reconstructs KV tiles into Q's dtype. [Capability][vllm-triton-query]; [conversion][vllm-kv-cast] |
| Per-token/head INT8 or FP8 | The common layer excludes these modes from Q quantization. Codes are cast to Q's model dtype; K scales multiply scores and V scales multiply softmax weights. [Q exclusion][vllm-query-quant]; [arithmetic][vllm-kv-compute] |
| Per-token/head INT4 | A dedicated kernel unpacks codes into floating-point operands and applies scale/zero-point corrections; Q and the result undergo the transforms below. [INT4 dispatch][vllm-int4-dispatch] |

For one INT8 key, $q^T\hat k=s_k(q^T\operatorname{cast}(k_{\mathrm{code}}))$. Applying the scale to the score reconstructs the result without scaling every cached component or materializing a full dequantized cache. V scales are incorporated before value accumulation. [Kernel arithmetic][vllm-kv-compute].

#### INT4 per-token/head workflow

**History.** JartX proposed the INT2/INT4 extension in [PR #39074][vllm-int4-origin] on April 6, 2026, already describing randomized Hadamard rotation for INT4. It was reorganized through [PR #40633][vllm-int4-parent], then narrowed to INT4 in [PR #40835][vllm-int4-pr] (opened April 24). The feature entered vLLM's `main` branch on **June 24, 2026**, as commit [`70749fdcca7d`][vllm-int4-merge]; this is the integration date, not a release date.

**Relation to TurboQuant.** Both use a change of coordinates before quantization, but the inspected INT4 PR descriptions and discussions provide no explicit attribution to TurboQuant. vLLM's separate [TurboQuant PR #38479][vllm-turbo-pr] merged on April 15, 2026; chronological proximity alone does not establish borrowing. Rotation for low-bit KV quantization also predates both integrations, notably in [QuaRot (2024)][quarot-paper]. The implementations differ: this INT4 path rotates K and V and uses uniform quantization, whereas vLLM's low-bit-key TurboQuant path uses centroid quantization for rotated K and leaves V unrotated, as detailed in the TurboQuant profile.

`int4_per_token_head` uses a **randomized Hadamard transform (RHT) on both K and V**, followed by asymmetric uniform quantization. The corresponding query and output transforms preserve attention's coordinate system, up to quantization and numerical error. [Public entry points][vllm-int4-flow].

```text
New K/V → RHT → per-token/head min–max quantization → packed INT4 cache
Q       → RHT → attention over unpacked cache tiles → inverse RHT → output
```

1. **Transform and quantize.** For each head vector, apply $T(x)=HDx$, where $D$ contains deterministic random signs and $H$ is an unnormalized Hadamard matrix. Compute $s=\max((x_{\max}-x_{\min})/15,10^{-6})$, $z=\operatorname{clip}(\operatorname{round}(-x_{\min}/s),0,15)$, and codes $u_i=\operatorname{clip}(\operatorname{round}(x_i/s+z),0,15)$. K and V have independent parameters. [Transform][vllm-int4-rht]; [quantizer][vllm-int4-pack].
2. **Store.** Pack two codes per byte. Store each four-bit zero point in the lowest four mantissa bits of its FP32 scale word; reading extracts the zero point and clears those bits, slightly truncating the scale. [Packing][vllm-int4-pack].
3. **Compute attention.** Transform Q but retain its floating-point dtype. Since $T(q)^T T(k)=d\,q^Tk$, divide the usual attention scale by head dimension $d$. Inside the kernel, unpack K/V codes into FP32 operands and apply scale/zero-point corrections: the key dot product is $s_k(\tilde q^Tu_k-z_k\sum_i \tilde q_i)$, where $\tilde q=T(q)$, while V scales and offsets are incorporated into weighted accumulation. INT4 describes the packed cache; the inspected dot products receive floating-point operands. [Attention loop][vllm-int4-read].
4. **Restore output.** Apply $T^{-1}=DH/d$ to the result because V was stored in transformed coordinates. The transform requires a power-of-two head dimension; the inspected dispatch requires causal attention. [Output transform][vllm-int4-flow]; [dispatch][vllm-int4-dispatch].

### Triton: constraints used by vLLM

The adapter's general checks require head dimension at least 32 and kernel blocks divisible by 16. CUDA initialization additionally rejects FP8 caches below SM89. Format-specific dispatch is stricter than the general adapter declarations: the inspected INT4 route requires causal attention and its transform requires a power-of-two head dimension. Consequently, passing an initial allowlist is not sufficient to guarantee a kernel call. [General checks][vllm-triton-shape-gates]; [CUDA FP8 check][vllm-triton-query]; [INT4 dispatch][vllm-int4-dispatch].

## FlashInfer and TRT-LLM: multiple attention implementations

The `FLASHINFER` backend name denotes a vLLM adapter, which can call several implementations distributed through the FlashInfer library:

| Term | Meaning in this discussion |
| --- | --- |
| Regular (“native”) FlashInfer | The adapter's ordinary prefill/decode wrappers; prefill can use FA2/FA3 kernels. “Native” names this integration route, not one fixed kernel or dtype. |
| XQA | A decode kernel family originating in TensorRT-LLM, particularly for MQA/GQA, exposed through FlashInfer's XQA API. |
| TRTLLM-gen | A separate TensorRT-LLM-derived kernel route exposed through FlashInfer, used here on SM100 for eligible prefill/decode. |
| `FLASH_ATTN` | A separate vLLM adapter. It should not be confused with FA2/FA3 kernels used inside FlashInfer. |

[vLLM integration][vllm-fi-kernel-imports]; [XQA origin][trtllm-xqa-origin]; [route selection][vllm-fi-decode-selection].

### FlashInfer: cache types

The ordinary `FLASHINFER` adapter lists `auto`, FP16/BF16, conventional FP8 (E4M3/E5M2), `nvfp4`, and `nvfp4_4over6`. It does not list Triton's per-token/head modes or TurboQuant presets. NVFP4 additionally requires SM100-family hardware and availability of TRT-LLM attention for both prefill and decode. These are vLLM integration conditions, not a claim that all listed FP8 variants work in every internal kernel. [Types and hardware gates][vllm-fi-shape-gates].

**Where TRT-LLM enters.** At the pinned revision, the general attention registry has no standalone `TRTLLM` entry. The `FLASHINFER` adapter imports TRT-LLM context/decode functions from FlashInfer. Separately, `TRTLLM_RAGGED` belongs to the **MLA-prefill registry**: its vLLM adapter imports `flashinfer.prefill.trtllm_ragged_attention_deepseek`. Thus a TRT-LLM kernel name identifies the implementation being called, not necessarily a separate kernel package or serving engine installed behind a direct vLLM adapter. [Standard route][vllm-fi-kernel-imports]; [MLA decode route][vllm-fi-mla-adapter]; [prefill registry][vllm-mla-prefill-registry]; [ragged adapter][vllm-trtllm-ragged-adapter].

### FlashInfer: storage layout

On SM100, the adapter declares `[LBHNC, BLHNC]`, in preference order. Both are head-major within a page; the former groups pages by layer and the latter groups layers by block. Other devices inherit `None` at this interface. Conventional FP8 uses separate scaling arguments; NVFP4 has additional page regions and scale-address permutations. Thus matching axis order does not make two formats' bytes interchangeable. [Layout declaration][vllm-fi-layout]; [payload views][vllm-fi-payload].

**NVFP4: separate regions within each page.** For head dimension 128, each K or V vector contributes 64 bytes of FP4 codes and eight FP8 scales, one per 16 components. Unlike Triton's inline records, the inspected writer groups data and scales into page-level regions. Let $T$ be tokens per page and $H$ the number of KV heads:

```text
One page, head dimension 128:
[ K data: 64TH bytes ][ K scales: 8TH bytes ]
[ V data: 64TH bytes ][ V scales: 8TH bytes ]
       ← these four regions occur consecutively in memory →

Within a vector, logical scale association:
  16 FP4 values (8 bytes) ↔ one FP8 scale (1 byte)
  repeated 8 times; scales reside in their separate region.
```

This is 144 bytes per token/head for K and V combined, before padding and separate global-scale metadata. The writer stores K scales linearly and permutes V scale addresses for the SM100 attention kernel. `nvfp4_4over6` uses the same layout with a different scale-selection rule. [Page regions and scale addressing][vllm-nvfp4-layout-write]; [tensor views][vllm-nvfp4-views].

NVFP4's groups contain 16 values. The `4over6` variant selects a scale based on the group maximum divided by 4 or 6 to reduce squared reconstruction error; its storage layout is unchanged. [Scale selection][vllm-nvfp4-config]; [writer][vllm-nvfp4-write].

### FlashInfer: attention paths

The vLLM adapter selects prefill and decode independently; Q dtype can therefore differ between phases while the persistent cache is unchanged. **The dtype passed into an API need not be the dtype used by its matrix instructions.** The XQA internals below are checked against FlashInfer `v0.6.18`.

| Inspected route with FP8 KV | Q at the API boundary | Attention computation |
| --- | --- | --- |
| SM90 regular FA3 prefill, Q quantization enabled | FP8 | FP8 Q/K/V operands |
| SM90 XQA, eligible E4M3 branch | FP16/BF16 | Converts Q to FP8 internally; uses FP8 tensor-core GMMA |
| Generic XQA, including the inspected SM12x route | FP16/BF16 | Converts cached K/V tiles to Q's dtype inside the kernel; uses FP16/BF16 MMA |
| SM100 TRTLLM-gen, Q quantization enabled | FP8 | Accepts FP8 Q and KV |
| TRTLLM prefill with Q quantization disabled | Model dtype | vLLM reconstructs the required FP8 KV into a temporary higher-precision buffer |

The adapter's `get_q_data_type(is_prefill)` determines the API input and can quantize only the relevant query slice. `disable_flashinfer_q_quantization` retains model-dtype Q **at that boundary**; it does not guarantee that a downstream kernel avoids internal FP8 conversion. [Q policy][vllm-fi-query-policy]; [slice conversion][vllm-fi-query-convert]; [prefill reconstruction][vllm-fi-nvfp4-query].

For XQA, FlashInfer enables `run_sm90_fp8_mha` for Hopper, E4M3 KV, and head dimension ≤256, then disables it for certain speculative-decode combinations. Its `F16QToF8Converter` converts Q, and `mha_sm90.cu` uses FP8 `MathElem` in GMMA with FP32 accumulation. The generic `mha.cu` instead converts K/V tiles and calls `mma<InputElem>`. Thus **XQA supports direct FP8 matrix operands on a specific path**; neither its name nor its FP16/BF16 query API implies one universal compute dtype. Softmax and accumulation are separate precision choices. [Dispatch][fi-xqa-api]; [Hopper kernel][fi-xqa-sm90]; [generic kernel][fi-xqa-generic].

For NVFP4, the inspected TRTLLM-gen call requires FP8 Q and supplies packed FP4 KV plus block scales. Those API dtypes alone do not establish the internal MMA operand types; NVFP4 kernel arithmetic remains outside this study. [NVFP4 call][vllm-fi-nvfp4-query].

### FlashInfer: constraints used by vLLM

There are two head-dimension checks, with different responsibilities:

| Check at the pinned vLLM revision | Accepted dimensions | Purpose |
| --- | --- | --- |
| `FLASHINFER` adapter allowlist | `{64,128,256,512}` | Declares which model head dimensions vLLM permits for this adapter as a whole. |
| Adapter's XQA routing predicate | `{16,32,…,256}` | Decides whether an admitted cache group can take the XQA route. |
| Both checks | `{64,128,256}` | Head dimensions eligible for XQA through this integration, subject to other constraints. |

For example, dimension 96 passes the XQA predicate but fails the adapter allowlist; dimension 512 passes the adapter check but takes regular FlashInfer decode. These sets are not subsets of each other, so calling one “narrower” is misleading. They describe vLLM's integration contract, not all upstream kernel capabilities: FlashInfer `v0.6.18` also permits dimensions above 256 through its generic XQA kernel under additional restrictions. [Adapter checks][vllm-fi-shape-gates]; [XQA routing guard][vllm-fi-decode-selection]; [upstream limits][fi-xqa-jit].

The adapter gates devices to SM80–SM121. Kernel pages are normally `{16,32,64}`; `{128,256,512,1024}` are additionally advertised for eligible SM100 GQA/MQA models with more query heads than KV heads and TRTLLM availability. [Shape/page checks][vllm-fi-shape-gates].

GPU generation alone does not select the internal kernel. Causality, query/KV-head ratio, context parallelism, explicit overrides, and phase also enter vLLM's FlashInfer dispatch, shown in the engine-specific tree below.

## TurboQuant: a quantization-specific backend

### TurboQuant: cache types

`TURBOQUANT` is a dedicated vLLM backend for selected variants of the quantization method. Its formats combine different key and value encodings:

| TurboQuant preset | Keys | Values |
| --- | --- | --- |
| `turboquant_k8v4` | FP8 | Uniform 4-bit |
| `turboquant_4bit_nc` | Nonuniform 4-bit | Uniform 4-bit |
| `turboquant_k3v4_nc` | Nonuniform 3-bit | Uniform 4-bit |
| `turboquant_3bit_nc` | Nonuniform 3-bit | Uniform 3-bit |

The low-bit key variants apply Hadamard rotation followed by Lloyd–Max scalar quantization; `nc` enables norm correction. The FP8-key preset bypasses that rotation/nonuniform-quantization path. These describe vLLM's implementation, whose configuration explicitly omits QJL, rather than every component of the TurboQuant paper. [Preset definitions][vllm-turbo-config].

### TurboQuant: storage layout

The adapter requires token-major `[LBNHC]`. The standard path uses complete per-token/head records; when FlyDSL is available, the implementation selects SoA packing and matching writers/readers. The axis-order enum alone therefore does not describe all field offsets. [Layout requirement][vllm-turbo-layout]; [packing selection][vllm-turbo-packing-select].

**TurboQuant: standard packed records.** For `turboquant_4bit_nc` with head dimension 128, one token/head occupies 134 bytes:

```text
[ K indices: 64B ][ K norm: 2B ][ V codes: 64B ][ V scale: 2B ][ V offset: 2B ]
    4 bits each       FP16        4 bits each       FP16            FP16
```

The value offset is the stored minimum, used as $\hat v=s_v u+v_{\min}$, rather than an integer zero point. Key centroids are shared separately. The byte counts below follow the pinned code, including per-record metadata and even-byte alignment, but excluding page padding and shared tables. The alternative structure-of-arrays path rearranges fields as shown next. [Record-size definitions][vllm-turbo-record]; [value writer][vllm-turbo-value]; [alternative layout][vllm-turbo-soa].

| Preset, head dimension 128 | Key payload + metadata | Value payload + metadata | Total per token/head |
| --- | --- | --- | --- |
| `turboquant_k8v4` | 128B FP8; no norm | 64B codes + 4B scale/offset | 196B |
| `turboquant_4bit_nc` | 64B indices + 2B norm | 64B codes + 4B scale/offset | 134B |
| `turboquant_k3v4_nc` | 48B indices + 2B norm | 64B codes + 4B scale/offset | 118B |
| `turboquant_3bit_nc` | 48B indices + 2B norm | 48B codes + 4B scale/offset | 102B |

TurboQuant's alternative structure-of-arrays (SoA) layout for low-bit keys is:

```text
One page:
  Code region: [K00|V00][K01|V01][K10|V10][K11|V11] ...
  Metadata:    head 0 → [all T norms][all T V scales][all T V offsets]
               head 1 → [all T norms][all T V scales][all T V offsets]
```

### TurboQuant: attention paths

vLLM's low-bit-key TurboQuant presets use a **normalized, deterministic Hadamard matrix** $R=H/\sqrt d$, without the random sign flips used in Triton's INT4 path. They normalize K before nonuniform quantization, while V is uniformly quantized in its original coordinates. [Rotation construction][vllm-turbo-rotation]; [cache writer][vllm-turbo-write].

```text
K → separate norm and direction → Hadamard → centroid indices + norm → cache
V → min–max uniform quantization → packed codes + scale/offset      → cache
Q → Hadamard → centroid-based scores → softmax → reconstructed V sum → output
```

1. **Encode keys.** Compute $\rho=\lVert k\rVert_2$ and $y=R(k/\rho)$, with a small denominator stabilizer. Map each coordinate to a 3- or 4-bit Lloyd–Max centroid index and store the packed indices plus an FP16 norm. The shared codebook is derived from a Gaussian coordinate model with variance $1/d$, rather than fitted independently to each token. [Normalization and encoding][vllm-turbo-write]; [codebook construction][vllm-turbo-centroids].
2. **Encode values.** Apply 3- or 4-bit min–max quantization to each unrotated V vector. Store packed codes with FP16 scale and minimum-value offset, reconstructing $\hat v=s_v u_v+v_{\min}$. The `turboquant_k8v4` preset instead casts K to FP8 directly, bypassing key normalization, rotation, and centroid encoding; its V path remains uniform 4-bit. [Value and FP8-key writers][vllm-turbo-value].
3. **Decode attention.** Transform Q with the same $R$. Unpack key indices, gather centroid values $c$, and compute scores proportional to $\rho\,(Rq)^Tc$. The `nc` presets replace $c$ by $c/\lVert c\rVert_2$; an alternative cache layout folds this correction into the stored norm during insertion. Unpack/dequantize V during the softmax-weighted accumulation. **No inverse output rotation is needed**, because V was never rotated. The FP8-key preset uses unrotated Q. [Decode kernel][vllm-turbo-read]; [query preparation][vllm-turbo-query]; [norm-correction placement][vllm-turbo-soa].

Prefill has a separate path: an initial prompt uses raw, uncompressed Q/K/V for attention while populating the compressed cache. In the inspected continuation path, chunks of at most 128 query tokens use compressed-cache decode kernels; larger chunks reconstruct the cached prefix into FP16 buffers, inverse-rotate low-bit keys, and combine it with the current chunk's raw K/V for attention. Thus avoiding a full cache reconstruction is a property of the decode path, not every TurboQuant invocation. [Prefill and continuation dispatch][vllm-turbo-prefill]; [prefix reconstruction][vllm-turbo-continuation].

### TurboQuant: constraints used by vLLM

The adapter accepts FP16/BF16 model inputs, decoder attention, and kernel block sizes `{16,32,64,128}`. Its `supports_quant_query_input` is `False`: the common attention layer does not convert Q to FP8, even for the FP8-key preset. Query transformation and reconstruction are handled by the method-specific paths above. [Adapter declarations][vllm-turbo-gates]; [implementation][vllm-turbo-impl].

## Other backend families: scope for further study

The four profiles above cover the inspected ordinary-attention paths. The registry also contains the following families, which add distinct implementation or hardware cases:

| Family | Representative adapters | Relevance to this investigation |
| --- | --- | --- |
| PyTorch attention | `FLEX_ATTENTION` | Connects vLLM to PyTorch FlexAttention, adding another kernel-generation route. [Adapter][vllm-flex-adapter] |
| Composite adapters | `TRITON_FLASH_ATTN`, `TRITON_FLASHINFER` | Combine existing implementations within one adapter; adapter names need not map one-to-one to libraries. [Example][vllm-composite-adapter] |
| MLA and sparse attention | `FLASHMLA`, `FLASHINFER_MLA`, `CUTLASS_MLA`, `TRITON_MLA`, plus sparse variants | Require a separate study of compressed latent states, layouts, and prefill/decode integration. [Registry][vllm-attention-registry] |
| Platform-specific implementations | `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`, `CPU_ATTN` | Extend the comparison to AMD/AITER and CPU paths rather than assuming CUDA behavior applies everywhere. [Registry][vllm-attention-registry]; [AITER adapter][vllm-aiter-adapter] |

This is an orientation map, **not a quantization-support matrix**. The detailed profiles remain focused on the verified routes; additional adapters should be compared through the same sequence—cache representation, writer, attention arithmetic, and dispatch—before claiming equivalent format support or performance.

## vLLM: backend selection, kernel dispatch, and cache management

The preceding profiles describe individual integrations. **The following decision trees belong to vLLM**, not to FlashAttention, FlashInfer, or another engine. Three related decisions must be distinguished: adapter selection, kernel-path/input-dtype selection, and layout resolution for allocation. Backend and layout choices are established during initialization; phase- and workload-dependent dispatch can occur later for each batch. Kernel libraries may perform further specialization internally.

### vLLM: selecting an attention backend

The selector collects model and execution requirements, including model dtype, actual cache type, head dimension, page size, attention kind, masks, sinks, scale granularity, and parallelism. A per-cache-kind backend override takes precedence over the global choice. The CUDA platform then validates candidate adapters. [Selector inputs][vllm-selector-inputs]; [validation][vllm-backend-validation].

```text
vLLM CUDA selection (ordinary non-MLA attention)
  ├─ Explicit backend, including a per-cache-kind override
  │    ├─ Import + hardware/shape/dtype/feature checks pass → select it
  │    └─ Any required check fails                       → error
  └─ Automatic backend
       ├─ SM100 family, causal configuration
       │    → FLASHINFER → FLASH_ATTN → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT
       └─ Other ordinary CUDA configurations
            → FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT
       For each candidate:
         package/build available?
         GPU + model dtype + cache type + head/page sizes supported?
         attention features + scale granularity + parallelism supported together?
           → retain compatible candidates; choose highest priority
           → none compatible: error
```

The lists are **priorities, not unconditional routing**: choosing INT4 or TurboQuant removes incompatible higher-priority candidates. Multimodal-prefix attention can prepend composite adapters, and MLA/sparse models use separate candidate lists. This is compatibility filtering, not runtime benchmarking; later initialization and kernel assertions can impose additional checks. [CUDA priorities][vllm-cuda-backend-priorities]; [candidate selection][vllm-cuda-backend-selection].

### vLLM: selecting a kernel path within the backend

For FlashAttention, vLLM first resolves the library version, then validates the cache/version combination and prepares Q. The following is the core rule, with representative feature-dependent adjustments:

```text
FLASH_ATTN selected
  → Default version: SM90 + available FA3 → FA3
                     SM100 + available FA4 → FA4
                     otherwise → FA2
  → Apply requested version and feature/shape compatibility rules
       Examples: ALiBi → FA2
                 SM90 head dimension > 256 → FA4 if available
                 FA4 + batch invariance → FA2
                 unsupported Blackwell FA4 head shape → FA2
  → Actual cache type
       FP16/BF16 → matching higher-precision Q/K/V
       E4M3 FP8 → require FA3/SM90 or FA4/SM100
                    pass → vLLM quantizes Q → FP8 Q/K/V kernel
                    fail → incompatible backend; no same-adapter dequant fallback
  → Library specializes for input dtype, head shape, masks, and sequence shape
```

The version checks also cover FA4's dedicated head-dimension-256 path, its page requirements, and other feature combinations; the source is authoritative for the full predicate list. The library's separate FA4 `fp8_kv_dequant` capability is not enabled by this wrapper. [Version resolver][vllm-fa-version-policy]; [cache gate][vllm-fa-fp8-gate]; [wrapper][vllm-fa4-wrapper].

FlashInfer's ordinary, non-cascade path makes **separate prefill and decode decisions**. Let `h_q` and `h_kv` be query and KV head counts. Eligible TRTLLM/XQA calls require available kernels, an enabled configuration, and `h_q % h_kv == 0`:

```text
FLASHINFER selected (non-cascade path)
  → Establish target Q dtype per phase from GPU, cache type, and Q-quantization option
  ├─ Prefill
  │    ├─ Non-causal, DCP enabled, explicit disable, or unsupported TRTLLM platform
  │    │    → regular FlashInfer path
  │    └─ Eligible causal SM100 path
  │         ├─ FP8 Q, attention sinks, explicit enable, or kernel page >= 128
  │         │    → TRTLLM-gen prefill
  │         ├─ Auto policy + cache dtype "auto" → TRTLLM-gen prefill
  │         └─ Otherwise → regular FlashInfer prefill
  └─ Decode
       ├─ Eligible SM90/SM12x → XQA
       │    unsupported XQA head dimension or DCP → regular FlashInfer
       ├─ Eligible SM100 + causal attention → TRTLLM-gen
       └─ Otherwise → regular FlashInfer
  → Quantize each query slice to its phase's target dtype when needed
  → Invoke that kernel with its cache views, scales, and addressing metadata
```

Here DCP denotes decode context parallelism. Eligibility also depends on batch-invariance mode and kernel availability; invalid feature combinations can error rather than fall back. The two head-dimension checks above admit `{64,128,256}` for XQA through this adapter, while dimension 512 falls back per cache group. Q dtype follows the phase-specific policy in the FlashInfer profile. [Eligibility and prefill policy][vllm-fi-path-policy]; [decode checks][vllm-fi-decode-selection]; [phase dispatch][vllm-fi-phase-dispatch]; [Q policy][vllm-fi-query-policy].

**This is a vLLM dispatch policy, not a universal FlashInfer ranking of XQA > TRTLLM-gen > FlashAttention.** For decode, vLLM prefers the eligible hardware-specific route—XQA on SM90/SM12x or TRTLLM-gen on SM100—over the regular wrapper. Prefill is selected separately. FlashInfer then dispatches internal kernel specializations, including XQA's FP8-versus-16-bit arithmetic. Selecting the separate `FLASH_ATTN` adapter belongs to the earlier backend-selection stage. These branches establish defaults and compatibility, not a performance ordering across workloads. [Decode selection][vllm-fi-decode-selection]; [internal XQA dispatch][fi-xqa-api].

For the remaining two profiles, the cache format selects the major numerical path; workload shape can further select an implementation:

```text
TRITON_ATTN
  ├─ int4_per_token_head → causal INT4 kernel + query/output transforms
  ├─ int8/fp8_per_token_head → model-dtype Q + reconstruction in attention
  └─ conventional E4M3 FP8 on supported CUDA → FP8 Q/K/V dot products

TURBOQUANT
  ├─ Initial prompt → uncompressed attention + compressed-cache insertion
  ├─ Decode / continuation with <= 128 query tokens → compressed-cache kernels
  └─ Larger continuation → reconstruct cached prefix into FP16 buffers
       + attend with current raw K/V
  Packing/kernel implementation also follows Triton versus FlyDSL availability.
```

These are the inspected branches, not a universal choice between “decode” and “prefill” kernels for all models. [Triton dispatch][vllm-int4-dispatch]; [query conversion][vllm-query-quant]; [TurboQuant dispatch][vllm-turbo-prefill].

### vLLM: page ownership, token addressing, and allocation

**Paging and quantization define different groups.** A vLLM cache page covers multiple tokens for allocation and reuse; a quantization group contains numerical components sharing reconstruction parameters. At the pinned revision, three structures connect a request to its cached values:

| Layer | Data structures | Role |
| --- | --- | --- |
| Allocation and reuse | `KVCacheBlock`, `BlockPool`, free-block queue, prefix-hash map | Track block IDs, reference counts, and reusable prefixes; these objects hold management metadata |
| Token addressing | INT32 block table; INT64 slot mapping | Map each request's logical token positions to physical cache slots |
| Numerical storage | `AttentionSpec`, `KVCacheTensor`, and strided `torch.Tensor` views | Describe formats, allocation offsets, and the packed K/V payload |

[Block metadata][vllm-block-metadata]; [block tables][vllm-block-table]; [allocation descriptor][vllm-cache-tensor]. For ordinary attention without context parallelism, addressing is:

```text
token position t
  → logical block = t // block_size
  → physical block = block_table[request, logical block]
  → slot = physical block × block_size + t % block_size
  → K/V record for each KV head at that slot
```

Consecutive logical blocks can occupy nonconsecutive physical blocks. Allocation blocks may also be subdivided into smaller kernel blocks. [Slot calculation][vllm-slot-mapping]; [block subdivision][vllm-block-table].

The allocator creates the worker's shared backing buffer for **persistent cache storage**, covering many pages and layers:

```python
buf = torch.zeros(buf_size, dtype=torch.int8, device=device)
```

`buf_size` is the total allocation in bytes, not one page's size. Page sizes are computed from the cache specifications before allocation. Paging assigns and reuses regions within this backing storage; it does not require a separate GPU allocation per page or request. Here `int8` provides byte storage; the **cache format is specified separately**. The allocator returns per-layer views with logical byte shape `[B, H, N, C]`: physical blocks, head slots, stored states (normally tokens), and content bytes. `create_kv_cache_views()` uses `torch.as_strided()` and dtype reinterpretation without copying the backing storage. [Allocator][vllm-cache-allocate]; [view construction][vllm-cache-views].

### Cache management: vLLM pages versus llama.cpp cells

**Both reuse preallocated capacity and permit logically consecutive tokens to occupy nonconsecutive locations.** Their main distinction is allocation granularity and how attention locates a request's data. For the ordinary paths inspected here:

| Aspect | llama.cpp | vLLM |
| --- | --- | --- |
| Allocation unit | One token cell | A page containing multiple tokens |
| Addressing | Cell positions and sequence IDs; attention views over a stream's slot range with masks | Per-request block tables map logical blocks to physical pages read by paged kernels |
| Sharing and reuse | Several sequence IDs can reference a cell; reclaim individual cells | Requests can reference shared blocks; reference counts, hash lookup, and a free-block queue manage reuse |

[llama.cpp metadata][llama-cache-cells] and [masks][llama-cache-mask]; [vLLM block tables][vllm-block-table] and [block pool][vllm-block-pool].

```text
llama.cpp unified stream, illustrative interleaved cells:
  slots:          [A0][B0][A1][B1]
  visible to A:   [yes no  yes no ]

vLLM, illustrative two-token pages:
  page 0: [B0 B1]   page 1: [free]   page 2: [A0 A1]
  A's block table: [2] → access page 2
  B's block table: [0] → access page 0
```

Layer/head details are omitted. vLLM's block tables provide request-specific traversal, while pages group allocation, sharing, and eviction operations across multiple tokens. Its automatic prefix cache uses hashes to discover reusable blocks. These mechanisms support concurrent serving; pooling and prefix sharing themselves are not exclusive to vLLM. [Paged traversal][vllm-paged-design]; [prefix reuse][vllm-prefix-design].

The tradeoff is granularity: a request's final partial page can leave unused token capacity, whereas llama.cpp can reuse individual cells. Conversely, a unified llama.cpp attention view can span unrelated or empty cells that are masked; actual work on those regions depends on kernel optimizations. The designs alone do not establish a performance winner.

### vLLM: resolving a compatible page layout

**Layout selection is an initialization policy, not runtime benchmarking.** After attention backends have been selected, each worker reports an ordered list compatible with its backends. The engine resolves one layout before memory profiling and broadcasts it to workers before cache allocation. A previously resolved configuration is reused. [Worker reporting][vllm-layout-worker]; [engine initialization][vllm-layout-engine].

```text
Selected attention backends
  → Collect their layout declarations; ignore None entries
  → Intersect declared sets; no common layout → error
    If none declares a set, use defaults headed by LBNHC
  → Require workers to report identical ordered candidates
  → Filter for cache-spec packing compatibility
  → Choose:
      VLLM_KV_CACHE_LAYOUT set?
        → use it if compatible; otherwise error
      Else, KV-transfer connector requests a layout?
        → use it if compatible; otherwise warn and use first candidate
      Else → use first candidate
  → Record layout, broadcast to workers, allocate strided views
```

Identical backend lists retain their order. When lists differ, their intersection is ranked by how many backends list each layout first; ties follow the layout enum's order. Mixed head/state/page-size specifications further require each layer's page interior to occupy a contiguous region. The environment aliases `NHD` and `HND` mean `LBNHC` and `LBHNC`, respectively. These checks constrain ordering in addition to each backend's dtype, shape, and hardware checks. [Candidate ranking and resolver][vllm-layout-resolver].

For example, Triton plus FlashAttention impose no ordering preference, so the unmodified default is token-major. Adding FlashInfer on SM100 narrows candidates to head-major layouts. Combining that FlashInfer declaration with TurboQuant's token-major-only declaration produces an empty intersection and fails this resolver; it does not automatically repack between the two layouts or reselect a backend. [Intersection logic][vllm-layout-resolver].

### vLLM: model loading and quantization configuration

vLLM separates **weight/activation quantization** from **KV-cache quantization**. Its integrations support weight-only schemes such as INT4 W4A16, weight–activation schemes such as INT8/FP8 W8A8, and other hardware-dependent formats. A quantization configuration selects a layer-specific method, whose interface covers parameter creation, post-load processing, and execution. AWQ, GPTQ, and compressed-tensors consequently represent different preparation/loading paths rather than one universal low-bit kernel. [Overview][vllm-quant-doc]; [method interface][vllm-quant-interface].

vLLM also supports **load-time quantization** of floating-point weights, with dynamic activation quantization during inference. This differs from loading an already quantized checkpoint. The scheme and execution backend jointly determine operand precision; a low-bit weight format alone does not establish the activation or KV-cache dtype. [Online quantization][vllm-online-doc].

## SGLang: cache formats, storage, and attention paths

This section follows commit `2f5c9ac43d76` (2026-09-25), primarily ordinary CUDA MHA/GQA without context parallelism. Read the design in three stages: **the recipe determines codes and scales; the pool stores them; the attention adapter selects how they are consumed**. Request allocation and prefix reuse are separate from these numerical choices. MLA, recurrent/hybrid models, CPU/ROCm paths, and offloading require additional contracts.

### SGLang: cache types and quantization algorithms

The inspected CUDA paths distinguish four quantized families; `auto` normally retains the model's cache precision. These are scaled floating-point encodings, not a common symmetric-integer format:

| Cache option | Codes and reconstruction | Scale granularity |
| --- | --- | --- |
| `fp8_e4m3`, `fp8_e5m2` | Cast $x/s$ to the selected FP8 format; reconstruct as $s q$ | Separate scalar K/V scales per layer, shared across tokens and heads; checkpoint/JSON values or unity |
| `mxfp8` | E4M3 codes with an exponent-only UE8M0 scale: $\hat x=2^e q$ | One dynamically computed scale per 32 adjacent head components, separately for each token/head and K/V |
| `nvfp4` | Packed E2M1 codes with two-level scaling: $\hat x=g\,s_b q$ | Dynamic E4M3 scale $s_b$ per 16 components, plus separate per-layer FP32 global scales $g$ for K/V |
| `fp4_mx_block16` | Packed E2M1 codes with power-of-two scaling: $\hat x=2^e q$ | One dynamic exponent byte per 16 components; no NVFP4 global scale |

Conventional FP8 conversion occurs in the pool writer unless inputs have already been converted. MXFP8 chooses a power-of-two scale from the group's maximum magnitude; `fp4_mx_block16` similarly uses $e=\lceil\log_2(\max|x|/6)\rceil$, with a small-value clamp. The latter stores a biased exponent byte and is **not standard block-32 MXFP4**. The quantization groups run along components, not allocation pages or token sequences. [FP8 writer][sg-pool]; [scale configuration][sg-quant-doc]; [MXFP8 quantizer][sg-mxfp8-quant]; [FP4 algorithms][sg-fp4-algorithms].

### SGLang: physical storage and indexing

**Start with the ordinary NHD pool.** Its management and numerical storage have distinct roles:

| Component | Responsibility | Representation |
| --- | --- | --- |
| `ReqToTokenPool` | Assign a request row and record its token locations | GPU integer table `req_to_token[request_row, position]` |
| `TokenToKVPoolAllocator` / `PagedTokenToKVPoolAllocator` | Assign and reclaim locations inside existing capacity | Free token IDs when `P=1`; free page IDs when `P>1` |
| `MHATokenToKVPool` | Own K/V tensors and write new values, including dtype conversion | Separate K and V tensors for each local layer |
| `RadixCache` | Find reusable token prefixes and protect or evict their locations | Radix-tree keys contain token sequences; values contain cache-location IDs |
| Attention adapter | Convert request mappings into kernel metadata and select cache views | Page tables or packed index arrays, lengths, scales, and tensor views |

The request-table row and a KV location are **different indices**: freeing a request row does not imply that its reusable prefix must disappear. [Request and storage pools][sg-pool]; [allocators][sg-paged-alloc]; [radix cache][sg-radix].

The ordinary pool is preallocated from the available memory budget, subject to token-capacity and page-alignment constraints. Allocation during serving normally assigns locations within this capacity. With `T` usable token slots, `P` tokens per allocation page, `H` local KV heads, and component dimensions `Dk/Dv`, its default storage is equivalent to this pseudocode:

```python
# Separate contiguous allocations; adjacency between these tensors is not guaranteed.
K = [zeros((T + P, H, Dk), storage_dtype) for layer in local_layers]
V = [zeros((T + P, H, Dv), storage_dtype) for layer in local_layers]

u = req_to_token[request_row, token_position]
key   = K[layer][u, head, component]
value = V[layer][u, head, component]
page_id, offset = u // P, u % P
```

**Page 0 is reserved for dummy KV writes from padded execution.** For example, a CUDA graph captured for four requests can replay a batch containing three real requests and one dummy entry:

```text
Execution batch:    [Request A][Request B][Request C][Dummy]
KV write location: [real slot][real slot][real slot][  0  ]
                                                    ↓
                                             reserved page 0
```

The graph-buffer policy sets padded `out_cache_loc` entries to zero, giving the cache writer a valid destination without overwriting real request data. Because allocation operates on whole pages, the allocator excludes page ID 0: slots `0…P−1` are reserved, and real allocation begins at page 1. For `P=1`, only one slot is reserved. Its contents need not remain zero; this is scratch space for dummy work, distinct from unused positions in a real request's partially filled page. [Graph padding policy][sg-graph-padding]; [reserved-page allocation][sg-paged-alloc].

Within each ordinary K tensor, components are contiguous, followed by heads, then token slots; V has its own allocation. A location `u` identifies corresponding rows across layers and K/V—it does **not** identify one contiguous record containing all layers. [Pool sizing][sg-pool-config]; [buffer construction and writes][sg-pool].

**K and V are separate, not interleaved.** A page ID identifies corresponding token ranges in two allocations within each layer; it does not identify one contiguous K-plus-V record. The default NHD ordering is:

```text
One layer l, two separate allocations:

K allocation: [K page 0][K page 1][K page 2] ...
V allocation: [V page 0][V page 1][V page 2] ...
              Same page IDs; separate memory regions

Zoom into page p (two KV heads shown):

K page p: [token 0: head 0 components | head 1 components]
          [token 1: head 0 components | head 1 components] ...

V page p: [token 0: head 0 components | head 1 components]
          [token 1: head 0 components | head 1 components] ...

Token offset o in page p → slot u = p * P + o
  key vector   = K[l][u, h, :]
  value vector = V[l][u, h, :]
```

Each page above is a range inside its parent allocation, not a separate GPU allocation. HND variants change the order within each K/V tensor while retaining their separation; quantization-specific scale buffers are described below.

For example, let `P=4` and two requests share their first four tokens:

```text
Logical position:       0   1   2   3    4   5
req_to_token[A, :6] = [ 8,  9, 10, 11,  20, 21 ]
req_to_token[B, :5] = [ 8,  9, 10, 11,  12     ]

Physical pages, repeated in EVERY layer's separate K and V arrays:
  page 0: slots  0.. 3    reserved for padding
  page 2: slots  8..11    shared prefix
  page 3: slots 12..15    B's suffix; three positions still unused
  page 5: slots 20..23    A's suffix; two positions still unused

Example: A's position 4, layer l, head h
         → slot 20 → K[l][20,h,:] and V[l][20,h,:]
```

Thus logically adjacent tokens can cross nonadjacent physical pages. Unused positions in a request's partial page remain part of that allocated page. Other pool layouts change the address calculation: HND uses `[page, head, offset, component]`, and ROCm AITER has a vectorized layout. Unified-memory variants additionally translate virtual IDs into kernel-facing IDs. These require compatible consumers; a different view alone cannot arbitrarily change physical ordering. [Layouts][sg-pool]; [index translation][sg-index].

**Quantization adds format-specific payload and scale buffers.** For equal K/V head dimension $D$, let $U=T+P$ include padding:

| Format | Per-layer data allocations | Separate scale allocations |
| --- | --- | --- |
| Conventional FP8 | `K[U,H,D]`, `V[U,H,D]`; stored as `uint8` bytes and exposed as FP8 views | Layer-level K/V scalars; no per-token scale records |
| MXFP8 | `K[U,H,D]`, `V[U,H,D]` in E4M3 | One UE8M0 byte per 32 components; production page-128 path uses `[pages,H,32,4,D/32]` |
| NVFP4 | `K[U,H,D/2]`, `V[U,H,D/2]` in packed bytes | Linear `[U,H,D/16]` and/or kernel-specific `[pages,H,P,D/16]` E4M3 scale buffers |
| FP4 block-16 | `K[U,H,D/2]`, `V[U,H,D/2]` in packed bytes | Exponent-byte arrays `[U,H*D/16]` for K and V |

For $D=128$, the logical association for **one token/head** is:

```text
                         K allocation       K-scale allocation
Conventional FP8:       [128 code bytes]     scalar shared across the layer
MXFP8:                  [128 code bytes]     [4 exponent bytes]
NVFP4 / FP4 block-16:   [ 64 code bytes]     [8 scale bytes]

V has its own data and scale allocations with the same logical association.
NVFP4 additionally uses a separate global scale for each layer's K and V.
These boxes describe associated values, not adjacent memory regions.
```

Scale order is part of the kernel contract. MXFP8 interleaves scale rows: token offset `o` maps to the `[o % 32, o // 32]` axes in a 128-token page. SM100 NVFP4 keeps data in NHD but presents an HND view through strides; its native K scales are head-major and V scales use a four-token permutation. The mixed NVFP4 recipe maintains both native and linear scales plus an FP8 prefill workspace shared across layers. Native-only SM100 execution omits those compatibility buffers. [MXFP8 pool][sg-mxfp8-pool]; [FP4 buffers and memory accounting][sg-fp4-method]; [NVFP4 scale addressing][sg-nvfp4-storage].

**The four-token permutation rearranges V-scale bytes, not the packed V values.** For one head with `D=64`, each token has four scales, one per 16 components. Let A–D denote four adjacent token slots within a page, and `a0…a3` the scales belonging to A:

```text
Logical token-major scales           Native V-scale storage

Token A: a0 a1 a2 a3                 Row 0: a0 b0 c0 d0
Token B: b0 b1 b2 b3       →         Row 1: a1 b1 c1 d1
Token C: c0 c1 c2 c3                 Row 2: a2 b2 c2 d2
Token D: d0 d1 d2 d3                 Row 3: a3 b3 c3 d3
```

A physical scale row therefore no longer belongs to one token. The kernel uses the matching address rule; scale values and their token associations remain unchanged, with no overlapping writes. For general head dimension `D`, logical token offset `t` and scale index `j` map to:

```python
g = (D // 16) // 4
physical_row = (t // 4) * 4 + j // g
physical_col = (j % g) * 4 + t % 4
# Coordinates within native_V_scales[page, head, :, :]
```

This permutation satisfies TRT-LLM GenMHA's native NVFP4 scale contract. K scales retain token-major order within each page/head; the permutation is not shared by all SGLang cache formats. [Index mapping and incremental writer][sg-nvfp4-storage].

**Implication: choose numerical groups independently of physical ordering.** Here V is quantized in 16-component groups within each token/head, then its scale bytes are permuted for the consumer. Transposing a tensor need not change these groups; grouping across tokens after a transpose would instead define a different quantization scheme.

The clearest efficiency benefit is **independent insertion**: given the configured global scale, a new token's block scales and codes require only its own values, and the writer scatters them directly into the required layout. This avoids buffering or revising dynamic cross-token groups. Reconstruction likewise needs only that token's codes and associated scales. However, a precomputed scale shared across tokens also permits independent reconstruction; token-local grouping does not uniquely enable it. GPU kernels still load tiles spanning tokens, and interleaved scale bytes may share memory transactions. The layout therefore does not guarantee isolated physical reads or direct FP4 arithmetic. [Quantization and insertion][sg-fp4-method]; [scale scatter][sg-nvfp4-storage].

For quality, token-local scales can prevent one token's magnitude range from setting another's quantization range. KIVI provides empirical evidence favoring per-token V quantization in its evaluated setting, but this does not establish that SGLang's NVFP4 grouping is universally optimal. The supported inference is that **token-local groups simplify online updates, while the subsequent permutation accommodates kernel access without changing numerical accuracy**. [KIVI study][kivi-paper].

### SGLang: attention paths

Here **the `flashinfer` wrapper route** means SGLang's adapter using FlashInfer's paged prefill/decode wrappers. It is distinct from SGLang's `trtllm_mha` adapter, which calls TRT-LLM generation attention or XQA through FlashInfer APIs. The library name therefore does not uniquely identify the SGLang backend, nor imply that it is the default. [Wrapper construction][sg-fi]; [TRT-LLM adapter][sg-trtllm].

An adapter first translates token locations into the addressing scheme expected by its kernel. An **allocation page** groups ownership; a **kernel page** groups addresses, and can have a different size:

| Adapter, ordinary MHA route | Cache addressing |
| --- | --- |
| FlashAttention | Separate K/V views `[page,P,H,D]`; page table derived from every `P`-th request-table entry divided by `P` |
| `flashinfer` wrapper decode | Separate NHD K/V arrays; packed token-location IDs and row boundaries; wrapper planning uses kernel page size **1** |
| Triton, ordinary decode | Separate K/V arrays; packed token-location IDs, row boundaries, and allocation page size |
| `trtllm_mha` | Page tables and route-specific NHD/HND views; native FP4 additionally passes the required scale buffers |

**FlashInfer also uses page IDs here, but its kernel pages contain one token.** SGLang's `flashinfer` decode adapter explicitly plans with kernel page size 1, so a physical token-slot ID is also a one-token page ID. This is an integration choice, not a FlashInfer limitation. For the earlier request A, using an illustrative four-token allocation page:

```text
Physical token slots: [8, 9, 10, 11, 20, 21]

Integration                     Kernel page size   IDs passed for this request
SGLang → FlashAttention                 4            [2, 5]
SGLang → FlashInfer wrapper             1            [8, 9, 10, 11, 20, 21]
vLLM   → ordinary FlashInfer            4            [2, 5]
                         (assuming four-token kernel pages for comparison)
```

FlashAttention supplies a page-table row and sequence length 6. FlashInfer packs requests' page-ID lists into one array with `indptr` boundaries; the final page length is 1 for SGLang's one-token pages, versus 2 in the illustrated vLLM case. Thus the same `indices` API can describe different address units. Adapting metadata and tensor views need not move K/V bytes. [SGLang planning][sg-fi]; [FlashAttention views][sg-fa]; [vLLM page metadata][vllm-fi-page-metadata].

**Why use one-token pages?** The functional benefit is a direct match to SGLang's request-to-token table: arbitrary token locations can be represented without requiring contiguous groups, including when allocation itself uses `P=1`. The ordinary FlashAttention route instead summarizes contiguous groups with one page ID. The trade-off is approximately one index per token versus one per page. This explains the integration's flexibility and metadata cost; it is an inference from the addressing contract, not evidence that one-token pages are faster.

One-token kernel pages still allow GPU computation over tiles containing many tokens. Conversely, larger kernel pages expose contiguous runs with fewer indices, but that advantage alone does not determine latency. **Prefix reuse follows allocation/cache-management granularity:** a one-token kernel view does not recover token-level sharing when reusable prefixes are managed in larger pages. Thus backend selection must also consider supported precision/features, hardware-specific kernels, head shapes, batch size, and prefill/decode workload. [Read metadata][sg-fi]; [prefix-cache alignment][sg-radix].

**Comparison with vLLM.** Its inspected ordinary paged reads use block/page IDs: FlashAttention, Triton, and TurboQuant consume block tables; regular FlashInfer receives flattened page IDs, while XQA/TRTLLM-gen use block tables directly. This does not mean every operation uses page IDs: cache insertion uses token-slot mappings, kernel pages may subdivide allocation pages, and some prefill/reconstruction routes consume dense tensors. The relevant distinction is the unit addressed by an index, independently of its array name or the allocation unit. [vLLM FlashInfer routing][vllm-fi-address-routing]; [block subdivision][vllm-block-table]; [token-slot mapping][vllm-slot-mapping].

**Conventional FP8: the backend and shape determine the operands.** Three possibilities must be distinguished: FP8 dot-product operands, reconstruction inside attention, and reconstruction into a temporary buffer before attention. An FP8 cache alone does not identify which occurs.

| Inspected CUDA route | Query preparation and attention arithmetic |
| --- | --- |
| Supported FA3 paged FP8, head dimension ≤256 | The adapter casts Q to the cache dtype and passes K/V descales; the supported kernel consumes FP8 operands |
| `flashinfer` wrapper tensor-core decode, FA2 implementation | The adapter keeps Q in the model dtype; the kernel converts FP8 K/V fragments to Q's FP16/BF16 dtype before MMA |
| Triton decode, one Q head per KV head | Vector multiply/reduction over cache values, with scale corrections; this is not the grouped FP8 matrix-dot path |
| Triton decode, multiple Q heads per KV head | The CUDA kernel casts Q to K's dtype before `tl.dot`; it also casts softmax probabilities to V's dtype before the value dot product. An FP8 cache therefore supplies FP8 operands here, even though the adapter did not cast Q |
| `trtllm_mha`, ordinary decode | SM100 uses TRT-LLM generation attention and prepares FP8 Q for an FP8 cache. SM90/SM120 use XQA with model-dtype Q at the API; internal operands depend on XQA's architecture specialization |

For XQA, API query dtype is insufficient evidence: the SM90 E4M3 specialization can quantize Q internally for FP8 MMA, whereas the SM120 path reconstructs cache operands to FP16/BF16. These are the kernel distinctions established in the earlier FlashInfer profile. [FA query preparation][sg-fa]; [FlashInfer adapter][sg-fi] and [FA2 fragment conversion][fi-fa2-kv-read]; [Triton decode kernels][sg-triton-decode]; [TRT-LLM dispatch][sg-trtllm]; [SM90 XQA][fi-xqa-sm90] and [generic XQA][fi-xqa-generic].

For conventional scalar-scaled KV, Triton folds the K scale into the score calculation and applies the V scale to the accumulated result; it need not multiply each cached component by its scale first. Operand precision remains distinct from accumulation and softmax precision. The FA3 row is deliberately scoped: FA4 prefill skips that adapter's ordinary Q-cast branch, so its support cannot be inferred from FA3's behavior.

**MXFP8: one attention kernel combines two arithmetic paths.** The inspected Blackwell FA4 route expects E4M3 Q/K with per-32-component UE8M0 scales, including query scales supplied by the model path. Its paged kernel also consumes the block-scaled V cache:

```text
Q codes + Q scales ─┐
K codes + K scales ─┴→ block-scaled FP8 QKᵀ → softmax
V codes + V scales ──→ reconstruct V to BF16 inside the kernel
                                           → BF16 probability × V → output
```

Here the second product multiplies softmax probabilities by V. Thus “MXFP8 attention” does not mean that both matrix products use FP8 operands. The implementation uses `tcgen05` block-scaled QK MMA, while its correction warp reconstructs V into shared memory for BF16 PV. The integrated pool requires head dimensions divisible by 32 and uses 128-token pages on the inspected production path. [Adapter scale contract][sg-fa]; [FA4 interface][sg-fa4-interface]; [SM100 kernel][sg-fa4-sm100]; [MXFP8 pool][sg-mxfp8-pool].

**FP4: native cache consumption and reconstructed-buffer access are different contracts.** SGLang's quantization-method registry specifies which contract each recipe/backend/phase combination implements:

| Cache recipe and route | What attention receives | Conversion boundary |
| --- | --- | --- |
| NVFP4, native `trtllm_mha` prefill/decode | Packed FP4 K/V plus block/global scales | The external kernel consumes compressed storage; this alone does **not** establish FP4 MMA |
| NVFP4, `flashinfer` wrapper prefill | FP8 E4M3 K/V workspace | Cached prefix values are reconstructed and converted to FP8 before attention; persistent storage remains FP4 |
| `fp4_mx_block16`, registered plain-access backends | BF16 K/V tensors | Pool getters unpack and reconstruct the supplied layer buffers before attention |

The last route is a compatibility mechanism: its registered backends include `triton`, `torch_native`, `flex_attention`, and `trtllm_mha`, plus `fa4` for prefill. In the generic getters, conversion covers the layer's allocated K/V arrays, not just the selected request's tokens. Consequently, compact persistent storage does not imply equally small temporary memory or attention traffic. NVFP4's FP8 workspace also has pool-sized capacity, but its prefill preparation populates the required prefix locations and reuses the workspace across layers. [Access registry and reconstruction][sg-fp4-method]; [pool getters/workspace][sg-pool]; [FlashInfer prefill][sg-fi].

### SGLang: recipe and backend selection

SGLang first resolves `--attention-backend` or separate prefill/decode backends using GPU generation, model architecture, cache dtype, and requested features. The FP4 access registry then validates the recipe/backend/phase combination; the pool allocates its required payload, scale, and workspace buffers. Each adapter subsequently dispatches by phase, shape, dtype, and library capability. This policy belongs to SGLang; the library makes the final kernel selection. [Resolution hooks][sg-attn-hook]; [backend registry][sg-registry]; [cache access contracts][sg-fp4-method].

**The default is selected by fixed rules, not by benchmarking FlashAttention against FlashInfer for each request.** At this revision, the base choice for CUDA MHA/GQA simplifies as follows **with speculative decoding disabled and no model-specific override**:

```text
Explicit attention backend?
  ├─ Yes → retain it, subject to compatibility rules
  └─ No
      ├─ Hopper with CUDA ≥12.3 → fa3
      ├─ SM100
      │   ├─ unequal K/V head dimensions → fa4
      │   └─ otherwise → trtllm_mha
      └─ Other CUDA cases
          ├─ FlashInfer available, no attention sinks → flashinfer
          └─ otherwise → triton
```

The executable selector cites a historical FlashInfer 0.6.1 Hopper performance regression as its reason for preferring FA3 there. This is a kernel/version consideration, not a consequence of token versus page indexing. The tree describes the base default, not the complete support matrix: speculative decoding, model overrides, and cache recipes can change or restrict the result. For example, a subsequent rule changes `fa3` with E5M2 KV to `triton`. [Default selector][sg-default-attn]; [resolution and compatibility rules][sg-attn-overrides].

Users can set `--prefill-attention-backend` and `--decode-attention-backend` separately. For a deployment, first establish feature and format compatibility, then compare measured performance at the intended batch sizes and context lengths; neither addressing scheme guarantees the better kernel. NVFP4 illustrates how a cache recipe further constrains those choices:

```text
NVFP4 cache + supported model/features
  ├─ Prefill input = auto, with no explicit backend
  │    ├─ SM100 → native FP4 → trtllm_mha
  │    └─ SM120 → FP8 workspace → flashinfer
  ├─ Explicit prefill input = native FP4
  │    └─ requires SM100 → trtllm_mha
  ├─ Explicit prefill input = FP8 E4M3
  │    └─ flashinfer, with reconstructed FP8 workspace
  └─ Decode → trtllm_mha, native packed FP4 cache
       ├─ SM100 → TRT-LLM generation path; FP8 Q/output at the call
       └─ SM120 → XQA path; FP16/BF16 Q/output at the call

Incompatible backend/recipe choices are rejected, not silently repacked.
```

With an explicit compatible backend, `auto` instead follows that backend’s input contract. The prefill-input option is `--prefill-kv-cache-dequant-dtype`. Here SM100/SM120 denote compute capabilities 10.0/12.0; they are separate specializations, not interchangeable “Blackwell” paths. Native SM100 scale packing additionally requires page size divisible by four and head dimension divisible by 64. These are representation constraints alongside attention's model and shape checks. [Recipe resolution][sg-kv-hook]; [native scale construction][sg-fp4-method]; [TRT-LLM calls][sg-trtllm].

Conventional FP8 has different checks. For example, The `flashinfer` wrapper decode policy honors `SGLANG_FLASHINFER_USE_TENSOR_CORE`; otherwise its modern-library branch requests tensor cores for FP8, or for FP16/BF16 when the Q-head/KV-head ratio is at least four. That flag chooses a kernel family, not necessarily FP8 operands. Likewise, acceptance by SGLang's backend registry does not prove every cache dtype is supported by that backend's external kernels. [Decode policy][sg-fi]; [registration checks][sg-registry].

### SGLang: allocation and prefix reuse

With `P=1`, the allocator assigns individual token locations. With `P>1`, it assigns whole pages and extends a request into its partial page before allocating another. The request table still records **one location per token**. [Token allocator][sg-token-alloc]; [paged allocator][sg-paged-alloc].

`RadixCache` finds the longest reusable token prefix and returns its existing cache locations; new suffixes receive new locations. It protects referenced tree nodes from eviction and releases unprotected segments back to the allocator when needed. Insertion stores indices, without copying their K/V payload. With `P>1`, reusable prefixes are rounded down to complete pages. In the earlier example, A and B share slots `8..11`, while their suffixes remain separate. This prefix-management mechanism does not prescribe an attention kernel or quantization algorithm. [Prefix matching and lifecycle][sg-radix].

Like vLLM, SGLang normally allocates capacity ahead of serving and dynamically assigns portions of it. The distinctive organization here is a persistent request-to-token table, radix-tree prefix lookup, and pool/adapter-specific data layouts; vLLM's discussed path centers on request block tables, hashed prefix blocks, and its layout resolver. Neither requires a separate GPU allocation for each request page.

**Evidence boundary:** this is source analysis, with no GPU benchmark. FlashInfer is pinned to `0.6.18`. The inspected FA4 wrapper defaults to SGLang's vendored kernel at the pinned commit; `SGLANG_INKLING_FA4_USE_PIP=1` selects the external package instead. The manifest's FA3 package constraint does not identify an immutable kernel source revision. Specialized model pools and unlisted backend/dtype combinations are outside this analysis. [FA4 wrapper][sg-fa4-wrapper]; [dependency manifest][sg-dependencies].

## Hugging Face Transformers: compressed storage with dense attention inputs

This case follows Transformers `96331a9f93b7` (2026-09-26), with Quanto `5b94afb31bcd` and HQQ `d88a488ec8aa` inspected separately for their encodings. The scope is **`QuantizedCache` in ordinary decoder-only generation**, using Llama as the model example. Transformers also has static, offloaded, and paged/continuous-batching paths; their storage contracts must not be inferred from this cache class.

The defining boundary is `cache.update(...)`: **the cache owns compression but returns reconstructed floating-point K/V tensors to attention**. Quanto/HQQ are quantization backends here; eager/SDPA/FlashAttention are attention implementations selected separately. [Cache implementation][hf-cache]; [Llama call sequence][hf-llama].

### Transformers: cache formats and grouping

Both integrations use group-wise **affine integer quantization**, with separate parameters for K and V. They do not expose FP8, NVFP4, Hadamard rotation, or TurboQuant through this `QuantizedCache` selector.

| Quantization backend | Bit widths | Reconstruction and parameter selection |
| --- | --- | --- |
| Quanto | 2, 4 | Unsigned packed codes; $\hat x=sq-b$. `MaxOptimizer` derives scale and floating-point shift from each group's minimum and maximum |
| HQQ | Adapter accepts 1, 2, 3, 4, 8; user guide lists 2, 4, 8 | Unsigned codes; $\hat x=s(q-z)$. Min/max initialization followed by HQQ's default proximal optimization; the zero parameter need not be an integer |

Defaults are four bits, group size 64, residual length 128, and `axis_key=axis_value=0`; generation selects Quanto unless configured otherwise. The variable named `zeropoint` in the Quanto adapter actually receives a floating-point **shift** in this route. Neither representation requires zero to be exactly representable. [Adapters and defaults][hf-cache]; [documented bit widths][hf-cache-guide]; [Quanto parameter calculation][quanto-max] and [reconstruction][quanto-qbits]; [HQQ quantizer][hqq-quant] and [optimizer][hqq-optimize].

**An axis number is backend-specific, not a universal token/head label.** For contiguous input `X[B,H,S,D]` and group size `G`:

| Setting | Actual grouping |
| --- | --- |
| Quanto `axis=0` | Reshape to `[-1,G]`; one parameter pair per row |
| Quanto `axis=-1` | Preserve the final component axis and group flattened leading positions; this can group the same component across tokens |
| HQQ `axis=1` | Reshape to `[-1,G]`; reduce along each row |
| HQQ `axis=0` | Reshape to `[G,-1]`; reduce along each column, grouping strided positions in the flattened input |

Thus Quanto axis 0 and HQQ axis 1 both produce two 64-component groups per token/head when `D=128, G=64`. Other shapes or axes can cross token/head boundaries. Grouping and packing impose divisibility constraints. The HQQ allowlist is not an end-to-end guarantee: its pinned 3-bit, axis-1 dequantizer uses a two-dimensional shape assumption when trimming padding, which is incompatible with general four-dimensional KV inputs. [Quanto grouping][quanto-group]; [HQQ grouping and dequantization][hqq-quant].

### Transformers: storage layout

`QuantizedCache.layers[l]` owns four separate objects: compressed K, compressed V, and their full-precision residual tails. There is **no page table or shared slot allocator in this path**. Sequence growth uses concatenation and replacement of tensors.

```text
Layer l — separately owned storage:

  _quantized_keys   → packed K codes + K scales/offsets + shape metadata
  keys             → recent full-precision K tail [B,H,R,D]

  _quantized_values → packed V codes + V scales/offsets + shape metadata
  values           → recent full-precision V tail [B,H,R,D]

Logical attention inputs, reconstructed during update:
  K_full[B,H,S,D] = concat(dequantize(K_prefix), K_tail, K_new, dim=2)
  V_full[B,H,S,D] = concat(dequantize(V_prefix), V_tail, V_new, dim=2)
```

K/V, payload/metadata, and prefix/tail are not one interleaved byte record. Quanto stores a `WeightQBitsTensor` containing packed data, scale, shift, and logical shape/stride information. HQQ stores `(packed_tensor, metadata_dict)`, including scale, zero, original shape, grouping, and packing mode. A quantized wrapper's logical shape is therefore not its physical byte layout. [Cache objects][hf-cache]; [Quanto tensor][quanto-weight-qbits]; [HQQ metadata][hqq-quant].

For a concrete four-bit example, let `B=1, H=2, S=4, D=128, G=64`, using Quanto axis 0 or HQQ axis 1:

```text
One K prefix (V is stored independently):
  logical values:       [1,2,4,128] = 1024 components
  grouped codes q:      [16,64]     = 16 groups
  physical code bytes:  [8,64]      = 512 bytes
  scales and offsets:  [16,1] each, in separate tensors

For i=0..7 and c=0..63:
  Quanto byte[i,c] = q[i,c]       | (q[i+8,c] << 4)
  HQQ    byte[i,c] = (q[i,c] << 4) | q[i+8,c]
```

These packers combine corresponding positions from different **group rows**, not necessarily neighboring head components. Two-bit packing combines four rows per byte; HQQ additionally provides one-bit and padded three-bit packing, the latter placing ten codes in an INT32 word. This is a library-specific compressed tensor format, not llama.cpp's inline block structures or a serving-engine page layout. [Quanto packing][quanto-pack]; [HQQ packing][hqq-pack].

### Transformers: cache update and attention workflow

The residual is a tail accumulated between flushes, not a permanently preserved window of the latest 128 tokens. The inspected single-token decode workflow is:

```text
First update / prefill:
  quantize incoming K/V → store compressed prefix
  return original incoming K/V → attention
  residual tail remains empty

Subsequent update:
  dequantize the entire compressed prefix
    → concatenate prefix + residual tail + new K/V
    → return dense floating-point K/V to attention

  Meanwhile, update persistent storage:
    tail reaches threshold? (default residual_length = 128)
      yes → requantize the combined history; clear the tail
      no  → append new K/V to the full-precision tail
```

At a flush, the implementation requantizes **the reconstructed old prefix together with the new tail**, not just the tail independently. Every subsequent update still reconstructs the prefix even when no flush occurs. The exact threshold test is `tail.ndim == 4 and tail.shape[-2] + 1 >= residual_length`, so this description is scoped to single-token decoding; it is not a general chunk-length formula. [Update implementation][hf-cache-update].

In Llama, RoPE is applied before cache insertion; `cache.update` then returns K/V before the attention interface is called. Q is not quantized merely because this cache is enabled:

| Attention implementation | Inputs and computation after the cache update |
| --- | --- |
| Eager | Dense Q/K/V; floating-point score matmul, FP32 softmax, and value matmul |
| SDPA | Dense Q/K/V passed to PyTorch `scaled_dot_product_attention`; PyTorch selects its applicable kernel |
| FlashAttention | Dense Q/K/V transposed from `[B,H,S,D]` to `[B,S,H,D]` before the library call; packed Quanto/HQQ codes and their scales are not passed to it |

Consequently, this integration performs **dequantization before attention**, rather than an integer attention dot product or reconstruction fused into the attention kernel. Selecting FlashAttention does not turn Quanto INT4 storage into INT4 attention. [Llama ordering and eager arithmetic][hf-llama]; [SDPA adapter][hf-sdpa]; [FlashAttention adapter][hf-flash].

### Transformers: selection and implications

The two selection mechanisms are independent:

```text
cache_implementation="quantized"
  → QuantizedCache(backend="quanto" or "hqq", bits, axes, group size, residual)
  → layer cache update returns dense K/V

attn_implementation
  → unspecified: try SDPA; fall back to eager if its model-level checks fail
  → explicit implementation: validate model/library/device/dtype support
  → attention library chooses its kernel from the dense inputs
```

The quantized constructor accepts only full-attention layer types at this revision; the generation resolver rejects encoder-decoder models for this route. These restrictions are distinct from whether an attention library can process a particular dtype. [Generation resolver][hf-generation]; [cache constructor][hf-cache]; [attention selection][hf-attn-select].

**The design favors a reusable dense-tensor attention interface.** New compression methods can reconstruct to that interface without changing each attention kernel, but the attention call does not retain low-bit KV bandwidth. Persistent cache memory can shrink while temporary dense prefixes, concatenated inputs, and periodic requantization still consume memory and time. This differs from the inspected llama.cpp/vLLM/SGLang routes that consume compressed storage inside attention. It is a source-level trade-off, not a measured speed ranking. No model/GPU run was performed, and the external-library pins identify inspected encodings rather than an automatically enforced dependency combination.

## MNN: backend-specific quantization for on-device attention

This section follows MNN `67450001627f` (2026-09-23), focusing on ordinary fused attention with an in-memory cache: the CPU implementation and Metal's separate INT8 path. MNN integrates cache allocation, packing, and attention kernels within each backend. Consequently, the same configuration value does **not** imply the same encoding or arithmetic across devices. [CPU attention][mnn-cpu-attn]; [Metal attention][mnn-metal-attn].

### MNN: cache formats and algorithms

The public `attention_mode` replaces the deprecated `quant_qkv`: `mode = 8 × flash_attention + kv_quant_mode`. The default is `8`, meaning unquantized KV with MNN's own FlashAttention-style execution; it does not select the external FlashAttention library. The CPU modes are:

| `kv_quant_mode` | Persistent K | Persistent V |
| --- | --- | --- |
| 0 | Floating-point | Floating-point |
| 1 / 2 | INT8 | Floating-point / INT8 |
| 3 / 4 | TQ3 | Floating-point / TQ3 |
| 5 / 6 | TQ4 | Floating-point / TQ4 |

Thus `10`, `12`, and `14` request flash-style attention with both K/V in INT8, TQ3, and TQ4, respectively. Floating-point storage follows the CPU backend's precision rather than always being FP16. [Configuration][mnn-docs]; [mode decoding][mnn-cpu-attn].

**CPU INT8 is affine and treats K and V differently.** In the inspected generic CPU routine, K subtracts a **signed channel-wise maximum**, not a mean or absolute maximum: for each head/component, `reference[d] = max_t K[t,d]` over the initial multi-token prefill. It then quantizes each shifted token/head vector with its own scale/bias pair (the default is one group per head). The code maintains this reference in `mKeyMax`: calls with `seqLen > 1` update the running maxima; ordinary single-token decoding reuses them. Thus “initial-prefill reference” describes the usual prefill-then-decode workflow, not an unconditional freeze across later multi-token calls. [Initialization][mnn-cpu-cache]; [quantization routine][mnn-cpu-quant].

For a fixed reference vector shared by all keys, subtracting it shifts every score for one query by the same constant: `q·(k_t − reference) = q·k_t − q·reference`. Ordinary softmax cancels that constant before considering quantization error. This explains why shifted keys can be used directly; it does not establish that choosing maxima always improves accuracy.

V instead uses a scale/bias pair per channel, estimated from the initial prefill and reused during decoding; subsequent values are clamped to the representable range. These V parameters are copied into physical cache chunks, so chunking does not imply independently recalibrated scales. This differs from symmetric per-token/head INT8 and introduces sensitivity to the initial value range. [Quantization routines][mnn-cpu-quant].

**CPU TQ3/TQ4 are nonuniform codebook formats.** Each 32-component block is RMS-normalized, transformed by a fixed sign pattern plus normalized Walsh–Hadamard transform, and mapped to an 8- or 16-entry Lloyd–Max codebook designed for a standard Gaussian. It stores an FP16 RMS scale followed by packed indices: **14 bytes for TQ3 (3.5 bits/value), 18 bytes for TQ4 (4.5 bits/value)**. These totals include scale overhead. MNN implements the MSE-oriented rotation/codebook scheme without the paper's additional QJL residual correction; its layout is not interchangeable with vLLM's TurboQuant cache. The inspected loops assume head dimensions divisible by 32. [Encoding and codebooks][mnn-tq].

### MNN: physical cache storage

For the ordinary CPU path, each attention layer owns separate K and V byte tensors, each shaped `[KV_heads, bytes_per_head]`. The manager tracks valid length and capacity; initial allocation and subsequent growth reserve **required length + 64 tokens**, copying/repacking existing data when necessary. These are growing sequence buffers, without the request-to-page tables used in the vLLM/SGLang paths above. The tensor's `int8_t` allocation type denotes raw storage and does not imply that every mode stores INT8 numbers. [Allocation and writes][mnn-cpu-cache].

```text
One CPU layer, TQ3 example: H=2, head dimension D=64, capacity C
K buffer                                     V buffer (separate)
  head 0: token 0 | token 1 | ... | token C-1   head 0: token 0 | ...
  head 1: token 0 | token 1 | ... | token C-1   head 1: token 0 | ...

One token/head record = 28 bytes:
  [FP16 RMS | 32 three-bit codes] [FP16 RMS | 32 three-bit codes]
  <--------- 14 bytes ---------> <--------- 14 bytes --------->

K byte offset(h,t,component block b) = (h*C + t)*28 + b*14
V uses the same formula relative to its own buffer base.
Slots beyond the valid length are reserved capacity, not attended tokens.
```

CPU INT8 uses a different physical order optimized for packed GEMM: K interleaves token lanes within packing tiles; V packs channels and sequence positions within flash-attention chunks. FP32 scale/bias arrays follow the corresponding packed payload tiles, with auxiliary sum buffers for affine corrections. Therefore, the simple TQ token-record formula must not be applied to INT8. The packing factors depend on the CPU kernel; these tiles are neither quantization groups nor request-allocation pages. [INT8 packing][mnn-cpu-quant]; [cache sizes][mnn-cpu-cache].

A **token lane** is a token's position within a K packing tile: `tile = t // hP8`, `lane = t % hP8`. The following example uses illustrative packing factors, one KV head, and one quantization group per head:

```text
Head dimension D=4; hP8=2 token lanes; lP8=2 components per packed group

Logical INT8 K codes:
  token 0: [a0 a1 a2 a3]     -> tile 0, lane 0
  token 1: [b0 b1 b2 b3]     -> tile 0, lane 1
  token 2: [...]             -> tile 1, lane 0

Tile 0, increasing byte addresses:
  [a0 a1 | b0 b1 | a2 a3 | b2 b3] [scale0 scale1] [bias0 bias1]
   lane 0  lane 1  lane 0  lane 1   <---------- FP32 ---------->
  <--------- 8 INT8 bytes -------> <--------- 16 bytes -------->

Within the INT8 payload:
  byte_offset(lane,d) = (d // lP8)*hP8*lP8 + lane*lP8 + d % lP8
```

Each token keeps the same lane across component groups, but its full vector is not contiguous. This interleaving matches the GEMM kernel's input packing; **each token/head still has its own scale and bias**. A token lane denotes a storage position, not a GPU thread or a shared quantization group. [K write indices and metadata][mnn-cpu-quant].

### MNN: CPU attention paths and selection

| Cache choice | QK score calculation | Value accumulation |
| --- | --- | --- |
| Unquantized | Floating-point packed matmul | Floating-point packed matmul |
| INT8 K, floating-point V | Dynamically quantize Q to INT8; integer GEMM with scale/bias corrections | Floating-point |
| INT8 K/V | Same INT8 QK route | Quantize blockwise softmax weights to INT8; integer GEMM with cached INT8 V and affine corrections |
| TQ3/TQ4 | Rotate floating-point Q; unpack K indices, look up floating-point centroids, and dot with rotated Q using block scales | If V is quantized, accumulate scaled centroids in rotated space, then inverse-transform each accumulated component block |

TQ's fused lookup arithmetic avoids reconstructing a full historical K/V tensor. It nevertheless performs **floating-point arithmetic on reconstructed codebook values**, not three-/four-bit integer matrix multiplication; Q remains floating-point. For V, the inverse transform follows weighted accumulation within each attention chunk, rather than being repeated for every cached token. [CPU execution][mnn-cpu-attn]; [dot/accumulation helpers][mnn-tq].

Selection is primarily explicit configuration followed by backend constraints. CPU disables INT8 V when flash-style execution is off; the LLM runtime also changes `attention_mode=10` to `9` when `reuse_kv=true`, retaining INT8 K but floating-point V. The ordinary quantized-cache branch requires fewer than five Attention inputs; the extra-input route bypasses it. CPU capabilities then choose packed GEMM/vector implementations, while sequence length, head grouping, and thread count influence tiling. Cache format, arithmetic, and scheduling are therefore separate decisions. [Runtime override][mnn-runtime]; [CPU dispatch][mnn-cpu-attn].

### MNN: Metal is a separate implementation

**Metal API support and MNN support are separate.** Metal is Apple's GPU compute API. Apple documents TensorOps INT4/INT8 support in updates to iOS/macOS 26, and FP8 (E4M3/E5M2), FP4 (E2M1), and block-scaling additions in iOS/macOS 27. TensorOps can consume quantized tensors and handle dequantization using available GPU acceleration; this API contract does not establish native multiplication at the stored bit width on every Apple GPU. Availability depends on the OS/API and device. [Apple TensorOps overview][metal-tensorops]; [format/capability tables][metal-capabilities].

**Evidence for low-bit computation on M5.** External kernel implementations clarify what datatype support alone cannot (sources checked 2026-09-27):

| Format | Established execution path |
| --- | --- |
| INT8 | Modular feeds INT8 operands directly to an integer widening matrix operation, accumulates in INT32, then applies scales. This confirms integer matrix arithmetic. [Kernel][metal-m5-int8] |
| FP8 | Modular's M5 MMA interface accepts E4M3/E5M2 operands with FP32 accumulation. This establishes direct FP8 operands at the matrix-intrinsic interface, without resolving the multiplier's internal circuitry. [MMA implementation][metal-m5-mma] |
| INT4 / FP4 | Inspected implementations unpack INT4 for INT8 computation or reconstruct FP4 into BF16. These paths do not establish native four-bit multiplication. [INT4 implementation notes][metal-m5-quant-kernels]; [FP4 kernel][metal-m5-fp4] |

These findings are generation-specific: an M4 Max study reports emulated FP8 TensorOps execution, rather than a native FP8 throughput advantage. [Measurements][metal-m4-rigel]. **MNN's floating-point INT8-cache attention therefore does not imply that M5 lacks integer matrix hardware**; using that hardware would require a compatible attention implementation, including Q quantization and affine corrections.

At the pinned revision, **MNN's Metal KV-cache path implements floating-point and INT8 storage**, without FP8, INT4, FP4, or CPU TQ3/TQ4 cache kernels. These are implementation limits of this MNN path, not general Metal format limits. [Cache allocation][mnn-metal-cache]; [attention implementation][mnn-metal-attn].

MNN supports INT8 K or K/V on Metal. Use its documented low-bit mode values `0/1/2`; the current code tests inequalities, so passing a CPU TQ mode can activate INT8 instead of producing TQ storage.

**Dynamic INT8 algorithm.** On cache insertion, treat one token's K as a vector of `KV_heads × head_dim` components; process V independently in the same way. For either vector `x`, the shader computes:

```text
lo = min(x)                           # across all heads/components of this token
hi = max(x)
scale = max((hi - lo) / 255, 1e-6)
bias  = lo + 128 * scale
codes = clamp(rint((x - bias) / scale), -128, 127)

store: signed INT8 codes + floating-point (scale, bias)
read:  reconstructed_x = scale * codes + bias
```

This is **asymmetric per-token quantization across all KV heads**, with one independent parameter pair for K and one for V at each layer. `bias` is a floating-point reconstruction offset, not a stored integer zero point. Each new token supplies its own min/max, so decoding requires neither initial-prefill calibration nor rewriting older tokens. Both K and V use this recipe; there is no CPU-style channel-maximum subtraction for K or fixed per-channel calibration for V. [Insertion shader][mnn-metal-shader].

**Static-scale alternative.** The conventional shader also has a model-supplied scalar-scale route: `codes = clamp(rint(x / scale), -128, 127)` and `reconstructed_x = scale * codes`, with separate K/V scales and zero offset. The adapter checks for nonzero model-provided scales when choosing static versus dynamic shader variants; the dynamic recipe above describes the ordinary configuration-driven case without those scales. [Variant selection][mnn-metal-attn]; [encoding and reconstruction][mnn-metal-shader].

**Storage.** For the ordinary single-sequence dynamic INT8 path:

```text
K codes: [capacity, KV_heads, head_dim]   separate contiguous region
V codes: [capacity, KV_heads, head_dim]   separate contiguous region
K parameters: [capacity, 2]              scale, bias per token
V parameters: [capacity, 2]              scale, bias per token
```

K/V payloads and their parameter arrays occupy separate regions. Parameters use the backend's floating-point storage type (FP16 or FP32). Sharing one parameter pair across all heads reduces metadata relative to per-token/head quantization, but a large range in one head can coarsen the quantization of other heads in the same token. [Metal allocation][mnn-metal-cache]; [quantization shader][mnn-metal-shader].

MNN's Metal attention reconstructs INT8 values for floating-point computation; it does not inherit CPU's INT8-Q GEMM path. Its fused prefill path reconstructs K/V tiles into FP16 before simdgroup matrix multiplication. That path requires compatible matrix hardware, FP16 precision, causal layout, in-memory KV, head dimension 64/128/256, GQA ratio 1/2/4/8, and at least 128 query tokens. Configuration/environment controls and an M4-class performance heuristic further affect selection; otherwise, separate QK, softmax, and PV kernels are used. These are MNN's local kernel decisions. TQ availability or Metal's INT8 behavior should not be generalized to other GPU backends. [Selection][mnn-metal-attn]; [fused shader][mnn-metal-flash].

**Implication.** MNN demonstrates that one engine can combine CPU integer GEMM, rotation/codebook vector arithmetic, and GPU floating-point reconstruction. Its backend-specific packing enables these routes, but each new format needs matching storage writers and attention readers. TQ's compact records reduce persistent storage; their effect on latency and accuracy still requires device- and workload-specific measurements.

## ExecuTorch: cache quantization selected during export

ExecuTorch exports a model and lowers graph regions to device backends. Consequently, its cache design belongs to a **model/export-path/backend combination**, rather than one framework-wide cache ABI. This section follows revision `12668f8b81c39` (2026-09-27): the Llama export transformations, CUDA TQ4, an MLX TQ4 implementation, and Llama deployment through ExecuTorch's Qualcomm/QNN backend. These are distinct implementations, not interchangeable runtime options. [Export pipeline][et-export]; [configuration][et-config].

**The Qualcomm/QNN path** refers to deploying a Llama model on Qualcomm hardware: `Llama model → ExecuTorch export/quantization → Qualcomm AI Engine Direct (QNN) → target accelerator`. Llama identifies the model; QNN supplies the device execution backend. ExecuTorch's example runner manages the model's inputs, outputs, and KV-cache buffers. The cache details below describe this deployment example, not a separate model family or a universal Qualcomm cache format. [Deployment guide][et-qnn-doc]; [runner][et-qnn-cache].

### ExecuTorch: cache formats and quantization units

Let `B` denote batch size, `H` KV heads, `C` cache capacity, and `D` components per head. Quantization settings for weights do not establish KV-cache support.

| Inspected path | Cache encoding | Reconstruction parameters |
| --- | --- | --- |
| Llama `quantize_kv_cache` | Dynamic **asymmetric INT8**, independently for each new K/V token/head vector | One FP32 scale and INT8 zero point per `D` values: `x̂ = s(q − z)` |
| Llama `static_quantize_kv_cache` | **Symmetric INT8**, with fixed parameters during generation | By default, separate K/V scales per component `d`, shared across batch, tokens, and heads; no nonzero zero point |
| Shared TurboQuant cache, with CUDA/MLX consumers | **TQ4:** two nonuniform codebook indices per byte | One BF16 L2 norm per token/head vector, plus a shared rotation matrix and 16-entry codebook |
| Llama through the Qualcomm/QNN backend | Calibrated **8-bit cache I/O**, normally represented as `uint8` | Graph quantization encodings determined by the selected recipe; not the dynamic token/head metadata above |

The dynamic INT8 replacement explicitly selects asymmetric quantization, despite the cache class also declaring symmetric and groupwise enum names. The static path defaults to component-wise scales: calibration takes the maximum absolute value over `(B, tokens, H)` and divides by 127. Without calibration tasks, each scale uses the configured constant, initially `1/127`; the class also supports a single scale per K/V tensor. Thus “static” describes **fixed quantization parameters**, not a distinct allocation policy. [INT8 implementations][et-cache]; [calibration/export][et-exportlib]; [Qualcomm annotations][et-qnn].

For TQ4, a vector is L2-normalized, multiplied by a seeded random orthogonal matrix generated through QR, then assigned to Lloyd–Max centroids for an approximate Gaussian coordinate distribution. This implementation uses a dense rotation, not a Hadamard transform. A head vector occupies `D/2 + 2` bytes for codes and norm—`4 + 16/D` bits/value, excluding shared constants. [Compression/storage][et-tq-cache]; [codebook and rotation][et-tq-book].

### ExecuTorch: physical storage and token addressing

The inspected cache modules reserve **fixed-capacity, separate tensors per layer**, updated using token positions. They do not use request block tables or a shared paged allocator. K, V, and their metadata are distinct tensors; their relative addresses in a runtime memory arena are not part of the layout contract. The ordinary floating-point cache supports `[B,H,C,D]` or `[B,C,H,D]`; the INT8 transformations specifically store `[B,C,H,D]`. Returning a transposed attention view does not change those underlying bytes. [Base cache][et-base-cache]; [INT8 cache][et-cache].

```text
One layer, B=1; each box is a separately addressed tensor.

Dynamic INT8: token-major storage
K codes [1,C,H,D]: [t0:h0 D codes | t0:h1 D codes | ...]
                 [t1:h0 D codes | t1:h1 D codes | ...] ...
V codes [1,C,H,D]: same order, separate storage
K scales / K zero points [1,C,H,1]: one pair per token/head
V scales / V zero points [1,C,H,1]: separate metadata

K_code[b,t,h,d] byte offset = ((b*C + t)*H + h)*D + d
Static INT8: same code layout; scales instead have shape [1,1,1,D].

TQ4: head-major storage; shared cache module supports B=1
K packed [1,H,C,D/2]: [h0:t0 packed vector | h0:t1 packed vector | ...]
                     [h1:t0 packed vector | h1:t1 packed vector | ...]
K norms  [1,H,C,1]:   [h0:t0 norm | h0:t1 norm | ...] ...
V packed / V norms:   two additional tensors with the same ordering

K_packed[0,h,t,d//2] byte offset = (h*C + t)*(D/2) + d//2
even d: high nibble; odd d: low nibble
```

Capacity `C` bounds the allocated history; a valid-position mask or length bounds what attention may read. An unused position is reserved space, not an allocated page belonging to another request. Cache writes replace the addressed positions without requantizing the older history. The dynamic INT8 payload plus metadata costs `D + 5` bytes per K or V token/head vector. [INT8 writes][et-cache]; [TQ4 writes][et-tq-cache].

**The Qualcomm/QNN Llama example uses a different cache interface.** Its runner manages explicit cache inputs and outputs, with K ordered `[B,H,D,C]` and V `[B,H,C,D]`; newly produced chunks use the corresponding order. It allocates separate K/V input and output buffers per layer and copies/rearranges data when advancing or changing execution shapes. Quantization encodings belong to the delegated graph, rather than a scale record appended to every token. The public graph annotations describe quantized attention, but do not expose the proprietary QNN kernel's physical multiplication instructions. [Runner buffers and updates][et-qnn-cache]; [8-bit cache annotations][et-qnn]; [official deployment guide][et-qnn-doc].

### ExecuTorch: attention paths and selection

**Llama dynamic INT8 has two consumers.** The exporter installs `QuantizedKVCache`, then attempts to replace attention with `QuantizedSDPA`. This replacement requires an eligible `Attention` module whose existing SDPA is `SDPACustom`, normally installed by `use_sdpa_with_kv_cache`. On that route, it sets `return_float_values=False`, quantizes Q asymmetrically per token/head, and passes INT8 Q/K/V plus their metadata to `llama.custom_quantized_sdpa`. Without that matching SDPA replacement, the cache returns reconstructed FP32 K/V and substitutes the original, unquantized values at the newly written positions for the current attention call. Cache quantization alone therefore does **not** guarantee INT8 Q. [Transformation guards and query preparation][et-sdpa]; [export ordering][et-exportlib]; [float-return path][et-cache].

Within the custom CPU operator, let `m` be the number of query rows in the current attention block—not request batch size:

```text
Quantized custom CPU SDPA
├─ QK
│  ├─ Apple build + BLAS available + query block m > 4
│  │    → reconstruct Q/K tiles to FP32, then floating-point GEMM
│  └─ Otherwise → TorchAO INT8 × INT8 routine with scale/zero-point correction
├─ Softmax → FP32
└─ PV
   ├─ m > 4 → reconstruct V tile to FP32, then floating-point GEMM
   └─ m ≤ 4 → specialized FP32-probability × INT8-V routine
```

The small-`m` PV interface retains floating-point probabilities; its “quantized” name does not mean both operands are INT8. The CPU implementation chooses these routes inside attention, while the exporter selects the operator and persistent cache format. [CPU arithmetic and dispatch][et-cpu-sdpa].

**Static INT8 reconstructs before attention.** The inspected transformation returns dense K/V in the incoming tensor dtype and restores the current unquantized chunk, leaving INT8 persistent state. It requires static export shapes and rejects `use_sdpa_with_kv_cache`, attention sinks, and local/global cache transformations. A delegate may further lower/fuse the exported graph; these Python operations alone do not establish its final machine arithmetic. [Static cache][et-cache]; [export restrictions][et-exportlib].

**CUDA TQ4 reconstructs inside attention tiles.** The caller supplies BF16 Q; the implementation rotates Q, looks up BF16 centroid values from packed K/V codes, and performs BF16 dot products with FP32 accumulation/softmax. K norms multiply scores, while V norms multiply softmax weights before value accumulation; the output is rotated back. It avoids materializing the full reconstructed cache and does not perform INT4 matrix multiplication. The kernel requires power-of-two `D` and an integral query/KV head ratio. [Triton kernel][et-tq-cuda].

The Qwen3.5-MoE exporter enables this route through `--turboquant` for full-attention layers. Kernel-level split-K decode additionally requires one query token, an explicit `kv_len`, and allocated capacity at least 256; the inspected Qwen caller omits `kv_len`, so that optimization is not activated merely by enabling TQ4. [Exporter][et-tq-export]; [model call][et-tq-model]; [dispatch][et-tq-cuda].

**The inspected MLX TQ4 implementation instead reconstructs temporary K/V tensors.** It uses the same packed-code/norm representation, rotates Q, reconstructs the live prefix in rotated space, calls `mlx.custom_sdpa`, and rotates the output back. Its class requires batch size one and `D` divisible by 64. This implementation should not be confused with universal exporter support: the Qwen3.5-MoE script above explicitly rejects its `--turboquant` option for MLX and Metal. [MLX cache/attention implementation][et-tq-mlx]; [exporter restrictions][et-tq-export].

**Implication.** ExecuTorch makes cache state and attention graph transformations extensible during export. Matching a new compressed representation to a consumer remains essential: the CPU path can exploit INT8 operands, CUDA TQ4 reconstructs tiles, and the inspected MLX/static-INT8 paths create dense temporaries. Fixed capacity simplifies position updates and memory planning, but reserves space for the configured context and does not provide the multi-request page sharing studied in vLLM/SGLang. Storage savings, temporary-memory costs, and kernel efficiency must therefore be assessed separately.

## vLLM cache-option reference

At revision `17e50b9b7610`, the cache configuration exposes the following families. **A recognized option does not imply support on every model, GPU, or attention backend.** [Configuration options][vllm-cache-config]; [quantization modes][vllm-kv-modes].

| Technique | `--kv-cache-dtype` | Representation |
| --- | --- | --- |
| Scaled FP8 | `fp8`, `fp8_e4m3`, `fp8_e5m2` | Eight-bit floating-point K/V with separate scales; `fp8` aliases E4M3 |
| Dynamic FP8 per token/head | `fp8_per_token_head` | FP8 codes with scales computed for each incoming token and KV head |
| Dynamic INT8 per token/head | `int8_per_token_head` | Integer codes with the same scale granularity |
| INT4 per token/head | `int4_per_token_head` | Packed four-bit integers, Hadamard rotation, and asymmetric quantization |
| NVFP4 | `nvfp4`, `nvfp4_4over6` | Packed four-bit floating-point values with FP8 block scales |
| TurboQuant | `turboquant_*` | Presets with different key/value precision, detailed in the TurboQuant profile |
| Specialized formats | `fp8_inc`, `fp8_ds_mla`, `nvfp4_ds_mla` | Platform-specific FP8 or specialized DeepSeek MLA cache representations |

This configuration inventory is broader than any single backend. The profiles above establish the inspected ordinary-attention routes; specialized MLA/platform formats need their own adapter and kernel analysis.

## Small experiments to connect code and behavior

### Reproducible setup

**TODO:** Record model revision, quantization recipe and calibration data where applicable, hardware, software versions, build options, launch commands, and the observed kernel/backend. Keep commands alongside the logs they produced.

### Workloads and measurements

- Compare against a specified higher-precision baseline for the same model.
- Separate short and long prompts, prefill and decode, and low and higher concurrency.
- Measure peak memory, time to first token, inter-token latency, and throughput; define the units and aggregation used.
- Evaluate output quality with a specified task or dataset.
- Where supported, vary weight/activation precision and KV cache precision independently.

| Experiment | Hypothesis | Controlled settings | Evidence / result |
| --- | --- | --- | --- |
| Weight quantization | TODO | TODO | Not run |
| Activation quantization | TODO | TODO | Not run |
| KV cache quantization | TODO | TODO | Not run |
| Kernel or backend change | TODO | TODO | Not run |

**Plot placeholders:** Memory versus context length; prefill/decode performance versus workload size; quality versus precision. Include warm-up, repeated runs, and variability in the measurement notes.

## Take-aways: six KV-cache designs

The comparison below concerns the **inspected revisions and ordinary-attention paths**, not every supported model or platform. Numerical format, physical layout, and attention arithmetic are separate design choices.

| Framework | Cache formats | Storage and addressing | Attention paths |
| --- | --- | --- | --- |
| **llama.cpp** | Legacy Q4/Q5/Q8 and IQ4_NL; parameters per 32 components; independently selected K/V types | Slot identities address separate per-layer K/V tensors with inline quantization blocks | Selected quantized dot products; floating-point reconstruction on other paths |
| **vLLM** | Conventional FP8; token/head FP8, INT8, rotated asymmetric INT4; NVFP4 and TurboQuant | Request block tables, preallocated pages, adapter-specific strides and metadata | FP8 operands, in-kernel reconstruction/scale correction, or format-specific conversion |
| **SGLang** | Conventional FP8, MXFP8, NVFP4, FP4 block-16 | Request-to-token tables; separate per-layer K/V allocations and scale buffers | FP8 operands; MXFP8 QK/BF16 value computation; native FP4 input or reconstructed buffers |
| **Transformers** | Quanto INT2/INT4; HQQ affine integer formats, subject to packing constraints | Separate compressed prefixes and floating-point tails per layer; no page allocator in this path | Reconstruct dense K/V before eager, SDPA, or FlashAttention |
| **MNN** | CPU affine INT8 and 32-component TQ3/TQ4; separate Metal INT8 recipes | Growing per-layer K/V buffers; CPU packed tiles or TQ records, Metal token-major codes | CPU integer GEMM or rotated codebook arithmetic; Metal floating-point reconstruction |
| **ExecuTorch** | Dynamic asymmetric INT8, static symmetric INT8, TQ4, and delegate-specific 8-bit encodings | Fixed-capacity per-layer tensors; token-major INT8, head-major TQ4, separate Qualcomm K/V layouts | Export-selected operators: CPU quantized routines or reconstruction, CUDA TQ4 tile reconstruction, MLX dense reconstruction |

Sources: [llama.cpp cache types][cache-types] and [attention dispatch][cuda-attention]; [vLLM formats][vllm-cache-config] and [layout resolution][vllm-layout-resolver]; [SGLang pools][sg-pool] and [access contracts][sg-fp4-method]; [Transformers cache updates][hf-cache-update]; [MNN CPU][mnn-cpu-attn] and [Metal][mnn-metal-attn]; [ExecuTorch INT8][et-cache] and [TQ4][et-tq-cuda]. An exposed option is not universal kernel support: llama.cpp's K-quants are not selectable cache types here, and its CUDA FlashAttention rejects IQ4_NL.

**Quantization granularity and allocation granularity solve different problems.** Component blocks or token/head scales control numerical approximation and metadata cost; slots and pages control ownership, reuse, and traversal. Finer scales can adapt to local magnitude variation, but require more metadata. Neither paged allocation nor separate K/V tensors inherently limits the available quantization families.

**vLLM offers broader implemented numerical diversity; extensibility depends on the complete execution path.** Adding a format to any of these engines requires compatible writers, metadata addressing, and attention consumers. SGLang's native/workspace/plain-access contracts support multiple execution routes; Transformers instead exposes compression through a dense-tensor interface, simplifying attention reuse at the cost of reconstruction. MNN shows that even an INT8 option can have different parameter granularity, packing, and arithmetic across CPU and GPU. ExecuTorch moves much of this specialization into model export and delegate lowering. More named formats therefore do not establish greater portability across backends or hardware.

**Compressed capacity and attention efficiency must be evaluated separately.** In-kernel reconstruction can retain bandwidth savings without low-bit matrix multiplication; temporary reconstructed buffers add memory and conversion traffic. Even one kernel may mix operand precisions, as SGLang's MXFP8 path demonstrates. The practical comparison is the supported **format–layout–kernel combination**, including metadata and workspace costs; performance and accuracy rankings require workload-specific measurements.

## References and source-reading log

Start with the official documentation:

- [vLLM quantization documentation](https://docs.vllm.ai/en/latest/features/quantization/)
- [SGLang quantization documentation](https://docs.sglang.io/docs/advanced_features/quantization)
- [Transformers cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [llama.cpp quantization tool documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

Additional sources for the llama.cpp section:

- Iulia Turc's [legacy-quant guide][legacy-guide] and [K-quant guide][k-guide] (conceptual explanations; exact layouts checked against code).
- Uygar Kurt, [*Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct*][quant-study], arXiv:2601.14277v1, 2026.
- llama.cpp [`ggml-common.h`][llama-blocks], [`ggml-quants.c`][legacy-dequant], and [`llama-quant.cpp`][llama-presets], pinned to `a894dae939d426954ce54bb604824f1ae918a0c5`.

| Finding / question | Stack and version | File / symbol / permalink | Evidence and notes |
| --- | --- | --- | --- |
| KV-cache scales and insertion | vLLM `17e50b9b7610` | [Scale loader][vllm-kv-scales]; [dynamic writer][vllm-kv-write] | Fixed checkpoint scales versus live token/head scales |
| Attention computation with compressed KV | Same | [Query conversion][vllm-query-quant]; [Triton accumulation][vllm-kv-compute] | Backend-dependent operands and scale application; no model run |
| vLLM cache allocation and byte layouts | Same | [Allocator][vllm-cache-allocate]; [scale views][vllm-cache-scale-views] | Paging blocks distinguished from quantization groups; byte-layout examples for head dimension 128 |
| Exact block layout and storage cost | llama.cpp `a894dae939d4` | [`block_q*` definitions][llama-blocks] | Metadata included; distinct from whole-model average |
| Reconstruction and packed parameters | Same | [Legacy dequantizers][legacy-dequant]; [Q4_K dequantizer][k-dequant] | Centered versus affine grids; hierarchical scales |
| Meaning of S/M/L | Same | [Preset mapping][llama-presets]; [tensor selection][llama-selection] | Conversion policies, not new block layouts |
| Online activation quantization | Same | [CPU traits][cpu-types] and [conversion][cpu-mulmat]; [CUDA matrix-vector path][cuda-mmvq] | Operation-local work buffers; backend-dependent formats |
| Online KV-cache quantization | Same | [Server options][server-cache]; [row insertion][cache-write]; [CPU attention][cpu-attention] | Persistent independently typed cache; live block parameters |

[legacy-guide]: https://github.com/iuliaturc/gguf-docs/blob/main/legacy-quants.md
[k-guide]: https://github.com/iuliaturc/gguf-docs/blob/main/k-quants.md
[k-design]: https://github.com/ggml-org/llama.cpp/pull/1684
[quant-study]: https://arxiv.org/abs/2601.14277v1
[llama-blocks]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-common.h
[llama-legacy-blocks]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-common.h#L199-L279
[llama-k-blocks]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-common.h#L298-L403
[legacy-dequant]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-quants.c#L459-L565
[k-dequant]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-quants.c#L1529-L1556
[k-unpack]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-quants.c#L880-L889
[llama-presets]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-quant.cpp#L861-L873
[llama-selection]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-quant.cpp#L533-L564
[cpu-types]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cpu/ggml-cpu.c#L240-L337
[cpu-mulmat]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cpu/ggml-cpu.c#L1255-L1367
[online-q8]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-quants.c#L276-L335
[cuda-mmvq]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/mmvq.cu#L1484-L1521
[server-cache]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/tools/server/README.md#L71-L72
[cache-types]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/common/arg.cpp#L304-L323
[cache-write]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L1318-L1406
[cpu-cache-write]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cpu/ops.cpp#L5221-L5290
[cuda-cache-write]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/set-rows.cu#L6-L72
[cpu-attention]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cpu/ops.cpp#L8688-L8805
[cache-rotation]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L315-L338
[cache-graph]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-graph.cpp#L2855-L2896
[cache-validation]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-context.cpp#L3727-L3765
[cuda-attention]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/fattn.cu#L423-L512
[cuda-fa-build]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/docs/build.md#L303-L304
[cuda-fa-query]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/fattn-vec.cuh#L150-L200
[cuda-fa-dot]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/fattn-common.cuh#L144-L211
[cuda-fa-value]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/fattn-vec.cuh#L329-L375
[cuda-fa-select]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/fattn.cu#L621-L703
[cuda-fa-convert]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/fattn-common.cuh#L1010-L1089
[cuda-fa-mma]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/fattn-mma-f16.cuh#L2078-L2079
[fa-auto]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-context.cpp#L510-L558
[fa-graph]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-graph.cpp#L2615-L2695
[cpu-fa-select]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cpu/ops.cpp#L9254-L9343
[cuda-fa-gates]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/fattn.cu#L512-L624
[cuda-fa-capabilities]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/common.cuh#L346-L369
[cuda-fa-sparse-shape]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1760-L1766
[cuda-fa-sparse-work]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/fattn.cu#L119-L141
[cuda-fa-buffers]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/fattn.cu#L706-L738
[vllm-revision]: https://github.com/vllm-project/vllm/tree/17e50b9b761023d5f2499f062507df1ff49092a4
[vllm-block-metadata]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/core/kv_cache_utils.py#L175-L239
[vllm-block-pool]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/core/block_pool.py
[vllm-paged-design]: https://docs.vllm.ai/en/latest/design/paged_attention/
[vllm-prefix-design]: https://docs.vllm.ai/en/latest/design/prefix_caching/
[vllm-block-table]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/worker/block_table.py#L58-L122
[vllm-slot-mapping]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/worker/block_table.py#L445-L477
[vllm-cache-tensor]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/kv_cache_interface.py#L1379-L1400
[vllm-cache-allocate]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/worker/utils.py#L389-L455
[vllm-cache-views]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/kv_cache_interface.py#L285-L408
[vllm-cache-layout]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/kv_cache_layout.py#L11-L28
[vllm-backend-layout]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backend.py#L365-L369
[vllm-fa-layout]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flash_attn.py#L1238-L1295
[vllm-fi-layout]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L542-L549
[vllm-fi-payload]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L2097-L2132
[vllm-turbo-layout]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/turboquant_attn.py#L159-L165
[vllm-turbo-soa-layout]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/turboquant_soa/triton_turboquant_store.py#L274-L301
[vllm-turbo-packing-select]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/turboquant_attn.py#L359-L370
[vllm-layout-worker]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/worker/worker_base.py#L107-L114
[vllm-layout-engine]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/engine/core.py#L290-L313
[vllm-layout-resolver]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/utils.py#L188-L322
[vllm-cache-content]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/triton_attn.py#L274-L287
[vllm-cache-scale-views]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/triton_attn.py#L381-L447
[vllm-turbo-record]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/model_executor/layers/quantization/turboquant/config.py#L130-L177
[vllm-nvfp4-views]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/utils/torch_utils.py#L543-L597
[vllm-nvfp4-layout-write]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu#L263-L378
[vllm-quant-doc]: https://docs.vllm.ai/en/latest/features/quantization/
[vllm-quant-interface]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/model_executor/layers/quantization/base_config.py
[vllm-online-doc]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/docs/features/quantization/online.md
[vllm-kv-doc]: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
[vllm-kv-scales]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/model_executor/layers/quantization/kv_cache.py#L106-L176
[vllm-kv-heads]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py#L1035-L1100
[vllm-kv-write]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py#L144-L262
[vllm-cache-config]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/config/cache.py#L39-L58
[vllm-query-quant]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/model_executor/layers/attention/attention.py#L463-L517
[vllm-fa-call]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flash_attn.py#L1259-L1295
[vllm-kv-cast]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/triton_unified_attention.py#L38-L56
[vllm-kv-compute]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/triton_unified_attention.py#L540-L589
[vllm-kv-modes]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/kv_cache_interface.py#L39-L104
[vllm-nvfp4-config]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/config/cache.py#L111-L120
[vllm-nvfp4-write]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu#L162-L203
[vllm-turbo-config]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/model_executor/layers/quantization/turboquant/config.py#L17-L105
[vllm-triton-cache-types]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/triton_attn.py#L294-L304
[vllm-flashinfer-cache-types]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L414-L423
[vllm-int4-origin]: https://github.com/vllm-project/vllm/pull/39074
[vllm-int4-parent]: https://github.com/vllm-project/vllm/pull/40633
[vllm-int4-pr]: https://github.com/vllm-project/vllm/pull/40835
[vllm-int4-merge]: https://github.com/vllm-project/vllm/commit/70749fdcca7da31be7206101e6f6fc77de9e6839
[vllm-turbo-pr]: https://github.com/vllm-project/vllm/pull/38479
[quarot-paper]: https://arxiv.org/abs/2404.00456
[vllm-int4-flow]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/int4_per_token_head.py#L855-L949
[vllm-int4-rht]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/int4_per_token_head.py#L1073-L1161
[vllm-int4-pack]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/int4_per_token_head.py#L114-L245
[vllm-int4-read]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/int4_per_token_head.py#L490-L608
[vllm-int4-dispatch]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/triton_unified_attention.py#L861-L871
[vllm-turbo-rotation]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/turboquant_attn.py#L409-L472
[vllm-turbo-write]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/triton_turboquant_store.py#L350-L448
[vllm-turbo-centroids]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/model_executor/layers/quantization/turboquant/centroids.py#L34-L85
[vllm-turbo-value]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/triton_turboquant_store.py#L26-L212
[vllm-turbo-read]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/triton_turboquant_decode.py#L157-L290
[vllm-turbo-query]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/triton_turboquant_decode.py#L519-L528
[vllm-turbo-soa]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/ops/turboquant_soa/triton_turboquant_store.py#L381-L408
[vllm-turbo-prefill]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/turboquant_attn.py#L723-L903
[vllm-turbo-continuation]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/turboquant_attn.py#L907-L1060
[flashattention-project]: https://github.com/Dao-AILab/flash-attention
[trtllm-project]: https://github.com/NVIDIA/TensorRT-LLM
[flashinfer-project]: https://github.com/flashinfer-ai/flashinfer
[triton-language]: https://triton-lang.org/main/index.html
[flashinfer-trtllm-api]: https://docs.flashinfer.ai/generated/flashinfer.decode.trtllm_batch_decode_with_kv_cache.html
[vllm-fa-dependency]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/cmake/external_projects/vllm_flash_attn.cmake#L25-L48
[vllm-triton-query]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/triton_attn.py#L483-L528
[vllm-fi-query-policy]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L933-L973
[vllm-fi-query-convert]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L1926-L1950
[vllm-fi-nvfp4-query]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L2278-L2342
[vllm-fa-fp8-gate]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/fa_utils.py#L282-L312
[vllm-fa-version-policy]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/fa_utils.py#L74-L219
[vllm-cuda-backend-selection]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/platforms/cuda.py#L448-L513
[vllm-cuda-backend-priorities]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/platforms/cuda.py#L154-L180
[fa3-dtype-dispatch]: https://github.com/vllm-project/flash-attention/blob/506341a143fcabd4bb79052a7605ada727d6b3f5/hopper/flash_api.cpp#L296-L426
[vllm-attention-registry]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/registry.py#L34-L173
[vllm-fi-kernel-imports]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L5-L49
[vllm-fi-mla-adapter]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/mla/flashinfer_mla.py#L1-L11
[vllm-mla-prefill-registry]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/mla/prefill/registry.py#L34-L62
[vllm-trtllm-ragged-adapter]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py#L120-L157
[vllm-flex-adapter]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flex_attention.py#L11-L23
[vllm-composite-adapter]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/triton_flashinfer.py#L1-L22
[vllm-aiter-adapter]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/rocm_aiter_unified_attn.py#L1-L31
[vllm-fa-query-capability]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flash_attn.py#L1148
[fa3-input-dtypes]: https://github.com/vllm-project/flash-attention/blob/506341a143fcabd4bb79052a7605ada727d6b3f5/hopper/flash_api.cpp#L781-L789
[fa4-kv-dequant]: https://github.com/vllm-project/flash-attention/blob/506341a143fcabd4bb79052a7605ada727d6b3f5/flash_attn/cute/interface.py#L736-L755
[vllm-fa4-wrapper]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/vllm_flash_attn/flash_attn_interface.py#L418-L466
[fa2-input-dtypes]: https://github.com/vllm-project/flash-attention/blob/506341a143fcabd4bb79052a7605ada727d6b3f5/csrc/flash_attn/flash_api.cpp#L598-L602
[vllm-fa-cache-allowlist]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flash_attn.py#L284-L292
[vllm-fa-shape-gates]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flash_attn.py#L296-L405
[vllm-triton-shape-gates]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/triton_attn.py#L289-L378
[vllm-fi-shape-gates]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L413-L518
[vllm-fi-decode-selection]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L810-L858
[vllm-fi-path-policy]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/utils/flashinfer.py#L592-L733
[vllm-fi-phase-dispatch]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L1320-L1373
[trtllm-xqa-origin]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/gpt-attention.md#xqa-optimization
[fi-xqa-api]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/xqa.py#L370-L410
[fi-xqa-jit]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/xqa.py#L58-L94
[fi-xqa-sm90]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/csrc/xqa/mha_sm90.cu
[fi-xqa-generic]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/csrc/xqa/mha.cu
[vllm-turbo-gates]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/turboquant_attn.py#L126-L192
[vllm-turbo-impl]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/turboquant_attn.py#L300-L370
[vllm-selector-inputs]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/selector.py#L110-L196
[vllm-backend-validation]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backend.py#L273-L362
[llama-cache-allocation]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L164-L291
[llama-cache-streams]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L83-L153
[llama-cache-strides]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml.c#L1833-L1837
[llama-cache-cells]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cells.h#L34-L82
[llama-cache-slot-indices]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L1476-L1520
[llama-cache-views]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L1265-L1316
[llama-cache-mask]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L1638-L1655
[llama-cache-slot-allocation]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L1021-L1156
[llama-cache-search]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L998-L1094
[llama-cache-cursor]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L1180-L1185
[llama-cache-extent]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L1250-L1263
[llama-cache-removal]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cache.cpp#L382-L447
[llama-cache-cell-release]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/src/llama-kv-cells.h#L228-L263
[sg-revision]: https://github.com/sgl-project/sglang/commit/2f5c9ac43d76eef11e8561c89c55a79db7db715a
[sg-pool]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/mem_cache/memory_pool.py
[sg-pool-config]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/mem_cache/kv_cache_configurator.py
[sg-token-alloc]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/mem_cache/allocator/token.py
[sg-paged-alloc]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/mem_cache/allocator/paged.py
[sg-radix]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/mem_cache/radix_cache.py
[sg-index]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/mem_cache/kv_index_translator.py
[sg-fa]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/layers/attention/flashattention_backend.py
[sg-fi]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/layers/attention/flashinfer_backend.py
[sg-triton]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/layers/attention/triton_backend.py
[sg-attn-hook]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/arg_groups/attention_hook.py
[sg-kv-hook]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/arg_groups/kv_cache_hook.py
[sg-registry]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/layers/attention/attention_registry.py
[sg-dependencies]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/pyproject.toml
[sg-quant-doc]: https://docs.sglang.io/docs/advanced_features/quantized_kv_cache
[iq4-nl-format]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-common.h
[iq4-nl-code]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-quants.c
[iq4-nl-origin]: https://github.com/ggml-org/llama.cpp/pull/5590
[iq4-nl-cuda]: https://github.com/ggml-org/llama.cpp/blob/a894dae939d426954ce54bb604824f1ae918a0c5/ggml/src/ggml-cuda/dequantize.cuh

[sg-mxfp8-quant]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/kernels/ops/quantization/mxfp8_quant.py
[sg-fp4-algorithms]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/layers/quantization/kvfp4_tensor.py
[sg-mxfp8-pool]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/mem_cache/memory_pool.py#L3491-L3850
[sg-fp4-method]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py
[sg-nvfp4-storage]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/layers/quantization/nvfp4_kv_cache.py
[sg-trtllm]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/layers/attention/trtllm_mha_backend.py
[sg-triton-decode]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/kernels/ops/attention/decode_attention.py
[sg-fa4-interface]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/kernels/ops/attention/flash_attn/cute/interface.py
[sg-fa4-sm100]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm100.py
[sg-fa4-wrapper]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/kernels/ops/attention/flash_attention_v4.py
[fi-fa2-kv-read]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/include/flashinfer/attention/prefill.cuh

[hf-revision]: https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/cache_utils.py
[hf-cache]: https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/cache_utils.py
[hf-cache-update]: https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/cache_utils.py#L717-L794
[hf-llama]: https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/models/llama/modeling_llama.py#L191-L280
[hf-sdpa]: https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/integrations/sdpa_attention.py
[hf-flash]: https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/integrations/flash_attention.py
[hf-generation]: https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/generation/utils.py#L2138-L2148
[hf-attn-select]: https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/modeling_utils.py#L1713-L1881
[quanto-max]: https://github.com/huggingface/optimum-quanto/blob/5b94afb31bcdaf7e6acde1f4d88555d8e10bd85c/optimum/quanto/tensor/optimizers/max_optimizer.py
[quanto-qbits]: https://github.com/huggingface/optimum-quanto/blob/5b94afb31bcdaf7e6acde1f4d88555d8e10bd85c/optimum/quanto/tensor/qbits.py
[quanto-group]: https://github.com/huggingface/optimum-quanto/blob/5b94afb31bcdaf7e6acde1f4d88555d8e10bd85c/optimum/quanto/tensor/grouped.py
[quanto-weight-qbits]: https://github.com/huggingface/optimum-quanto/blob/5b94afb31bcdaf7e6acde1f4d88555d8e10bd85c/optimum/quanto/tensor/weights/qbits.py
[quanto-pack]: https://github.com/huggingface/optimum-quanto/blob/5b94afb31bcdaf7e6acde1f4d88555d8e10bd85c/optimum/quanto/tensor/packed.py
[hqq-quant]: https://github.com/mobiusml/hqq/blob/d88a488ec8aa2d58362ef2038a52bca862db2e74/hqq/core/quantize.py
[hqq-optimize]: https://github.com/mobiusml/hqq/blob/d88a488ec8aa2d58362ef2038a52bca862db2e74/hqq/core/optimize.py
[hqq-pack]: https://github.com/mobiusml/hqq/blob/d88a488ec8aa2d58362ef2038a52bca862db2e74/hqq/core/bitpack.py
[hf-cache-guide]: https://huggingface.co/docs/transformers/main/en/kv_cache#quantized-cache
[kivi-paper]: https://arxiv.org/abs/2402.02750v2
[sg-graph-padding]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/model_executor/cuda_graph_buffer_registry.py#L536-L580
[vllm-fi-page-metadata]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L1250-L1288
[vllm-fi-address-routing]: https://github.com/vllm-project/vllm/blob/17e50b9b761023d5f2499f062507df1ff49092a4/vllm/v1/attention/backends/flashinfer.py#L1405-L1475
[sg-default-attn]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/arg_groups/model_override_base.py#L247-L304
[sg-attn-overrides]: https://github.com/sgl-project/sglang/blob/2f5c9ac43d76eef11e8561c89c55a79db7db715a/python/sglang/srt/arg_groups/overrides.py

[mnn-revision]: https://github.com/alibaba/MNN/commit/67450001627fc91232779068d85933c22845fae5
[mnn-docs]: https://github.com/alibaba/MNN/blob/67450001627fc91232779068d85933c22845fae5/docs/transformers/llm.md
[mnn-runtime]: https://github.com/alibaba/MNN/blob/67450001627fc91232779068d85933c22845fae5/transformers/llm/engine/src/llm.cpp#L182-L197
[mnn-cpu-attn]: https://github.com/alibaba/MNN/blob/67450001627fc91232779068d85933c22845fae5/source/backend/cpu/CPUAttention.cpp
[mnn-cpu-cache]: https://github.com/alibaba/MNN/blob/67450001627fc91232779068d85933c22845fae5/source/backend/cpu/CPUKVCacheManager.cpp
[mnn-cpu-quant]: https://github.com/alibaba/MNN/blob/67450001627fc91232779068d85933c22845fae5/source/backend/cpu/compute/CommonOptFunction.cpp#L1571-L1756
[mnn-tq]: https://github.com/alibaba/MNN/blob/67450001627fc91232779068d85933c22845fae5/source/backend/cpu/compute/TurboQuant.hpp
[mnn-metal-attn]: https://github.com/alibaba/MNN/blob/67450001627fc91232779068d85933c22845fae5/source/backend/metal/MetalAttention.mm
[mnn-metal-cache]: https://github.com/alibaba/MNN/blob/67450001627fc91232779068d85933c22845fae5/source/backend/metal/MetalKVCacheManager.mm
[mnn-metal-shader]: https://github.com/alibaba/MNN/blob/67450001627fc91232779068d85933c22845fae5/source/backend/metal/MetalAttentionShader.hpp
[mnn-metal-flash]: https://github.com/alibaba/MNN/blob/67450001627fc91232779068d85933c22845fae5/source/backend/metal/MetalFlashAttnShader.hpp

[metal-tensorops]: https://developer.apple.com/videos/play/wwdc2026/330/
[metal-capabilities]: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
[metal-m5-int8]: https://max.modular.com/stable/api/mojo/linalg/matmul/gpu/apple/int8_matmul/
[metal-m5-mma]: https://github.com/modular/modular/blob/main/max/mojo/max/gpu/compute/arch/mma_apple.mojo
[metal-m5-quant-kernels]: https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8
[metal-m5-fp4]: https://max.modular.com/stable/api/mojo/linalg/matmul/gpu/apple/matmul2d_fp4/
[metal-m4-rigel]: https://arxiv.org/abs/2606.12765

[et-revision]: https://github.com/pytorch/executorch/commit/12668f8b81c39e2a93d5e125127526671fd7c69f
[et-export]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/extension/llm/export/export_llm.py
[et-config]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/extension/llm/export/config/llm_config.py
[et-cache]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/examples/models/llama/source_transformation/custom_kv_cache.py
[et-exportlib]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/examples/models/llama/export_llama_lib.py
[et-qnn]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/backends/qualcomm/quantizer/custom_annotation.py
[et-tq-cache]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/extension/llm/modules/turboquant/kv_cache.py
[et-tq-book]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/extension/llm/modules/turboquant/codebook.py
[et-base-cache]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/extension/llm/modules/kv_cache.py
[et-qnn-cache]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/examples/qualcomm/oss_scripts/llama/runner/kv_manager.cpp
[et-sdpa]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/examples/models/llama/source_transformation/sdpa.py
[et-cpu-sdpa]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/extension/llm/custom_ops/op_sdpa_impl.h
[et-tq-cuda]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/backends/cuda/triton/kernels/tq4_sdpa.py
[et-tq-export]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/examples/models/qwen3_5_moe/export.py
[et-tq-model]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/examples/models/qwen3_5_moe/model.py
[et-tq-mlx]: https://github.com/pytorch/executorch/blob/12668f8b81c39e2a93d5e125127526671fd7c69f/backends/mlx/llm/turboquant_cache.py
[et-qnn-doc]: https://docs.pytorch.org/executorch/1.5/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.html
