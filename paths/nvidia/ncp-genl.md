# NVIDIA NCP-GENL path: Generative AI LLMs (Professional)

**Who this is for / prerequisites.** You have finished the [ML core](../../core-ml-engineering.md) (M0–M10) and the [AI core](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/core-llm-app-engineering.md) (C0–C13). You want the NVIDIA credential that tests model building: distributed training, quantization, profiling, TensorRT-LLM and Dynamo-Triton. **From the NVIDIA AI path:** if you took it, you already have this. Otherwise do just this module from it first: **NV-AMAD.1** parts 1A–1E (7.5 h: NGC, build.nvidia.com, NIM, NeMo family overview, NeMo Guardrails, TensorRT-LLM/Dynamo-Triton at concept level) in [nca-amad.md](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/nvidia/nca-amad.md). Part 1F is not needed here. This file teaches only the NVIDIA tooling, hardware specifics and exam framing. Roughly half of the blueprint is a light refresh of core modules [estimate: domains 1–3, 6, 9 and 10 plus parts of 4–5]. All exercises are in Python. Facts are as of 2026-10-31.

## Contents

- [Exam at a glance](#exam-at-a-glance)
- [NV-GENL.0 Core refresh: already covered](#nv-genl0-core-refresh-already-covered)
- [New modules NV-GENL.1–NV-GENL.10](#new-modules)
- [Coverage matrix](#coverage-matrix)
- [Schedule](#schedule)
- [Practice, readiness and exam day](#practice-readiness-and-exam-day)
- [Short note: NCA-AIIO](#short-note-nca-aiio)
- [Combining with other paths](#combining-with-other-paths)
- [Skipped on purpose](#skipped-on-purpose)
- [Gaps for the core](#gaps-for-the-core)
- [Sources](#sources)

## Exam at a glance

All facts below are as of 2026-10-31.

| Item | Value | Source |
|---|---|---|
| Name / code | NVIDIA-Certified Professional: Generative AI LLMs, **NCP-GENL** ("intermediate-level credential") | [exam page][p] |
| Status | Live. Registration goes through Certiverse and a Credly badge exists. The page still contains a hidden "(Coming soon)" leftover. The launch date is **unconfirmed** (study guide dated Sep 2025). | [p], [credly] |
| Price | $200 (taxes may apply) | [p], [cat] |
| Format | 60–70 multiple-choice questions, 120 min, online, remotely proctored. The page says English. | [p] |
| Passing score | **Not published.** Exams are pass/fail and you do not get a score. | [hub] |
| Eligibility | None formal. NVIDIA recommends 2–3 years of LLM work, Python (C++ a plus), Docker/K8s, distributed parallelism and PEFT. | [p] |
| Validity | 2 years. Renewing means retaking the exam. | [p], [hub] |
| Retakes | 14-day wait, at most 5 attempts per 12 months, and each attempt is paid | [hub] |
| Scheduling | Book up to 60 days ahead. Reschedule or cancel until 24 h before. No refund inside 24 h. | [hub] |
| Discount | Attendees of the "NVIDIA Certification: Your Roadmap Across the AI Stack" webinar (sessions on 2026-11-19) get a 50% exam code | [web] |
| Blueprint (Sep 2025 guide) | 1 LLM Architecture 6% · 2 Prompt Engineering 13% · 3 Data Preparation 9% · 4 Model Optimization 17% · 5 Fine-Tuning 13% · 6 Evaluation 7% · 7 GPU Acceleration & Optimization 14% · 8 Model Deployment 9% · 9 Production Monitoring & Reliability 7% · 10 Safety, Ethics & Compliance 5% | [g] |
| Official practice test | **None found** (only NCA-AMAD and NCA-AIIO have official practice sets) | [hub], research |

Use the **study guide**, not the web page, as the blueprint. On the page, the Fine-Tuning description is copy-pasted from the OpenUSD exam, and the Model Optimization row describes deployment [p].

## NV-GENL.0 Core refresh: already covered

**2 h** · prereqs: both cores · Tested by: PGENL (all domains)

**Skip test**: none. The refresh table below is the test: any row you can answer cold in under 2 min is done.

- [ ] Read the study guide's task lists 1.1–10.5 (40 min). Skip its course lists: several are retired (see [Skipped](#skipped-on-purpose)).
- [ ] Do each refresh action below (about 1 h 20 min total). If one takes more than 15 min, re-do that core module's exercise instead.

| Core module | Exam tasks it owns | Light refresh action |
|---|---|---|
| C1.2, M5.1 | 1.1, 1.2, 1.6, 8.1 | Draw an encoder-only, a decoder-only and an encoder-decoder block from memory. State which one T5, BERT and GPT use and why decoders dominate serving cost (8.1). |
| C1.3 | 1.4, 1.5, 4.5 (temperature) | List what temperature, top-k and top-p each do to the distribution in one line each |
| C1.1, C11.1 | 5.4 (hallucination causes, FT-vs-RAG) | Name three causes of hallucination and the cheapest fix for each |
| C2.1, C2.2 | 2.1, 2.2 | Re-run your C2.2 prompt regression test |
| C2.3 | 2.4 (validation and re-ask) | Re-run the C2.3 extractor and break the schema on purpose |
| M5.2 | 3.3 | Explain the BPE and WordPiece merge criteria and the effect of vocab size on embedding params and sequence length |
| M5.1, M5.3, M5.5 | 2.3, 4.7 (causal LM / MLM training) | Reopen build-nanogpt and the M5.5 MLM notebook and name the loss and masking difference |
| M5.4, M5.8 | 1.3 | Write a 20-line script: mean-pooled BERT embedding vs last-token decoder embedding for the same sentence |
| M3.1, M3.2, M3.4 | 3.1 | Re-state the imbalance options and your M3.4 data tests |
| C11.3, M5.7 | 3.2 (dataset formats) | List the SFT, chat-template and preference-pair JSONL shapes |
| M7.2, C10.2 | 4.1–4.3 (PTQ/QAT/pruning/distillation algorithms) | Explain PTQ vs QAT vs distillation in 3 sentences, plus GPTQ vs AWQ |
| M4.3, M4.4, M7.4 | 4.4, 5.3 | Re-read your M4.4 tuning log, noting LR schedule, batch-size scaling and early stopping |
| C11.2, M5.7, M5.8 | 5.1, 5.2 (SFT/DPO/GRPO/LoRA/contrastive) | Explain DPO vs GRPO vs RLHF in one line each and InfoNCE in one formula |
| M5.9, C3.2, C8.1 | 6.1 | Compute perplexity and ROUGE-L by hand for a toy example. State judge-bias checks. |
| C3.1, C8.2, C3.3 | 6.2, 6.4 | Re-open your C3.1 error-analysis sheet |
| C10.1, C10.4 | 4.6 (KV cache concept), 6.3 | Restate TTFT, ITL, throughput and goodput, and the KV-cache memory formula |
| M7.1, M7.3 | 7.1–7.4 (concepts) | Explain DDP vs FSDP vs TP vs PP vs CP vs EP in one line each, plus the 16-bytes/param Adam memory rule |
| M6.5, M6.6 | 8.2, 8.3 (containers, K8s, canary) | Re-deploy your M6.6 KServe InferenceService |
| C8.3, C8.4, M6.7, M6.8, M6.9 | 9.1–9.5 | Re-read your C8.4 eval gate and M6.8 drift alert |
| C9.4, C9.5, M9.1 | 10.1–10.5 | List the guardrail types and two fairness metrics |
| NV-AMAD.1 | NIM, NeMo Curator/Evaluator/Guardrails, NGC (named in readings for 3.1, 6.4, 8.3, 10.5) | Re-read NV-AMAD.1 notes on NIM profiles, the NIM Operator and Guardrails rail types |

<a id="new-modules"></a>
## New modules

### NV-GENL.1 NVIDIA GPU platform for model builders

**4 h** · prereqs: M7.1, M7.2, NV-AMAD.1 · Tested by: PGENL 4.2, 7.2; AIIO 1.8, 3.4

**Skip test** (passing means you skip the reading):
1. Which GPU architectures have FP8 Tensor Cores, and which one first runs FP4 natively?
2. Why is tensor parallelism normally kept inside one NVLink domain?
3. MIG vs time-slicing: which one gives memory isolation?
4. Can a 70B model be fully fine-tuned with BF16 Adam on 8×80 GB without sharding? Show the arithmetic.

**Concepts**
- Tensor Core precision by generation. Volta: FP16. Turing (T4): FP16/INT8/INT4. Ampere (A100): TF32, BF16, 2:4 structured sparsity, MIG. Ada (L4/L40S) and Hopper (H100): FP8 via Transformer Engine. Blackwell: NVFP4/FP4. TensorRT's support matrix (version 11.4.1, checked 2026-10-31) agrees: FP8 on L40S/H100 and newer, not on A100/A10/T4; BF16 and TF32 from Ampere (A100, A10) on; native FP4 needs Blackwell (SM 100+), and H100/L40S only emulate it.
- Memory and bandwidth hierarchy: HBM, then NVLink/NVSwitch inside a node, then InfiniBand/Ethernet between nodes. This sets the parallelism layout in NV-GENL.2.
- MIG: GPU instances vs compute instances, profile names such as `1g.10gb`, and hardware isolation. MPS and time-slicing share a GPU without isolation.
- The vocabulary NVIDIA expects you to recognise: NGC framework containers as the reproducible base (NGC itself → NV-AMAD.1), DGX, Base Command and AI Enterprise.

**Resources**
- NVIDIA *GPU Performance Background User's Guide*, whole (45 min)
- *MIG User Guide*: Introduction, Supported GPUs, Concepts, Supported MIG Profiles (1 h)
- TensorRT *Support Matrix*, section "GPU Architecture and Precision Support" plus its footnotes (20 min)

**Exercise** (about 2 h): write `gpu_fit.py`. Inputs: parameter count, precision, GPU or MIG profile, context length and batch. It prints memory for inference (weights + KV cache) and for training (weights + grads + Adam states + activations estimate).
- [ ] Done when it reports that 8B full fine-tuning in BF16 with Adam (about 128 GB) does not fit on one 80 GB GPU while LoRA does, and gives the KV-cache headroom for 70B FP8 on one H100.

**Self-check**
- [ ] Explain TF32 to a colleague.
- [ ] Name 2 workloads where MIG beats a whole GPU and 1 where it hurts.

### NV-GENL.2 NeMo training stack and parallelism configuration

**6 h** · prereqs: M7.3, M5.3, NV-GENL.1 · Tested by: PGENL 7.1, 7.3, 4.4

**Skip test**
1. 64 GPUs with TP=8 and PP=2 (no CP). What is DP?
2. Global batch 1024, micro-batch 2, DP 4. How many gradient-accumulation steps?
3. Which knob shards activations along the sequence for long contexts, and which one pairs with TP?
4. What does `use_distributed_optimizer` shard?

**Concepts**
- The 2026 stack map:
  - **Megatron Core**: the parallelism library.
  - **Megatron Bridge**: HF↔Megatron checkpoint conversion plus training recipes.
  - **NeMo AutoModel**: "PyTorch DTensor-native SPMD" training of any HF model, no conversion step.
  - **NeMo RL**: post-training.
  - **NeMo Curator**: data curation (→ NV-AMAD.1).
- Repo moves: NVIDIA/NeMo now redirects to NVIDIA-NeMo/Speech.
- Configuration knobs: `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `virtual_pipeline_model_parallel_size`, `sequence_parallel`, `context_parallel_size`, `expert_model_parallel_size`, `use_distributed_optimizer`.
- Layout arithmetic: world = TP×PP×CP×DP, and EP is carved out of DP for MoE layers. Check divisibility of heads by TP, layers by PP and experts by EP.
- Strategy selection:
  - TP inside a node, PP across nodes.
  - Interleaved (virtual) PP to shrink the pipeline bubble.
  - CP for long sequences, EP for MoE.
  - Distributed optimizer instead of plain DDP when optimizer state dominates.
- Distributing attention-head GEMMs means column/row-parallel linear layers (task 7.3).
- Gradient accumulation = global batch ÷ (micro-batch × DP). Activation recomputation trades compute for memory.

**Resources**
- Megatron Bridge *Parallelisms Guide*, whole, including "Strategy Selection Guide" and "Configuration Guidelines" (1.5 h)
- NVIDIA blog *Mastering LLM Techniques: Training*, whole (45 min)
- NeMo AutoModel docs: "About", "Key Features", "Distributed Setup (Python API)" and "Pipeline Parallelism" (45 min)
- Megatron-LM README, "About" section (Megatron-LM vs Megatron Core vs Megatron Bridge) (20 min)

**Exercise** (about 3 h): write `parallel_planner.py`. Given GPUs per node, nodes, layers, heads, experts and model size, it enumerates valid (TP, PP, CP, DP, EP) layouts, rejects invalid ones with a reason, estimates per-GPU memory (reuse `gpu_fit.py`) and outputs gradient-accumulation steps.
- [ ] Done when it accepts the guide's combined-parallelism example and rejects TP=16 on 8-GPU nodes.
- [ ] Optional (rented 2-GPU box, about $5 [estimate]): run one AutoModel recipe with FSDP2 and then with TP=2, and compare tokens/s.

**Self-check**
- [ ] Explain why PP adds a bubble and how virtual PP reduces it.
- [ ] Pick a layout for a 70B dense model on 4 nodes × 8 H100 and justify it.

### NV-GENL.3 Customization and alignment on NeMo

**5 h** · prereqs: M5.7, M5.8, C11.2, NV-GENL.2 · Tested by: PGENL 2.1, 2.3, 3.2, 4.7, 5.1–5.4

**Skip test**
1. Prompt tuning vs P-tuning vs prefix tuning: what is trained in each?
2. Which NeMo library runs GRPO, and which runs HF-native LoRA SFT?
3. What are BERT's two pretraining objectives, and which one did RoBERTa drop?
4. What fields does a DPO record need?

**Concepts**
- AutoModel SFT vs PEFT (LoRA, under 1% of params) recipes, with HF checkpoints in and out.
- NeMo RL algorithms: SFT, DPO, GRPO, PPO, reward modelling and on-policy distillation. Training backends: PyTorch/DTensor (via NeMo AutoModel) and Megatron; generation via vLLM or Megatron.
- The prompt-learning family, which the exam calls "prompt learning for small datasets":
  - Prompt tuning: soft prompt embeddings.
  - P-tuning: a prompt encoder (MLP/LSTM) generates virtual tokens.
  - Prefix tuning: per-layer prefixes.
  - Base model frozen in all three. Compare with LoRA and adapters.
  - AutoModel's "SFT & PEFT" guide documents only LoRA and QLoRA (checked 2026-10-31). Whether any current NeMo library still ships P-tuning is **unconfirmed**; learn it as exam vocabulary.
- MLM + NSP as encoder pretraining vocabulary (4.7).
- Preference data shape: prompt / chosen / rejected.
- Assessing fine-tuning impact: same eval before and after, plus a forgetting check (→ C11.1).

**Resources**
- NeMo AutoModel docs: "SFT & PEFT" recipe page (Supervised Fine-Tuning and Parameter-Efficient Fine-Tuning with NeMo AutoModel) (1.5 h)
- NeMo RL docs: "Overview", "Direct Preference Optimization in NeMo RL", "An In-depth Walkthrough of GRPO in NeMo RL" (1.5 h)
- Optional exam reading: NVIDIA blog *Train a Reasoning-Capable LLM in One Weekend With NVIDIA NeMo* (30 min)
- The customization-technique decision post belongs to NV-AMAD.3 and is not reassigned here

**Exercise** (about 2 h, Colab or rented L4): port your M5.7 TRL LoRA run to AutoModel (same base model and data), then run DPO in NeMo RL on about 500 preference pairs. If DPO does not fit your GPU, use the smallest supported model.
- [ ] Done when both runs log to MLflow (M6.2) and your M5.9 harness reports base vs LoRA vs DPO on one table.

**Self-check**
- [ ] Pick a method for: 300 labelled examples; a new output style; preference data only.

### NV-GENL.4 Quantization, sparsity and distillation with NVIDIA tools

**6 h** · prereqs: M7.2, C10.2, NV-GENL.1 · Tested by: PGENL 4.1, 4.2, 4.3, 4.6

**Skip test**
1. Why can't an A100 run FP8 Tensor Core GEMMs?
2. Does weight-only INT4 speed up prefill or decode more, and why?
3. What do Q/DQ nodes mean in TensorRT explicit quantization?
4. What hardware pattern is "2:4" sparsity?

**Concepts**
- **NVIDIA Model Optimizer** (formerly TensorRT Model Optimizer, renamed Dec 2025):
  - PTQ formats: FP8, INT8 SmoothQuant, INT4 AWQ, NVFP4.
  - Calibration set size and choice; QAT and quantization-aware distillation.
  - Pruning, distillation, sparsity.
  - Export as a unified Hugging Face checkpoint that TensorRT-LLM, vLLM and SGLang load (the TensorRT-LLM-specific checkpoint export is deprecated).
- Weight-only vs weight+activation quantization. FP8 KV-cache quantization.
- TensorRT explicit (Q/DQ) quantization. Implicit, calibrator-based INT8 quantization was removed in TensorRT 11.0, but the Sep 2025 guide predates that, so know both terms.
- 2:4 structured sparsity combined with INT8.
- Choose by GPU (FP8 on Ada/Hopper+, NVFP4 on Blackwell) and by bottleneck (memory-bound decode vs compute-bound prefill).
- Report the accuracy delta and the speed delta together.

**Resources**
- TensorRT-LLM docs, "Quantization" feature page, including the model and hardware support matrices (45 min)
- Model Optimizer docs: Getting Started, Quantization guide, Pruning and Distillation overviews (1.5 h)
- TensorRT *Working with Quantized Types* (45 min)
- NVIDIA blog *Sparsity in INT8: Training Workflow and Best Practices for TensorRT Acceleration* (30 min)
- Paid option: DLI *The Art of Compressing LLMs* ($90 / 8 h [cat]). Take it only if you want guided labs; this module is the free fallback.

**Exercise** (about 2.5 h, rented L4/L40S, about $5 [estimate]): quantize your M5.7 model to FP8 and INT4-AWQ with Model Optimizer. Measure perplexity and one task metric (M5.9), peak memory and tokens/s.
- [ ] Done when `quant_report.md` compares BF16, FP8 and INT4 with deltas and recommends one variant for "H100, latency SLO" and one for "L4, memory-bound".

**Self-check**
- [ ] Explain what a bad calibration set does.
- [ ] Explain when QAT is worth its cost over PTQ.

### NV-GENL.5 TensorRT-LLM: batching, KV cache and decoding

**6 h** · prereqs: C10.1, C10.3, NV-GENL.4, NV-AMAD.1 · Tested by: PGENL 1.4, 2.4, 4.5, 4.6, 6.3

**Skip test**
1. What problem does chunked context (chunked prefill) solve?
2. When does KV-cache reuse help, and what invalidates it?
3. Beam search width 4 vs sampling: what is the cost in KV memory and latency?
4. Name three speculative-decoding modes TensorRT-LLM supports.

**Concepts**
- The LLM API (Python) and `trtllm-serve` (OpenAI-compatible). PyTorch is now the **only** execution backend: the TensorRT engine backend, `trtllm-build` and per-model checkpoint conversion were removed, and Hugging Face checkpoints load directly (TensorRT-LLM "Migration Guide: TensorRT Backend Removed"). Older exam items may still describe engine builds.
- In-flight batching, paged KV cache, chunked context, scheduler policies, KV-cache memory fraction, KV reuse across requests.
- Attention window / sliding-window attention (for "streaming attention", 4.6).
- Speculative decoding modes on the current docs page: draft/target, EAGLE 3, n-gram, MTP, PARD, DFlash and user-provided drafting. Medusa appears only in the legacy (TensorRT-backend) docs.
- Decoding strategies (docs "Sampling" page):
  - Beam search: width, length penalty.
  - Penalties (repetition, presence, frequency).
  - Guided decoding (JSON schema, regex, grammar; docs page "Guided Decoding"), which gives the server-side constrained decoding for 2.4.
- `trtllm-bench` for throughput and latency.

**Resources**
- TensorRT-LLM docs (3 h; if you did NV-AMAD.1 part 1B you already read the first page and the reuse section of the second at concept level, so read only their configuration sections and save about 30 min):
  - "Paged Attention, IFB, and Request Scheduling" (configuration and scheduler-policy sections)
  - "KV Cache System" (sections other than reuse)
  - "Speculative Decoding"
  - "Sampling" and "Guided Decoding"
  - "LLM API Introduction"
  - `trtllm-serve`
  - `trtllm-bench`
- TensorRT *Best Practices*, "Performance Benchmarking" page (30 min)

**Exercise** (about 2.5 h): serve the NV-GENL.4 FP8 checkpoint with `trtllm-serve` and call it through your C0.2 `llm.py` adapter. Benchmark at concurrency 1, 8 and 32, with and without KV reuse, on a shared-prefix dataset. Run one beam-search config.
- [ ] Done when a TTFT / ITL / throughput table shows the effect of reuse and beam search.

**Self-check**
- [ ] Map each TensorRT-LLM feature to the C10.1 metric it moves.

### NV-GENL.6 Dynamo-Triton serving

**6 h** · prereqs: M6.5, M6.6, NV-GENL.5 · Tested by: PGENL 8.1–8.3, 9.1; PADS 6.5 (via NV-ADS)

**Skip test**
1. Name the files and folders of a Triton model repository.
2. What do `instance_group [{count: 2, kind: KIND_GPU}]` and `max_queue_delay_microseconds` do?
3. When do you need the sequence batcher?
4. Ensemble vs BLS: which one allows loops and conditionals?

**Concepts**
- Naming: NVIDIA Dynamo-Triton was previously Triton Inference Server; the docs still say "Triton". Dynamo's disaggregated serving is concept-only here (→ NV-AMAD.1).
- Model repository layout (`config.pbtxt`, versions) and backends: TensorRT-LLM, Python, ONNX Runtime, PyTorch, FIL.
- Schedulers: default, dynamic batcher (preferred sizes, queue delay), sequence batcher (stateful), ensemble.
- Concurrent model execution, BLS, response cache, model-control modes.
- Metrics endpoint (Prometheus), perf_analyzer, Model Analyzer (config search).
- Helm/K8s deployment with autoscaling on queue metrics.
- When to choose Triton vs NIM vs plain `trtllm-serve`.

**Resources**
- Triton user guide (3 h), pages as titled in its sidebar: Repository (model repository), Configuration (model configuration), Scheduler, Batcher (already read in NV-AMAD.1 part 1B if you took that path: re-skim only the sequence-batcher section), Concurrent Model Execution, Ensemble, Business Logic Scripting, Caching (response cache), Metrics, Controls (model management)
- perf_analyzer README and Model Analyzer README (45 min). Note: the perf_analyzer repo says GenAI-Perf is deprecated in favour of AIPerf; perf_analyzer itself is still maintained.

**Exercise** (about 2.5 h): build a Triton repository with an ensemble: Python-backend preprocessing (prompt template, tokenize), then the model (TensorRT-LLM; or your M7.2 ONNX model if you have no GPU budget), then postprocessing. Sweep concurrency with perf_analyzer, dynamic batching off and on, and scrape `/metrics` into Prometheus (Docker).
- [ ] Done when the throughput gain from dynamic batching and the queue-time vs compute-time split are both visible.

**Self-check**
- [ ] Write a `config.pbtxt` from memory for a dynamic-batched ONNX model with 2 GPU instances.

### NV-GENL.7 Profiling and GPU troubleshooting

**6 h** · prereqs: M7.1, NV-GENL.1 · Tested by: PGENL 6.3, 7.2, 7.3, 7.4; PADS 2.5, 5.8; AIIO 3.3

**Skip test**
1. Nsight Systems vs Nsight Compute: which one finds a dataloader stall?
2. Which DCGM metric shows Tensor Core use?
3. Name three fixes for CUDA OOM that do not reduce the parameter count.
4. Why can a hidden size of 4097 be slower than 4096?

**Concepts**
- The APOD cycle (Assess, Parallelize, Optimize, Deploy).
- Nsight Systems:
  - `nsys profile --trace=cuda,nvtx,osrt --stats=true`
  - NVTX ranges in PyTorch
  - Timeline reading
- Nsight Compute: per-kernel roofline, occupancy, memory throughput.
- **DLProf: final release 1.8.0, discontinued.** It is still named in NCP-ADS 2.5/5.8, so know what it was: a framework-level profiler that summarised Tensor Core usage.
- DCGM: health, diagnostics, job statistics, `dcgm-exporter` metrics, XID errors. `nvidia-smi` fields.
- Common pathologies:
  - Host-device syncs and launch overhead (fix with CUDA graphs or `torch.compile`, → M7.1)
  - Input-pipeline starvation
  - Tensor-Core-unfriendly shapes (multiples of 8/16; tile and wave quantization)
  - Accidental FP32
  - NCCL stalls
- CUDA OOM:
  - Fragmentation (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)
  - Activation checkpointing
  - Micro-batch plus gradient accumulation

**Resources**
- NVIDIA blog *Assess, Parallelize, Optimize, Deploy* (15 min)
- *Matrix Multiplication Background User's Guide*, whole (1 h)
- Nsight Systems User Guide: "Profiling from the CLI", "CLI Profile Command Switch Options", "Marking and Labeling Regions" (1 h)
- DCGM docs, "Feature Overview" (30 min)
- Paid option: DLI *Optimizing CUDA ML Codes With Nsight Profiling Tools* ($30; the catalog says 2 h, the exam page says 4 h [cat], [p]). Worth it for guided Nsight Compute; this module is the free fallback.

**Exercise** (about 3 h): add NVTX ranges to your M4.3/M5.3 training loop, profile it with `nsys`, and fix two issues (for example pinned memory and workers, padded vocab, AMP). Read one Nsight Compute kernel report.
- [ ] Done when `nsys stats` before and after shows a step-time reduction, or you have a written root cause of why there is none.

**Self-check**
- [ ] Given a timeline with GPU idle gaps between kernels, list three causes in order of likelihood.

### NV-GENL.8 Production monitoring, reliability and safety on the NVIDIA stack

**3 h** · prereqs: C8.3, C8.4, M6.8, C9.4, NV-GENL.6, NV-AMAD.1 · Tested by: PGENL 9.1–9.5, 10.3, 10.5

**Skip test**
1. Name four GPU-side and four serving-side signals for an LLM dashboard.
2. How do you tell a model regression from a GPU fault?
3. Which guardrail cost must be measured before enabling it in production?

**Concepts**
- Dashboards combine three layers:
  - Serving metrics: queue time, compute time, TTFT/ITL from Triton, NIM or TensorRT-LLM
  - DCGM GPU metrics: SM and Tensor activity, memory, XID/ECC, thermals
  - Application traces (C8.3)
- Root-cause analysis path: XID/ECC → node; queue spike → capacity; quality drop with healthy GPUs → model or data (M6.8).
- Benchmark each release against the previous one with NV-GENL.5/.6 tools, gated in CI (C8.4).
- The retrain and version loop: M6.7, M6.9.
- Guardrail latency and accuracy measurement (NeMo Guardrails itself → NV-AMAD.1).
- Domain 9's tasks are word for word NCP-AAI domain 8, including "benchmark deployed agents". If you took NV-AAI, it already covers the task text: NV-AAI.3 (tasks 8.1, 8.2, 8.5), NV-AAI.2 (8.3) and NV-AMAD.3 (8.4, the data flywheel) in [ncp-aai.md](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/nvidia/ncp-aai.md). In that case skip this module's Grafana dashboard, alert rules and CI regression gate, and do only the GPU-specific additions: DCGM XID/ECC and SM/Tensor-core metrics, telling a GPU fault from a model regression, and guardrail latency measurement.

**Resources**
- Exam reading: NVIDIA blog *Measuring the Effectiveness and Performance of AI Guardrails in Generative AI Applications* (30 min)

**Exercise** (about 2.5 h): build a Grafana dashboard with 6 panels (Prometheus from NV-GENL.6, plus `dcgm-exporter` if you have a GPU) and 2 alert rules (p95 TTFT, queue time). Add a CI job that runs perf_analyzer or `trtllm-bench` plus the M5.9 eval, and fails on a regression of more than 10%.
- [ ] Done when a deliberately slowed config trips the gate.

**Self-check**
- [ ] Explain 9.5 "uptime, transparency and trust" as three concrete mechanisms.

### NV-GENL.9 Exam readiness and final week

**4 h** · prereqs: NV-GENL.0–.8 · Tested by: PGENL (all)

**Skip test**: none. Always do this module.
- [ ] Practice plan (2 h), see [Practice, readiness and exam day](#practice-readiness-and-exam-day).
- [ ] Final-week review (2 h): re-run every module's skip test cold; redo any you miss. Make one-page sheets for: the precision×GPU table, the parallelism arithmetic, the Triton config fields, the TensorRT-LLM feature→metric map, and the domain 9 task text.

### NV-GENL.10 Capstone: re-platform the ML-core capstone on NVIDIA

**12 h** · prereqs: NV-GENL.1–.8, M6.10, M5.7 · Tested by: PGENL 4–9 (integrative)

**Skip test**: none. The capstone is the proof of the module skills.

Take your **M6.10** project and swap its model for the **M5.7** fine-tuned LLM. Then rebuild the training and serving half on NVIDIA. Budget a rented L4/L40S (or H100 for FP8 timing) at about $10–30 [estimate]. Check the TensorRT-LLM support matrix for your GPU first.

1. [ ] Train a LoRA SFT with NeMo AutoModel, logged and registered in MLflow (M6.2). (2 h)
2. [ ] Quantize with Model Optimizer: FP8 or INT4-AWQ with a calibration set. Record the accuracy delta (NV-GENL.4). (2 h)
3. [ ] Serve with TensorRT-LLM behind Dynamo-Triton in Docker, with an ensemble for pre/post-processing and dynamic batching (NV-GENL.5–.6). (3 h)
4. [ ] Benchmark against your C10.4 vLLM baseline: TTFT, ITL, throughput and cost per 1M tokens (C10.5). (2 h)
5. [ ] Observe: Prometheus and DCGM dashboard, one `nsys` trace of the serving path, and the CI regression gate (NV-GENL.7–.8). (2 h)
6. [ ] Write a 1-page ADR: chosen precision, parallelism and serving stack, and what you would change on 8×H100. (1 h)

**Done when** a single `make bench` reproduces the comparison table, and the ADR names the binding constraint (memory, compute or queue).

## Coverage matrix

C = AI core, M = ML core, NV = this path. "Ex" names the module whose exercise proves the task. Self-checks live in the same modules.

| Task | Modules | Ex |
|---|---|---|
| 1.1 encoder-decoder | C1.2, M5.1, M5.5 | M5.5 |
| 1.2 transformer/self-attention | C1.2, M5.1 | M5.1 |
| 1.3 extract embeddings (enc + dec) | M5.4, M5.8 | NV-GENL.0 refresh script |
| 1.4 implement advanced sampling | C1.3, M5.1, NV-GENL.5 (beam) | NV-GENL.5 |
| 1.5 decoder output sampling | C1.3 | C1.3 |
| 1.6 embeddings concept | C1.2, C7.3 | C7.3 |
| 2.1 prompts, CoT, prompt learning | C2.1, C2.2, NV-GENL.3 | NV-GENL.3 |
| 2.2 zero/one/few-shot | C2.1 | C2.1 |
| 2.3 causal-LM training | M5.1, M5.3, M5.5 | M5.3 |
| 2.4 validation + constrained decoding | C2.3, C1.3, NV-GENL.5 | NV-GENL.5 |
| 3.1 clean/curate, imbalance | M3.1, M3.2, M3.4, NV-AMAD.1 (Curator) | M3.4 |
| 3.2 dataset formats | C11.3, M5.7, NV-GENL.3 | NV-GENL.3 |
| 3.3 tokenizers, vocab size | M5.2 | M5.2 |
| 4.1 pruning, sparsity, quantization | M7.2, NV-GENL.4 | NV-GENL.4 |
| 4.2 PTQ/QAT per hardware | M7.2, NV-GENL.1, NV-GENL.4 | NV-GENL.4 |
| 4.3 knowledge distillation | M7.2, C11.2, NV-GENL.4 | M7.2 |
| 4.4 HPO, LR schedules, batch size | M4.3, M4.4, M7.4, NV-GENL.2 | M4.4 |
| 4.5 beam/temperature + ablations | C1.3, M4.4, NV-GENL.5 | NV-GENL.5 |
| 4.6 TensorRT, sliding-window, KV cache | C10.1, NV-GENL.4, NV-GENL.5 | NV-GENL.5 |
| 4.7 MLM/NSP + compression concepts | M5.5, NV-GENL.3, M7.2 | M5.5 |
| 5.1 SFT/RLHF/DPO/GRPO | C11.2, M5.7, NV-GENL.3 | NV-GENL.3 |
| 5.2 contrastive, LoRA/adapters/P-tuning | M5.8, M5.7, NV-GENL.3 | M5.8 |
| 5.3 early stopping, metrics | M4.3, M5.7 | M5.7 |
| 5.4 hallucination, FT impact, PEFT | C1.1, C11.1, NV-GENL.3 | NV-GENL.3 |
| 6.1 benchmarks, HITL, judge, BLEU/ROUGE/PPL | M5.9, C3.2, C8.1 | M5.9 |
| 6.2 failure modes, error analysis | C3.1 | C3.1 |
| 6.3 benchmark across platforms | C10.4, NV-GENL.5, NV-GENL.7 | NV-GENL.10 |
| 6.4 eval framework | C3.3, C8.2, M5.9, NV-AMAD.1 (Evaluator) | NV-GENL.8 |
| 7.1 multi-GPU parallelism config | M7.3, NV-GENL.2 | NV-GENL.2 |
| 7.2 Tensor Cores, mixed precision, memory | M7.2, NV-GENL.1, NV-GENL.7 | NV-GENL.7 |
| 7.3 attention GEMM distribution, grad accumulation | M7.3, M5.3, NV-GENL.2, NV-GENL.7 | NV-GENL.2 |
| 7.4 CUDA profiling, memory/kernel issues | M7.1, NV-GENL.7 | NV-GENL.7 |
| 8.1 compute trade-offs per architecture | C10.1, C1.2 | NV-GENL.0 refresh |
| 8.2 containers, dynamic batching, Dynamo-Triton | M6.5, NV-GENL.6 | NV-GENL.6 |
| 8.3 K8s, ensembles, live monitoring | M6.6, NV-GENL.6, NV-AMAD.1 (NIM Operator) | NV-GENL.6 |
| 9.1 dashboards, reliability metrics | C8.3, M6.8, NV-GENL.8 | NV-GENL.8 |
| 9.2 logs, anomalies, RCA | C8.3, NV-GENL.8 | NV-GENL.8 |
| 9.3 benchmark vs prior versions | C8.4, C8.2, NV-GENL.8 | NV-GENL.8 |
| 9.4 automated tuning, retraining, versioning | M6.7, M6.9, M6.2, C8.4 | M6.9 |
| 9.5 uptime, transparency, trust | C4.1, C8.3, M6.6, C9.5 | NV-GENL.8 |
| 10.1 RAI in deployment | C9.5 | C9.5 |
| 10.2 bias/fairness audit | M9.1, C9.5 | M9.1 |
| 10.3 monitoring production LLMs | C8.3, NV-GENL.8 | NV-GENL.8 |
| 10.4 bias detection/mitigation | M9.1 | M9.1 |
| 10.5 guardrails | C9.4, NV-AMAD.1, NV-GENL.8 | C9.4 |

## Schedule

| Module | h |
|---|---|
| NV-GENL.0 | 2 |
| NV-GENL.1 | 4 |
| NV-GENL.2 | 6 |
| NV-GENL.3 | 5 |
| NV-GENL.4 | 6 |
| NV-GENL.5 | 6 |
| NV-GENL.6 | 6 |
| NV-GENL.7 | 6 |
| NV-GENL.8 | 3 |
| NV-GENL.9 | 4 |
| NV-GENL.10 | 12 |
| **Total** | **60** |

The total excludes NV-AMAD.1 parts 1A–1E (7.5 h) if you have not done them and the optional paid DLI courses (+2 h Nsight, +8 h compression).

**Default schedule (about 6–8 h/week, 10 weeks)**

| Week | Modules | h |
|---|---|---|
| 1 | .0 + .1 | 6 |
| 2 | .2 | 6 |
| 3 | .3 | 5 |
| 4 | .4 | 6 |
| 5 | .5 | 6 |
| 6 | .6 | 6 |
| 7 | .7 | 6 |
| 8 | .8 + capstone steps 1–2 | 7 |
| 9 | capstone steps 3–6 | 8 |
| 10 | .9 + exam | 4 |

**Compressed (about 15 h/week, 4 weeks)**

| Week | Modules |
|---|---|
| 1 | .0–.3 (17 h) |
| 2 | .4–.6 (18 h) |
| 3 | .7, .8, capstone steps 1–3 (16 h) |
| 4 | capstone steps 4–6, .9, exam (9 h) |

Book the exam at the start of week 8 (or week 3 if compressed). It can be booked up to 60 days ahead [hub].

## Practice, readiness and exam day

**Practice strategy.** There is no official NCP-GENL practice set (as of 2026-10-31).
1. Learn NVIDIA's question style for free first. If you did NV-AMAD, you already met its 16 official practice questions. If not, use the 50 official NCA-AIIO questions (Learning Mode on the [AIIO page][aiio]). They are scenario-style 4-option items with rationales.
2. For each of the 45 task statements, write 2 scenario questions with Claude. Keep a question only after you have checked its answer against the doc page named in the module. This turns the doc reading into recall practice.
3. Drill the calculations: parallel layouts, gradient-accumulation steps, memory fit, KV-cache size, and precision support per GPU.

**Readiness signal to book:**
- [ ] Every skip test passes cold.
- [ ] The capstone's `make bench` runs.
- [ ] You score at least 85% on a fresh 60-question set from your bank, written at least 7 days earlier.

**Exam-day tactics:**
- 120 min for 60–70 questions is about 1.7–2 min per question. Flag and move on after 2 min.
- The remote exam needs the secure browser, a 360° room check and ID checks, and allows no breaks [hub]. The exam page requires a Certiverse account; create it before booking.
- Results arrive on screen and by email within 24 h. The badge is issued via Credly [hub].

**Traps (first-hand candidate reports: none collected; the research web budget ran out, so these are document-based):**
- The guide recommends retired DLI courses. Don't hunt for them.
- Product names move:
  - Triton → Dynamo-Triton
  - TensorRT Model Optimizer → NVIDIA Model Optimizer
  - The NVIDIA/NeMo GitHub repo now redirects to NVIDIA-NeMo/Speech
  - Answer in whichever name the question uses.
- Products changed after the Sep 2025 guide: TensorRT-LLM dropped its TensorRT engine backend (PyTorch only, no `trtllm-build`), and TensorRT 11 removed implicit INT8 quantization. Questions may still assume the old workflow; answer the concept.
- About 31% of the exam is Model Optimization plus GPU work. Those are the domains furthest from app engineering, so weight study time there.
- Domain 9 says "agents" in an LLM exam. It is NCP-AAI's text reused.
- Never use dumps. NVIDIA runs response-pattern forensics and can revoke credentials [hub].

## Short note: NCA-AIIO

**NVIDIA-Certified Associate: AI Infrastructure and Operations.** Facts as of 2026-10-31 [aiio], [hub]:

| Item | Value |
|---|---|
| Price | $125 |
| Format | 50 questions, 60 min, remote |
| Prerequisite | "basic understanding of data center infrastructure" |
| Validity | 2 years |
| Blueprint (Jan 2026 guide) | Essential AI knowledge 38% · AI Infrastructure 40% · AI Operations 22% |

**Verdict: skip, unless** you want GPU and data-center literacy on paper or your job touches cluster operations. It teaches no ML-engineering skill.

**Cost:** $125, plus optionally the official self-paced course *AI Infrastructure and Operations Fundamentals* ($50, about 7 h). A free set of 50 official practice questions is on the exam page.

**Extra study beyond the cores and this file** (about 14 h [estimate]):
- Already covered here:
  - NV-GENL.1: GPU vs CPU, Tensor Cores, MIG/virtualisation (1.8, 3.4)
  - NV-GENL.7: DCGM and GPU monitoring (3.3)
  - NV-AMAD.1: NVIDIA software stack (1.1, 1.6)
  - M7.3: training vs inference requirements (1.2)
- New: domain 2 (40%), which covers power and cooling, facility requirements, InfiniBand vs Ethernet/RoCE, DPUs, on-prem vs cloud and cluster components. Domain 3 adds Slurm/K8s scheduling.
- Do the course units the guide maps: 1, 2, 4, 5, 7 (incl. 7.1, 7.4), 8, 10/11, 12.4, 13 and 14. Units 1–2 (AI basics, use cases) can be skimmed after the cores. Then do all 50 practice questions twice.

**Trap:** two forum candidates (Jun–Jul 2026) passed every course check and the practice set, yet found the exam "significantly more challenging than expected" [forum]. The same thread suggests a dump site; do not use it.

## Combining with other paths

- **Order inside NVIDIA:** NV-AMAD.1 → NV-GENL, with [NV-ADS](./nca-ads-ncp-ads.md) optional after it. NV-ADS reuses NV-GENL.6 (Triton) and NV-GENL.7 (profiling) instead of re-teaching them.
- **With NV-AAI:** domain 9 here equals NV-AAI's domain 8, so study it once.
- **With [AWS-MLA](../aws/mla-c02.md), [GCP-PMLE](../google/professional-ml-engineer.md), [AZ-AI300](../azure/ai-300.md), [DBX-MLP](../databricks/ml-professional.md):** about zero content overlap beyond the shared M-module refresh. Those exams test managed training and serving services; this one tests the GPU layer underneath them. NV-GENL.7's profiling skills transfer directly to PMLE 3.3 (→ M7.1).
- **Best pairing:** GCP-PMLE or AWS-MLA for platform MLOps, plus NV-GENL for model-optimization depth.

## Skipped on purpose

| Item | Reason |
|---|---|
| DLI *Model Parallelism* ILT ($500) | Not in the Oct 2026 catalog; M7.3 + NV-GENL.2 cover it free |
| DLI *Deploying RAG Pipelines for Production at Scale* ($500 ILT; $90 intro) | RAG → C7; deployment → NV-GENL.6; Helm/NIM → NV-AMAD.1 |
| DLI *Adding New Knowledge to LLMs* ($90) | Duplicates M5.7 + C11; learners report lab-time and environment failures |
| DLI *Building RAG Agents With LLMs* ($90), *Rapid Application Development With LLMs* ($90) | Duplicate C5/C7 and M5.4 |
| *Building Transformer-Based NLP Applications*, *Building LLM Applications With Prompt Engineering* | Both absent from the Oct 2026 catalog (learner-reported retirement); only the prompt-engineering one was reissued as open source. Covered by M5.5 and C2.1 |
| NVDLI *Fundamentals of Deep Learning* (open source) | M4 covers it; no owner in this repo |
| Guide readings: Illustrated Transformer, Goodfellow ch. 7, BLEU/ROUGE Medium explainer, PTQ-vs-QAT blog posts | Concepts owned by C1.2/M5.1, M2.3/M4.3, C3.2/M5.9, M7.2 |
| Guide readings for domain 9 (BERT, GPT-1, Attention papers) | Do not match the domain; architecture is covered by M5.1 |
| Ragas NVIDIA-metrics page | Ragas vocabulary is owned by C7.5 |
| CUDA C++ Best Practices Guide (whole) | C++ kernel work is not needed for a multiple-choice exam; APOD and profiling come from NV-GENL.7 |
| Unofficial "practice exams" / dumps | Forensics and revocation policy [hub] |

## Gaps for the core

- Beam search, P-tuning and NSP are taught here only because NVIDIA is the only vendor testing them. Move them to C1.3 / C11.2 / M5.5 if another vendor adds them.
- Statistically sound ablation design (4.5) relies on M4.4 + M1.3. Neither has a dedicated ablation-design exercise.

## Sources

All accessed 2026-10-31.

- [p] NCP-GENL exam page: https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/
- [g] NCP-GENL study guide (Sep 2025): https://dam-cdn.nvd.orangelogic.com/AssetLink/a38r3688v26wf6u414c64805b8i6awa3.pdf
- [hub] Certification hub + FAQ: https://www.nvidia.com/en-us/learn/certification/
- [cat] DLI course catalog, Oct 2026: https://dam-cdn.nvd.orangelogic.com/AssetLink/f14exk056qm0g3x3550ibl0xi88880r6.pdf
- [web] Certification roadmap webinar: https://www.nvidia.com/en-us/events/webinars/nvidia-certification-roadmap-ai-stack/
- [credly] https://www.credly.com/org/nvidia/badge/nvidia-certified-professional-gen-ai-llms
- [aiio] NCA-AIIO page: https://www.nvidia.com/en-us/learn/certification/ai-infrastructure-operations-associate/ ; guide: https://dam-cdn.nvd.orangelogic.com/AssetLink/x874j05hy3m3r2sor84kpvp70750m468.pdf ; course: https://www.nvidia.com/en-us/training/academy/course-detail/?id=course%3A15139841
- [forum] https://forums.developer.nvidia.com/t/374226
- GPU Performance Background: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
- Matrix Multiplication Background: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
- MIG User Guide: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/
- TensorRT support matrix (precision per GPU): https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
- TensorRT capabilities (types and precision): https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/capabilities.html
- TensorRT quantized types: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html
- TensorRT 11.0.0 release notes (implicit quantization removed): https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html
- TensorRT performance benchmarking: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html
- Megatron Bridge parallelisms: https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- NeMo AutoModel: https://docs.nvidia.com/nemo/automodel/latest/ ; SFT & PEFT: https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/sft-peft ; distributed setup: https://docs.nvidia.com/nemo/automodel/latest/development/distributed-setup
- NeMo RL: https://docs.nvidia.com/nemo/rl/latest/ ; DPO: https://docs.nvidia.com/nemo/rl/latest/guides/dpo.html ; GRPO: https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html
- NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer (README notes the 2025-12-08 rebrand) ; docs https://nvidia.github.io/Model-Optimizer/
- TensorRT-LLM docs: https://nvidia.github.io/TensorRT-LLM/ ; quantization: https://nvidia.github.io/TensorRT-LLM/features/quantization.html ; IFB/scheduling: https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html ; KV cache: https://nvidia.github.io/TensorRT-LLM/features/kvcache.html ; speculative decoding: https://nvidia.github.io/TensorRT-LLM/features/speculative-decoding.html ; sampling: https://nvidia.github.io/TensorRT-LLM/features/sampling.html ; guided decoding: https://nvidia.github.io/TensorRT-LLM/features/guided-decoding.html ; LLM API: https://nvidia.github.io/TensorRT-LLM/llm-api/index.html ; trtllm-serve: https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/index.html ; trtllm-bench: https://nvidia.github.io/TensorRT-LLM/commands/trtllm-bench.html ; backend removal: https://nvidia.github.io/TensorRT-LLM/legacy/tensorrt-backend-removal.html
- Triton user guide: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html
- perf_analyzer: https://github.com/triton-inference-server/perf_analyzer ; Model Analyzer: https://github.com/triton-inference-server/model_analyzer
- Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- DCGM: https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html
- DLProf (final release): https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/
- Blogs: https://developer.nvidia.com/blog/mastering-llm-techniques-training/ · https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/ · https://developer.nvidia.com/blog/assess-parallelize-optimize-deploy/ · https://developer.nvidia.com/blog/train-a-reasoning-capable-llm-in-one-weekend-with-nvidia-nemo/ · https://developer.nvidia.com/blog/measuring-the-effectiveness-and-performance-of-ai-guardrails-in-generative-ai-applications/
- DLI Nsight course: https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-03+V2

[p]: https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/
[g]: https://dam-cdn.nvd.orangelogic.com/AssetLink/a38r3688v26wf6u414c64805b8i6awa3.pdf
[hub]: https://www.nvidia.com/en-us/learn/certification/
[cat]: https://dam-cdn.nvd.orangelogic.com/AssetLink/f14exk056qm0g3x3550ibl0xi88880r6.pdf
[web]: https://www.nvidia.com/en-us/events/webinars/nvidia-certification-roadmap-ai-stack/
[credly]: https://www.credly.com/org/nvidia/badge/nvidia-certified-professional-gen-ai-llms
[aiio]: https://www.nvidia.com/en-us/learn/certification/ai-infrastructure-operations-associate/
[forum]: https://forums.developer.nvidia.com/t/374226
