# NVIDIA Accelerated Data Science path: NCA-ADS + NCP-ADS delta

**Who this is for / prerequisites.** You have finished the [ML core](../../core-ml-engineering.md) (M0–M10, including M3.6 Spark) and the [AI core](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/core-llm-app-engineering.md). You want NVIDIA's GPU data-science credentials: RAPIDS (cuDF, cuML, cuGraph), Dask multi-GPU, RAPIDS Accelerator for Spark and GPU XGBoost. **This is the optional, lowest-priority ML path.** It adds GPU tooling to classic data science you already know. Most of the blueprint is a light refresh of M2, M3, M6 and M8.

**No NVIDIA AI-path prerequisite:** NV-ADS.1 covers the NGC RAPIDS container itself. (If you took the NVIDIA AI path, NV-AMAD.1 part 1A is useful background, not a requirement.) The NCP-ADS delta also reuses two modules from [NV-GENL](./ncp-genl.md): NV-GENL.6 (Triton) and NV-GENL.7 (profiling). If you have not done NV-GENL, do the parts named in NV-ADS.10.

Exercises are in Python. GPU work runs on the free Colab or Kaggle GPU tier unless a module says otherwise. Facts are as of 2026-10-31.

## Contents

- [Verdict and cost](#verdict-and-cost)
- [Exam at a glance](#exam-at-a-glance)
- [NV-ADS.0 Core refresh: already covered](#nv-ads0-core-refresh-already-covered)
- [NCA-ADS modules NV-ADS.1–NV-ADS.8](#nca-ads-modules)
- [NCP-ADS delta NV-ADS.9–NV-ADS.11](#ncp-ads-delta)
- [Coverage matrix](#coverage-matrix)
- [Schedule](#schedule)
- [Practice, readiness and exam day](#practice-readiness-and-exam-day)
- [Combining with other paths](#combining-with-other-paths)
- [Skipped on purpose](#skipped-on-purpose)
- [Gaps for the core](#gaps-for-the-core)
- [Sources](#sources)

## Verdict and cost

| | NCA-ADS (Associate) | NCP-ADS (Professional) |
|---|---|---|
| Verdict | **Only if** you process large tabular data on GPUs or want a cheap NVIDIA badge after the ML core | **Only if** NCA-ADS was easy and your employer values it. Its guide is older (Mar 2025). |
| Exam cost | $125 | $200 |
| Paid prep needed | None. The free RAPIDS docs replace the courses. | None |
| GPU cost | $0 (Colab or Kaggle free tier) | $0–10 (Kaggle 2×T4 or a short multi-GPU rental [estimate]) |
| Hours beyond cores | 42 | +18 (+4 if you skip NV-GENL) |

## Exam at a glance

All facts below are as of 2026-10-31.

| Item | NCA-ADS | NCP-ADS | Source |
|---|---|---|---|
| Level | Associate ("entry-level") | Professional ("intermediate-level") | [a], [pp] |
| Status | Active. Credly template updated 2026-03-12, so a 2026 launch is likely (**unconfirmed**) | Active | [a], [pp], research |
| Questions / time | 50–60 / 60 min | 60–70 / 120 min | [a], [pp] |
| Delivery | Online, remotely proctored, Certiverse account | Same | [a], [pp], [hub] |
| Language | English on the page; "Multiple" in the Oct 2026 catalog (**discrepancy**) | Same | [a], [pp], [cat] |
| Prerequisites (recommended) | 1–2 years of accelerated data science | 2–3 years, plus a "strong foundation in ML and GPU-accelerated computing" | [a], [pp] |
| Passing score | **Not published.** Pass/fail, no score shown. | Same | [hub] |
| Validity / retakes | 2 years, renew by retaking. 14-day wait, max 5 attempts per 12 months, each paid. | Same | [hub] |
| Study guide | Feb 2026 [ga] | Mar 2025 [gp] | |
| Official practice set | **None found** | **None found** | [hub], research |

**NCA-ADS weights** [ga]:
- Data Manipulation & Preparation 23%
- ML with RAPIDS 16%
- Pipelines & Workflow Automation 13%
- Descriptive Analysis & Visualization 13%
- Foundations 12%
- Introductory MLOps 10%
- Advanced Data Structures 7%
- Software & Environment Management 6%

**NCP-ADS weights** [gp]:
- Data Analysis 14%
- Data Manipulation & Software Literacy 19%
- Data Preparation 17%
- GPU & Cloud Computing 16%
- Machine Learning 15%
- MLOps 19%

The NCA-ADS guide lists topics without numbers, so labels like A1.c below are ours (domain 1, third bullet). NCP-ADS task numbers are the guide's own.

## NV-ADS.0 Core refresh: already covered

**2 h** · prereqs: ML core · Tested by: ADS, PADS

**Skip test**: none. The refresh table below is the test: any row you can answer cold in under 2 min is done.

- [ ] Read both guides' topic and task lists (30 min).
- [ ] Do the refresh actions below (90 min). If a refresh takes more than 15 min, redo that core module's exercise instead.

| Core module | ADS topics / PADS tasks | Light refresh action |
|---|---|---|
| M0.1 | A1.a pandas joins, A4.b–c plots, A5.a Python stack; P1.6 | Rebuild one EDA figure set: histogram, box plot, scatter matrix, line plot. Say which plot answers which question. |
| M0.2 | A8.a–b env files, pip/Docker | Re-create your uv environment from its lockfile |
| M1.3 | A4.d hypothesis tests | Run a t-test and a chi-square test on your M2.1 data. Interpret the p-values. |
| M2.1, M2.6 | A3.a pipeline design, A4.a EDA, A5.d workflow; P1.5, P5.1 | Re-run the M2.6 pipeline end to end |
| M2.2 | A2.f confusion matrix, metrics; A2.c | Recompute precision, recall and F1 from a confusion matrix by hand |
| M2.3 | A2.e cross-validation, A3.c / A5.f under- and overfitting | Explain learning curves for high bias vs high variance |
| M2.2–M2.5 | A2.b regression, classification, clustering; A1.f dimensionality reduction | List one CPU estimator per task type. You will swap it for a GPU version in NV-ADS.3. |
| M3.1 | A1.e imbalance and synthetic data, A3.d augmentation; P3.4 | List the resampling and cost-sensitive options |
| M3.2 | A1.d feature engineering, A3.b selection; P3.2, P3.3, P5.2 scaling | When does scaling matter (kNN, linear models) and when not (trees)? |
| M3.3 | A1.g Parquet; P3.6 | Explain row vs columnar formats and predicate pushdown |
| M3.4, C9.5 | A1.b cleaning, quality, governance | Re-run your M3.4 data tests |
| M3.6 | A1.c Spark ETL; P2.3 | Re-read your M3.6 pipeline; you will GPU-accelerate it in NV-ADS.4 |
| M4.4 | A2.d, A5.f hyperparameter tuning; P5.5 | Re-run 10 Optuna trials |
| M6.2 | A6.b–c, A6.e tracking, save/load, artifacts | Register a model in MLflow, then load it by alias |
| M6.4, M6.9 | A3.e automation, A6.a pipeline monitoring | Re-run your orchestrated M6.4 flow |
| M6.5, M6.8 | A6.d drift; P6.5 deploy and monitor | Re-run the M6.8 Evidently report |
| M6.7, C8.4 | A3.g data flywheel | Explain feedback → labels → retrain → gated release in 3 lines |
| M8.2 | A7.a time series; P1.1, P1.2, P1.6 | Re-run the M8.2 backtest and anomaly cell |
| M7.2, M7.3, M4.3 | P5.6 (DL part), P5.7 batching and mixed precision | Explain mixed precision and DDP in 2 lines each |
| Learner skill | A8.e Git basics | Nothing to study |

<a id="nca-ads-modules"></a>
## NCA-ADS modules

### NV-ADS.1 GPU data-science environment and when GPUs pay

**4 h** · prereqs: M0.2, M7.1 · Tested by: ADS A5.b–c, A5.e, A6.f, A8.a–d; PADS 4.4

**Skip test** (passing means you skip the reading):
1. What minimum GPU compute capability and which CUDA majors does current RAPIDS support?
2. `nvidia-smi` shows "CUDA Version 12.8". Is that the installed toolkit?
3. A 10k-row groupby is slower on the GPU than with pandas. Why?
4. Which RAPIDS install methods exist, and what is the WSL2 limitation?

**Concepts**
- RAPIDS requirements [inst]:
  - Compute capability 7.0+ (Volta or newer)
  - CUDA 12 (driver ≥525.60.13) or CUDA 13 (driver ≥580.65.06)
  - Linux glibc ≥2.28, or WSL2 (single GPU only)
- Install methods: conda via Miniforge and conda-forge; pip wheels matching the system CUDA (`-cu12`/`-cu13`); Docker with nvidia-container-toolkit (or the NGC RAPIDS container).
- Environment files: `environment.yml` vs a uv lockfile. Pin RAPIDS, CUDA and Python together. Mixing pip and conda causes most conflicts.
- Environment checks:
  - `nvidia-smi`: shows the driver and the **maximum** CUDA version that driver supports, not the installed toolkit
  - `CUDA_VISIBLE_DEVICES`
  - a one-line cuDF smoke test
- Multi-user GPUs: MIG and time-slicing (→ NV-GENL.1).
- When the GPU pays: throughput-oriented hardware, host↔device copies over PCIe, kernel-launch overhead on small data, and keeping data resident on the device. Measure the crossover point; don't assume it.

**Resources**
- RAPIDS installation guide: System Requirements, Conda, pip, Docker, WSL2 (45 min)

**Exercise** (about 3 h): ship `environment.yml`, a `Dockerfile` and `envcheck.py`. The check prints driver, CUDA and RAPIDS versions and visible devices, then runs a cuDF op. Also write `crossover.py`: pandas vs cuDF groupby and join at 10⁴, 10⁶ and 10⁸ rows, including transfer time, plotted.
- [ ] Done when `envcheck.py` passes in Colab and in your container, and the plot shows the crossover point.

**Self-check**
- [ ] Explain to a teammate why their `pip install cudf-cu12` fails on a CUDA 11 driver.

### NV-ADS.2 cuDF and cudf.pandas

**6 h** · prereqs: NV-ADS.1, M3.2 · Tested by: ADS A1.a–b, A1.d, A1.g, A4.a; PADS 2.7, 3.1, 3.6, 4.5

**Skip test**
1. Name three ways to enable cudf.pandas.
2. What happens when an operation isn't supported?
3. How do you find which lines fell back to the CPU?
4. Why is `df.apply(row_fn, axis=1)` a GPU anti-pattern?

**Concepts**
- cudf.pandas [cdp]:
  - Enable with `%load_ext cudf.pandas`, `python -m cudf.pandas` or `cudf.pandas.install()`
  - Proxy objects with automatic CPU fallback
  - Profilers: `%%cudf.pandas.profile`, `%%cudf.pandas.line_profile`, `--profile`
- Native cuDF API differences: no row iteration, `.to_pandas()`, CuPy interop.
- Dtypes: category, narrower ints and floats, strings, decimals, timestamps. Measure with `memory_usage(deep=True)`.
- Joins, groupby, string ops.
- Parquet/ORC IO with column and filter pushdown.

**Resources**
- cuDF docs, cudf.pandas pages: Usage, How it Works, FAQ and Known Issues, Benchmarks (1.5 h)
- cuDF user guide, "10 Minutes to cuDF and Dask cuDF" (the NCP-ADS guide cites it as "10 Minutes to cuDF"; 45 min)

**Exercise** (about 3.5 h): run your M3.2 feature-engineering notebook under cudf.pandas and profile it. Rewrite the top two CPU-fallback hotspots in native cuDF. Add an `assert_frame_equal` parity test against pandas.
- [ ] Done when the profiler shows GPU execution for the hot path, parity passes, and the speed-up is recorded.

**Self-check**
- [ ] Pick dtypes for: user_id (int), country (200 values), price, event_ts.

### NV-ADS.3 cuML and GPU XGBoost

**6 h** · prereqs: NV-ADS.2, M2.4, M2.6, M4.4 · Tested by: ADS A1.f, A2.a–f, A5.f; PADS 5.4–5.6

**Skip test**
1. How do you run unchanged scikit-learn code on the GPU?
2. Which two XGBoost parameters move training to the GPU?
3. How do you run Optuna HPO with GPU trials without running out of memory?
4. Besides scikit-learn, which two libraries' estimators does `cuml.accel` accelerate?

**Concepts**
- `cuml.accel` [cma]: `python -m cuml.accel` or `%load_ext cuml.accel`. It accelerates supported scikit-learn, UMAP and HDBSCAN code and falls back to the CPU otherwise.
- Native cuML estimators: linear and logistic regression, RF, k-means, DBSCAN, PCA/TSVD, UMAP, nearest neighbours. `output_type` controls the result type (cuDF, CuPy or NumPy).
- XGBoost GPU [xgb]:
  - `device="cuda"`, `tree_method="hist"`
  - Native categorical support (`enable_categorical`)
  - GPU SHAP (explainability concepts → M9.2)
- HPO with GPU trials: one trial per GPU, memory per trial; Optuna itself → M4.4.
- Cross-validation with GPU estimators.

**Resources**
- cuML docs: "cuml.accel Overview", "Getting Started with cuml.accel", "Accelerated Estimator Support" and "Logging and Profiling" (1 h)
- XGBoost docs, "GPU Support" (20 min)

**Exercise** (about 4 h): take your M2.6 pipeline and M2.4 XGBoost model. Run them on the CPU, under `cuml.accel`, and with GPU XGBoost. Then run 50 Optuna trials on CPU vs GPU.
- [ ] Done when a table shows wall-clock time and metric per setup, and metrics agree within cross-validation noise.

**Self-check**
- [ ] Explain why GPU RF results can differ slightly from scikit-learn's and how you'd test acceptability.

### NV-ADS.4 Scale-out: Dask-cuDF, Dask-CUDA, Spark RAPIDS

**6 h** · prereqs: NV-ADS.2, NV-ADS.3, M3.6 · Tested by: ADS A1.c, A3.f, A5.e; PADS 1.4, 2.1, 2.3, 2.4, 2.6, 5.3

**Skip test**
1. What does `LocalCUDACluster` give you over a plain `LocalCluster`?
2. How do you switch a Dask DataFrame to the cuDF backend?
3. Which Spark config enables the RAPIDS plugin, and which tool tells you whether a job will benefit?
4. Data is 3× GPU memory on one 4-GPU node. Which library do you use?

**Concepts**
- Dask basics: lazy task graph, partitions, `persist()` vs `compute()`, dashboard.
- Dask DataFrame with the cuDF backend: `dask.config.set({"dataframe.backend": "cudf"})`.
- Dask-CUDA [dcu]:
  - One worker per GPU, CPU affinity
  - RMM pool, device→host spilling
  - UCX over NVLink/InfiniBand
- Multi-GPU XGBoost with `xgboost.dask`.
- RAPIDS Accelerator for Apache Spark (now titled "cuDF plugin for Apache Spark") [spk]:
  - `spark.plugins=com.nvidia.spark.SQLPlugin` plus the plugin jar, and `spark.rapids.sql.enabled`
  - GPU operators in `explain` output
  - Qualification and profiling tools
- Library choice by data size (PADS 1.4/2.6/5.3):
  - Fits one GPU: cudf.pandas
  - Exceeds one GPU on one node: Dask-cuDF
  - Multi-node with an existing Spark estate: Spark RAPIDS
  - Small or latency-bound: pandas

**Resources**
- Dask-CUDA docs: Quickstart, "Spilling from device", Best Practices (1 h)
- NVIDIA blog *Best Practices for Multi-GPU Data Analysis Using RAPIDS with Dask* (Nov 2024; 30 min)
- Spark RAPIDS user guide: Overview, Qualification overview, Profiling overview (1 h)
- XGBoost "Distributed XGBoost with Dask" tutorial (30 min)

**Exercise** (about 3 h): on Kaggle's 2-GPU T4 option ([verify] it is still offered) or a short rental, ETL multi-GB NYC TLC trip-record Parquet files with Dask-cuDF on `LocalCUDACluster`. Then train `xgboost.dask` on the result. Run it on 1 vs 2 GPUs and note the spill events. Optionally, point the Spark qualification tool at the event logs from your M3.6 job.
- [ ] Done when you have a scaling table and a one-paragraph library-choice ADR.

**Self-check**
- [ ] Explain why a shuffle-heavy join may scale badly from 1 to 2 GPUs.

### NV-ADS.5 Graph analytics with cuGraph and nx-cugraph

**4 h** · prereqs: NV-ADS.2 · Tested by: ADS A7.c–d; PADS 1.3, 4.1

**Skip test**
1. Edge list vs adjacency matrix vs CSR: which one does a GPU prefer?
2. PageRank vs betweenness: which costs more, and what does each say about "importance"?
3. How do you run existing NetworkX code on the GPU with no code change?
4. When is a graph database a better fit than relational tables?

**Concepts**
- Graph representations: edge list, adjacency list/matrix, CSR; directed, weighted and property graphs. Graph vs relational modelling.
- Centrality: degree, betweenness, eigenvector, PageRank (node importance).
- Communities: Louvain, Leiden. Also connected components and BFS/SSSP.
- nx-cugraph [nxc]: `NX_CUGRAPH_AUTOCONFIG=True` or `backend="cugraph"`. Needs Linux, compute capability 7.0+, CUDA 12.2+.
- Native cuGraph on cuDF edge lists.
- Visualising graphs: draw only top-k subgraphs.

**Resources**
- nx-cugraph README (whole) plus its demo Colab notebook (1 h)
- NetworkX "Tutorial" (an NCP-ADS reading; 45 min)

**Exercise** (about 2 h): load a public edge list with at least 1M edges (for example from SNAP). Run PageRank, betweenness (k-sampled) and Louvain with NetworkX on the CPU, then with nx-cugraph.
- [ ] Done when a timing table exists, the top-10 PageRank nodes match, and a top-community subgraph is plotted.

**Self-check**
- [ ] Explain which centrality you'd use to find "bridge" accounts in a fraud graph.

### NV-ADS.6 Accelerated time series

**3 h** · prereqs: NV-ADS.2, M8.2 · Tested by: ADS A7.a–b; PADS 1.1, 1.2, 1.6

**Skip test**
1. Interpolate irregular sensor timestamps onto a 1-minute grid in cuDF. Name the calls.
2. Why must time-series CV never shuffle?
3. Name one residual-based anomaly detector.

**Concepts**
- cuDF datetime handling: `resample`/`rolling`, reindexing onto a regular grid, `interpolate`.
- Time-based splits (→ M3.2, M8.2).
- Rolling z-score and residual anomaly detection (concepts → M8.2).
- GPU forecasting as the guide's reading shows it: skforecast with a cuML regressor (for example `RandomForestRegressor`) plugged in. cuML's own time-series module `cuml.tsa` (ARIMA, Holt-Winters) was deprecated in cuML 26.08 and is absent from the 26.10 API reference, so don't build on it.

**Resources**
- NVIDIA blog *Accelerating Time Series Forecasting With RAPIDS cuML* (Jan 2025, an NCA-ADS reading: skforecast + cuML random forest; 30 min)

**Exercise** (about 2.5 h): port your M8.2 data prep to cuDF. Inject gaps and irregular timestamps, interpolate them, and add a rolling-z anomaly detector.
- [ ] Done when the backtest metric matches the CPU version and anomaly precision@k is reported.

**Self-check**
- [ ] Choose plot types for: seasonality, an anomaly report, and forecast vs actual.

### NV-ADS.7 NCA-ADS readiness

**3 h** · prereqs: NV-ADS.0–.6 · Tested by: ADS (all)

**Skip test**: none. Always do this module.
- [ ] Practice plan (2 h), see [Practice, readiness and exam day](#practice-readiness-and-exam-day).
- [ ] Final review (1 h): re-run every skip test cold. Make a one-page sheet of activation commands (cudf.pandas, cuml.accel, nx-cugraph, Dask backend, Spark plugin), install and driver rules, and the library-by-data-size table.

### NV-ADS.8 Capstone: re-platform the ML-core capstone on RAPIDS

**8 h** · prereqs: NV-ADS.1–.6, M6.10 · Tested by: ADS (integrative)

**Skip test**: none. The capstone is the proof of the module skills.

Take your **M6.10** project. If its data is under about 1 GB, scale it up with a public multi-GB Parquet dataset such as NYC TLC trip records.

1. [ ] ETL and features with cudf.pandas, falling back to Dask-cuDF if the data doesn't fit (NV-ADS.2, NV-ADS.4). (2 h)
2. [ ] Train with GPU XGBoost and a `cuml.accel` baseline, with Optuna on the GPU, logged in MLflow (NV-ADS.3, M6.2). (2 h)
3. [ ] Batch-score the data, write Parquet, and run the M6.8 Evidently drift report on GPU-produced features. (1.5 h)
4. [ ] Benchmark CPU vs GPU across the whole pipeline: wall-clock, $ per run, and the environment captured in Docker (NV-ADS.1). (1.5 h)
5. [ ] Write a README covering how to reproduce it, the hardware choice and when the GPU isn't worth it. (1 h)

**Done when** `make all` runs the pipeline on Colab or Kaggle from a clean environment and prints the CPU/GPU comparison.

<a id="ncp-ads-delta"></a>
## NCP-ADS delta

This section is a delta on NV-ADS.1–.8. Only these extra skills are new.

### NV-ADS.9 GPU memory, dtypes, caching, benchmarking and CRISP-DM

**6 h** · prereqs: NV-ADS.4 · Tested by: PADS 2.2, 2.8, 3.5, 4.2, 4.3, 4.5, 4.6, 6.1–6.4, 5.7 (data side)

**Skip test**
1. Estimate the GPU memory for 500M rows × (int64, float64, 30-character string, 12-value category).
2. Pool allocator vs managed memory: which one enables oversubscription?
3. Two ways to cut shuffle in a repeated Dask join.
4. Name the six CRISP-DM phases in order.
5. List three mistakes that make GPU benchmarks lie.

**Concepts**
- RMM [rmm]: pool and async allocators, managed (unified) memory for oversubscription, statistics.
- Spilling: cuDF spilling and the Dask-CUDA `device_memory_limit` / `rmm_pool_size` knobs.
- Memory sizing: the in-memory footprint vs compressed Parquet size, joins needing working space, comparing required vs device memory (P6.2–6.3).
- Dtype choice per feature (P4.5/6.1): category, int32, float32, bool, datetime precision.
- Reducing shuffle: `persist()` reused frames, broadcast small tables, partition by join key; Spark `cache()`.
- Benchmark method:
  - Warm-up run, then synchronise before reading the clock (`.compute()`, stream sync)
  - Median of N runs, include transfers
  - Report rows/s and $ per run
- Bottleneck monitoring: Dask dashboard task stream (`nsys` → NV-GENL.7).
- **CRISP-DM**: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment, an iterative cycle. Map each phase to M10.1, M2.1, M3.x, M2–M4, M6.3 and M6.5. Only NCP-ADS tests it, so it is taught here.

**Resources**
- RMM docs: "User Guide" plus the `rmm.mr` (Memory Resources) and `rmm.statistics` Python API pages (1 h)
- NCP-ADS guide, domain 4 and 6 prose (15 min)

**Exercise** (about 4 h):
- `mem_budget.py` estimates the GPU footprint from a schema and row count, then checks it against the measured value.
- `bench.py` is a reusable harness with warm-up, sync and median timing.
- Downcast dtypes in your NV-ADS.8 data. Force an out-of-core join that OOMs, then make it succeed with spilling or managed memory.
- [ ] Done when estimates land within ±20% of measured values and a before/after report exists.

**Self-check**
- [ ] Walk a stakeholder through your capstone in CRISP-DM terms.

### NV-ADS.10 Multi-GPU training and inference at scale

**5 h** (+4 h if you skipped NV-GENL) · prereqs: NV-ADS.9 · Tested by: PADS 2.5, 2.9, 5.4, 5.6–5.8, 6.5

If you took NV-GENL, skip its two modules here. Otherwise do these parts of [ncp-genl.md](./ncp-genl.md) first:
- **NV-GENL.6**, only the Model Repository, Model Configuration and Batchers pages and the exercise with an ONNX model (about 2 h)
- **NV-GENL.7**, only the Nsight Systems part plus the DLProf note (DLProf is discontinued but still tested in PADS 2.5/5.8) (about 2 h)

**Skip test**
1. How do you batch-score 1B rows with a tree model across 4 GPUs?
2. What is FIL, and which model formats does Triton's FIL backend load?
3. How would you trade accuracy for inference speed in a gradient-boosted model?

**Concepts**
- Multi-GPU training: `xgboost.dask` and `cuml.dask` estimators. Deep-learning multi-GPU → M7.3; mixed precision and batching → M7.2.
- Batch inference over large data: `map_partitions` with a broadcast model; the Forest Inference Library (FIL). FIL now ships as the standalone **nvForest** library (`nvforest.load_model(...)`, Treelite-based); `cuml.fil` was removed in cuML 26.10, so older material that imports `cuml.fil.ForestInference` is out of date.
- Online serving: Triton FIL backend [fil] loads XGBoost, LightGBM, scikit-learn RF, cuML RF and Treelite models. Triton configuration itself → NV-GENL.6.
- Accuracy vs inference cost: number of trees, depth, and a precision/latency curve.
- Deploy and monitor → M6.5, M6.8.

**Resources**
- Triton FIL backend README (45 min)
- nvForest README (15 min)

**Exercise** (about 3.5 h): batch-score at least 100M rows with FIL on Dask-cuDF. Serve the same model through the Triton FIL backend with dynamic batching, and compare 3 model sizes.
- [ ] Done when a table shows batch rows/s, online p95 latency and accuracy per model size.

**Self-check**
- [ ] Defend single-node multi-GPU vs multi-node for a nightly 2 TB scoring job.

### NV-ADS.11 NCP-ADS readiness and capstone extension

**7 h** · prereqs: NV-ADS.9, NV-ADS.10 · Tested by: PADS (all)

**Skip test**: none. Always do this module.

- [ ] Practice (2 h): same method as NV-ADS.7, using the 40 NCP-ADS task statements.
- [ ] Capstone extension (5 h): run NV-ADS.8 on 2+ GPUs with spilling, serve the model with Triton FIL, add a time-series anomaly monitor on prediction volume (NV-ADS.6), and write the CRISP-DM write-up.
- [ ] Done when the extended `make all` reports multi-GPU scaling and the online latency SLO.

## Coverage matrix

C = AI core, M = ML core, NV = NVIDIA paths. The "Ex" column names the module whose exercise proves the item. Self-checks live in the same modules.

**NCA-ADS** (topic labels are ours)

| Topic | Modules | Ex |
|---|---|---|
| A1.a cuDF/pandas joins | M0.1, NV-ADS.2 | NV-ADS.2 |
| A1.b cleaning, governance | M3.4, C9.5, NV-ADS.2 | M3.4 |
| A1.c GPU ETL (RAPIDS/Dask/Spark) | M3.6, NV-ADS.4 | NV-ADS.4 |
| A1.d feature engineering | M3.2, NV-ADS.2 | NV-ADS.2 |
| A1.e imbalance, synthetic data | M3.1 | M3.1 |
| A1.f dimensionality reduction, sampling | M2.5, M3.1, NV-ADS.3 | NV-ADS.3 |
| A1.g Parquet | M3.3, NV-ADS.2 | NV-ADS.2 |
| A2.a cuML + XGBoost GPU | NV-ADS.3 | NV-ADS.3 |
| A2.b–c model types, evaluation | M2.2–M2.5, M6.3 | M2.2 |
| A2.d tuning | M4.4, NV-ADS.3 | NV-ADS.3 |
| A2.e cross-validation | M2.3, M2.6 | M2.6 |
| A2.f metrics, confusion matrix | M2.2 | M2.2 |
| A3.a–d pipeline design, features, fit, augmentation | M2.1, M2.6, M3.1–M3.2, M2.3 | M2.6 |
| A3.e automation, scalability | M6.4 | M6.4 |
| A3.f reproducible RAPIDS + Dask pipelines | NV-ADS.4, NV-ADS.8 | NV-ADS.8 |
| A3.g data flywheel | C8.4, M6.7 | M6.7 |
| A4.a–c EDA, visualisation, plot choice | M0.1, M2.1 | NV-ADS.0 refresh |
| A4.d hypothesis testing | M1.3 | NV-ADS.0 refresh |
| A4.e patterns and trends | M2.1, M8.2 | M8.2 |
| A5.a Python stack | M0.1 | M0.1 |
| A5.b–c GPU concepts, CPU vs GPU, transfers | NV-ADS.1, M7.1 | NV-ADS.1 |
| A5.d end-to-end workflow | M2.1, M3.3 | NV-ADS.8 |
| A5.e distributed vs GPU frameworks | NV-ADS.4, M7.4 | NV-ADS.4 |
| A5.f parameters, tuning, fit | M2.3, M4.4 | M4.4 |
| A6.a pipeline monitoring | M6.4, M6.8 | M6.8 |
| A6.b tracking (MLflow/W&B) | M6.2 | M6.2 |
| A6.c save/load/predict | M6.2, M6.5 | M6.2 |
| A6.d drift | M6.8 | NV-ADS.8 |
| A6.e artifacts, configs | M6.2, M6.9 | M6.9 |
| A6.f benchmarking, hardware choice | NV-ADS.1, NV-ADS.8 | NV-ADS.8 |
| A7.a time-series handling and evaluation | M8.2, NV-ADS.6 | NV-ADS.6 |
| A7.b irregular timestamps, cuDF interpolation | NV-ADS.6 | NV-ADS.6 |
| A7.c–d graphs, node importance, visualisation | NV-ADS.5 | NV-ADS.5 |
| A8.a–c env files, Conda/pip/Docker, dependencies | M0.2, NV-ADS.1 | NV-ADS.1 |
| A8.d GPU environment check, conflict fix | NV-ADS.1 | NV-ADS.1 |
| A8.e Git basics | learner skill | — |

**NCP-ADS**

| Task | Modules | Ex |
|---|---|---|
| 1.1 time-series anomalies | M8.2, NV-ADS.6 | NV-ADS.6 |
| 1.2 time-series analysis | M8.2, NV-ADS.6 | NV-ADS.6 |
| 1.3 graph data with cuGraph | NV-ADS.5 | NV-ADS.5 |
| 1.4 how big is big data | NV-ADS.4, NV-ADS.9 | NV-ADS.4 |
| 1.5 EDA | M2.1 | M2.1 |
| 1.6 time-series visualisation | M0.1, M8.2 | NV-ADS.6 |
| 2.1 accelerated ETL | NV-ADS.4 | NV-ADS.4 |
| 2.2 caching to reduce shuffle | NV-ADS.9 | NV-ADS.9 |
| 2.3 distributed processing | M3.6, NV-ADS.4 | NV-ADS.4 |
| 2.4 Dask multi-GPU | NV-ADS.4 | NV-ADS.4 |
| 2.5 DLProf profiling | NV-GENL.7, NV-ADS.10 | NV-GENL.7 |
| 2.6 library per data size | NV-ADS.4 | NV-ADS.4 |
| 2.7 cuDF transform/standardise | NV-ADS.2, M3.2 | NV-ADS.2 |
| 2.8 GPU memory management | NV-ADS.9 | NV-ADS.9 |
| 2.9 scaled inference | NV-ADS.10 | NV-ADS.10 |
| 3.1 cleansing with cuDF/pandas | NV-ADS.2, M3.4 | NV-ADS.2 |
| 3.2–3.3 transform, standardise | M3.2 | M3.2 |
| 3.4 identify and acquire datasets | M3.1, M3.3 | NV-ADS.8 |
| 3.5 pipeline bottlenecks | NV-ADS.9, M6.4 | NV-ADS.9 |
| 3.6 process, organise, store | M3.3, NV-ADS.2 | NV-ADS.2 |
| 4.1 graph analysis on GPU | NV-ADS.5 | NV-ADS.5 |
| 4.2 GPU optimisation | NV-ADS.9 | NV-ADS.9 |
| 4.3 CRISP-DM | NV-ADS.9 | NV-ADS.11 |
| 4.4 Docker/Conda conflicts | NV-ADS.1 | NV-ADS.1 |
| 4.5 dtype per feature | NV-ADS.9, NV-ADS.2 | NV-ADS.9 |
| 4.6 framework benchmark | NV-ADS.9 | NV-ADS.9 |
| 5.1 split choice | M2.3, M3.2, M8.2 | M2.6 |
| 5.2 standardise | M3.2 | M3.2 |
| 5.3 big data | NV-ADS.4 | NV-ADS.4 |
| 5.4 accuracy vs inference speed | NV-ADS.10, M4.4 | NV-ADS.10 |
| 5.5 HPO | M4.4, NV-ADS.3 | NV-ADS.3 |
| 5.6 single- and multi-GPU training | NV-ADS.3, NV-ADS.10, M7.3 | NV-ADS.10 |
| 5.7 batching, mixed precision | M7.2, M4.3, NV-ADS.9 | M7.2 |
| 5.8 DLProf | NV-GENL.7 | NV-GENL.7 |
| 6.1 dtype choice | NV-ADS.9 | NV-ADS.9 |
| 6.2–6.3 dataset memory vs device | NV-ADS.9 | NV-ADS.9 |
| 6.4 benchmark GPU workflows | NV-ADS.9 | NV-ADS.9 |
| 6.5 deploy and monitor | M6.5, M6.8, NV-ADS.10, NV-GENL.6 | NV-ADS.10 |

## Schedule

| Module | h |
|---|---|
| NV-ADS.0 | 2 |
| NV-ADS.1 | 4 |
| NV-ADS.2 | 6 |
| NV-ADS.3 | 6 |
| NV-ADS.4 | 6 |
| NV-ADS.5 | 4 |
| NV-ADS.6 | 3 |
| NV-ADS.7 | 3 |
| NV-ADS.8 | 8 |
| **NCA-ADS subtotal** | **42** |
| NV-ADS.9 | 6 |
| NV-ADS.10 | 5 |
| NV-ADS.11 | 7 |
| **NCP-ADS delta** | **18** (+4 without NV-GENL) |
| **Both** | **60** |

**Default schedule (about 7 h/week)**

| Week | Modules | h |
|---|---|---|
| 1 | .0 + .1 | 6 |
| 2 | .2 | 6 |
| 3 | .3 | 6 |
| 4 | .4 | 6 |
| 5 | .5 + .6 | 7 |
| 6 | .8 | 8 |
| 7 | .7 + NCA-ADS exam | 3 |
| 8 | .9 | 6 |
| 9 | .10 | 5 |
| 10 | .11 + NCP-ADS exam | 7 |

**Compressed (about 20 h/week, 3 weeks)**

| Week | Modules |
|---|---|
| 1 | .0–.3 (18 h) |
| 2 | .4, .5, .6, .8 (21 h) |
| 3 | .7 + NCA-ADS exam, then .9–.11 + NCP-ADS exam (21 h) |

Book NCA-ADS at the start of week 5; exams can be booked up to 60 days ahead [hub].

## Practice, readiness and exam day

**Practice strategy.** There is no official practice set for either exam (as of 2026-10-31). NVIDIA's webinar promises practice tests "for select associate-level exams" [web], so check the NCA-ADS page again before booking.
1. Use the guides' suggested readings as the source of question style: cuDF vs pandas, cuML zero-code change, Dask, class imbalance and monitoring posts.
2. For each topic or task, write 2 scenario questions with Claude. Keep a question only after you have checked its answer against the doc named in the module.
3. Drill the decisions: which library for which data size, which dtype, which plot, which centrality, which allocator.

**Readiness signal to book:**
- [ ] All skip tests pass cold.
- [ ] The capstone `make all` runs from a clean environment.
- [ ] You score at least 85% on a fresh 50-question set written at least 7 days earlier (60 questions for NCP-ADS).

**Exam-day tactics:**
- NCA-ADS allows about 1 min per question, so answer on first read and flag. NCP-ADS allows about 1.7–2 min.
- Remote rules: secure browser, room scan, no breaks [hub].

**Traps (document-based; no first-hand reports were collected):**
- The NCP-ADS guide (Mar 2025) cites DLProf, which is now discontinued, and RAPIDS 24.04. Answer the concept; don't chase versions.
- RAPIDS docs moved from docs.rapids.ai to docs.nvidia.com, the cudf repo moved to NVIDIA/cudf, and the Spark plugin is now titled "cuDF plugin for Apache Spark" (its docs URL redirects to docs.nvidia.com/cudf-spark). Old names may still appear in questions.
- cuML 26.10 (Oct 2026) removed `cuml.fil` (FIL now lives in nvForest) and cuML 26.08 deprecated `cuml.tsa`. Both guides predate this.
- About half of NCA-ADS is classic data science. Don't over-study RAPIDS APIs at the expense of the M-module refresh.
- Never use dumps [hub].

## Combining with other paths

- **Order inside NVIDIA:** NV-ADS has no NVIDIA prerequisite. [NV-GENL](./ncp-genl.md) is independent; if you take both, do NV-GENL first and NV-ADS.10 shrinks by 4 h.
- **With [DBX-MLA](../databricks/ml-associate.md) / [DBX-MLP](../databricks/ml-professional.md):** both build on M3.6 Spark. NV-ADS.4 adds only the GPU plugin, about zero overlap.
- **With [AWS-MLA](../aws/mla-c02.md) / [GCP-PMLE](../google/professional-ml-engineer.md) / [AZ-AI300](../azure/ai-300.md):** about zero overlap beyond the shared M refresh. RAPIDS appears in none of their blueprints. If you pick only one data-science badge, those cloud ML exams carry more market weight than ADS.

## Skipped on purpose

| Item | Reason |
|---|---|
| DLI *Accelerating End-to-End Data Science Workflows* (S-DS-01), *Feature Engineering for Tabular Data* (S-DS-06), *Time Series Forecasting* (S-DS-10) | Linked from the exam page but absent from the Oct 2026 catalog; price and availability **unconfirmed**. The free RAPIDS docs plus M modules cover them. |
| ILT *Fundamentals of Accelerated Data Science*, *Enhancing Data Science Outcomes With Efficient Workflow* | Instructor-led and paid; same material as above |
| DLI *Fundamentals of Accelerated Computing With CUDA Python* ($90) | Writing Numba kernels is not tested |
| Python Data Science Handbook, GeeksforGeeks tutorials | Covered by M0.1 / M2; no owner in this repo |
| Weights & Biases | MLflow is the repo's tracking tool (M6.2); know W&B only as an equivalent |
| Guide readings on Plotly Dash/RAPIDS viz, the *Big Data* book, anomaly-detection surveys, the FPP book | Visualisation → M0.1; anomaly detection and forecasting → M8.2 (FPP owned there) |
| *RAPIDS on Databricks*, *ETL on Kubeflow* readings | Platform-specific; the concepts are in NV-ADS.4 |
| Confidential-computing reading (NCP-ADS MLOps prose) | Not a task statement; see Gaps |
| Unofficial practice exams / dumps | Forensics and revocation policy [hub] |

## Gaps for the core

- Encryption and confidential computing for ML data and models (NCP-ADS MLOps prose): no core owner. Only the NVIDIA wording is noted here.
- Resolved 2026-10-31: tabular synthetic-data generators (NCA-ADS A1.e) are now a concept bullet in M3.1.

## Sources

All accessed 2026-10-31.

- [a] NCA-ADS page: https://www.nvidia.com/en-us/learn/certification/accelerated-data-science-associate/
- [ga] NCA-ADS study guide (Feb 2026): https://dam-cdn.nvd.orangelogic.com/AssetLink/328u55865k15h7h0r28e70o023andi0r.pdf
- [pp] NCP-ADS page: https://www.nvidia.com/en-us/learn/certification/accelerated-data-science-professional/
- [gp] NCP-ADS study guide (Mar 2025): https://dam-cdn.nvd.orangelogic.com/AssetLink/6ql1cb8q776an2012vu86kpjxrvysm3q.pdf
- [hub] Certification hub + FAQ: https://www.nvidia.com/en-us/learn/certification/
- [cat] DLI catalog, Oct 2026: https://dam-cdn.nvd.orangelogic.com/AssetLink/f14exk056qm0g3x3550ibl0xi88880r6.pdf
- [web] Certification roadmap webinar: https://www.nvidia.com/en-us/events/webinars/nvidia-certification-roadmap-ai-stack/
- [inst] RAPIDS install: https://docs.nvidia.com/datascience/install/
- [cdp] cudf.pandas: https://docs.nvidia.com/cudf/latest/cudf_pandas/ ; usage: https://docs.nvidia.com/cudf/latest/cudf_pandas/usage/ ; cuDF docs root: https://docs.nvidia.com/cudf/latest/ ; 10 Minutes to cuDF and Dask cuDF: https://docs.nvidia.com/cudf/latest/cudf/10min/
- [cma] cuml.accel: https://docs.nvidia.com/cuml/latest/cuml-accel/ ; cuML docs: https://docs.nvidia.com/cuml/latest/ ; cuML changelog (FIL removal, tsa deprecation): https://github.com/NVIDIA/cuml/blob/main/CHANGELOG.md
- [xgb] XGBoost GPU support: https://xgboost.readthedocs.io/en/stable/gpu/index.html ; Distributed XGBoost with Dask: https://xgboost.readthedocs.io/en/stable/tutorials/dask.html
- [dcu] Dask-CUDA: https://docs.nvidia.com/dask-cuda/latest/
- Multi-GPU best practices blog (Nov 2024): https://developer.nvidia.com/blog/best-practices-for-multi-gpu-data-analysis-using-rapids-with-dask/
- [spk] cuDF plugin for Apache Spark (RAPIDS Accelerator): https://docs.nvidia.com/spark-rapids/user-guide/latest/ (redirects to https://docs.nvidia.com/cudf-spark/latest/)
- [nxc] nx-cugraph: https://github.com/rapidsai/nx-cugraph
- NetworkX tutorial: https://networkx.org/documentation/stable/tutorial.html
- Time-series blog: https://developer.nvidia.com/blog/accelerating-time-series-forecasting-with-rapids-cuml/
- [rmm] RMM: https://docs.nvidia.com/rmm/latest/ ; user guide: https://docs.nvidia.com/rmm/latest/user_guide/guide/
- [fil] Triton FIL backend: https://github.com/triton-inference-server/fil_backend ; nvForest: https://github.com/rapidsai/nvforest
- DLProf final release: https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/

[a]: https://www.nvidia.com/en-us/learn/certification/accelerated-data-science-associate/
[ga]: https://dam-cdn.nvd.orangelogic.com/AssetLink/328u55865k15h7h0r28e70o023andi0r.pdf
[pp]: https://www.nvidia.com/en-us/learn/certification/accelerated-data-science-professional/
[gp]: https://dam-cdn.nvd.orangelogic.com/AssetLink/6ql1cb8q776an2012vu86kpjxrvysm3q.pdf
[hub]: https://www.nvidia.com/en-us/learn/certification/
[cat]: https://dam-cdn.nvd.orangelogic.com/AssetLink/f14exk056qm0g3x3550ibl0xi88880r6.pdf
[web]: https://www.nvidia.com/en-us/events/webinars/nvidia-certification-roadmap-ai-stack/
[inst]: https://docs.nvidia.com/datascience/install/
[cdp]: https://docs.nvidia.com/cudf/latest/cudf_pandas/
[cma]: https://docs.nvidia.com/cuml/latest/cuml-accel/
[xgb]: https://xgboost.readthedocs.io/en/stable/gpu/index.html
[dcu]: https://docs.nvidia.com/dask-cuda/latest/
[spk]: https://docs.nvidia.com/spark-rapids/user-guide/latest/
[nxc]: https://github.com/rapidsai/nx-cugraph
[rmm]: https://docs.nvidia.com/rmm/latest/
[fil]: https://github.com/triton-inference-server/fil_backend
