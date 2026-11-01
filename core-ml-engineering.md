# ML Engineering Core (vendor-neutral)

**Who this is for / prerequisites.** This guide is for a senior backend engineer who has finished the [AI engineering core](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/core-llm-app-engineering.md) (modules C0–C13). That means you can already call, evaluate, trace, secure and cost LLM applications, and you work daily in Claude Code. This core teaches the other half: building, training, shipping and operating models yourself. It covers classical ML, feature and data work, deep learning, transformer training, MLOps, performance, recommenders and forecasting, responsible ML and system design. All exercises are in Python. You need the C0.2 lab (uv, Docker, local MLflow server), a Google account for Colab, a Kaggle account, and roughly $10–50 for rented GPU time. Nothing in the AI core is taught again here. Every overlap is a one-line cross-reference to a `C` ID. After this core, each vendor ML path in `paths/` is a delta on top of it.

## Contents

- [At a glance](#at-a-glance)
- [Already covered in the AI core](#already-covered-in-the-ai-core)
- [How to use this guide](#how-to-use-this-guide)
- [Diagnostic (1 h)](#diagnostic-1-h)
- [M0 Setup](#m0-setup) · [M1 Math](#m1-math) · [M2 Classical ML](#m2-classical-ml) · [M3 Data and features](#m3-data-and-features) · [M4 Deep learning](#m4-deep-learning) · [M5 Transformers and LLM training](#m5-transformers-and-llm-training) · [M6 MLOps](#m6-mlops) · [M7 Performance and scale](#m7-performance-and-scale) · [M8 Recsys and time series](#m8-recsys-and-time-series) · [M9 Responsible ML](#m9-responsible-ml) · [M10 ML system design](#m10-ml-system-design)
- [Capstone projects](#capstone-projects)
- [Coverage matrix](#coverage-matrix)
- [Where this core leads (vendor ML paths)](#where-this-core-leads-vendor-ml-paths)
- [Schedule](#schedule)
- [Tool setup](#tool-setup)
- [Skipped on purpose](#skipped-on-purpose)
- [Sources](#sources)

## At a glance

| Item | Value |
|---|---|
| Modules | 52 (M0.1–M10.2); 2 are conditional (M3.6 Spark, M7.4 Ray) |
| Essential hours | **290 h** = 1 h diagnostic + 281 h modules (Capstone A included in M6.10) + 8 h for one more capstone (B, C or D) |
| Conditional | +10 h (M3.6 + M7.4), required only before the DBX-MLA, DBX-MLP or GCP-PMLE paths |
| Optional depth (my estimates) | Evidently course (+6 h), MIT 6.5940 (+6 h), nanochat (+10–20 h), the remaining Aminian & Xu cases (+8 h), the other capstones (+8 h each) |
| Default pace | 7 h/week → about 42 weeks; compressed 14 h/week → about 21 weeks |
| Paid items | Géron (2025), *DMLS*, *Hands-On LLMs* (ch10 notebook is free), Aminian & Xu (free fallback named). Check whether your employer gives you O'Reilly access first; it covers the first three |
| GPU spend | about $10 for the GPT-2 repro (M5.3), $0–40 for optional multi-GPU work (M7.3); everything else runs on free Colab or Kaggle |
| Certification | None vendor-neutral. The ML certs worth their time are vendor ones; see [Where this core leads](#where-this-core-leads-vendor-ml-paths) |

## Already covered in the AI core

You did these in the AI core. Do not study them again. The "light refresh" column is the most you should spend, and only at the moment the M module named in that column needs it.

| Topic | Owner | Light refresh action (when) |
|---|---|---|
| Lab: uv env, Docker, local MLflow 3 server, API-key hygiene | C0.2 | Reuse the same server and env. M0.2 only adds ML extras |
| Pretraining → SFT → RL vocabulary, hallucination, knowledge cutoff | C1.1 | None. M5.3 and M5.7 implement the stages |
| Attention, embeddings, softmax intuition (3B1B DL ch5–6) | C1.2 | Rewatch 3B1B ch6 at 1.5× right before M5.1 (20 min) |
| Sampling, reasoning budgets, context windows, tokenization pitfalls | C1.3–C1.4 | Reread your C1.4 pitfalls notes before M5.2 (10 min) |
| Model landscape: SLM, MoE, base vs instruct, VLM, embedding models | C1.5 | None |
| BLEU/ROUGE/BERTScore meaning, golden and synthetic datasets, pairwise eval | C3.2 | M5.9 computes the metrics. Meaning stays in C3.2 |
| Eval harness, pytest checks, CI eval gate | C3.3, C8.4 | M6.9 adds model-quality gates to the same GitHub Actions pattern |
| Model selection, public benchmarks vs app evals | C3.4 | M5.9 adds harness mechanics and contamination only |
| Backend call patterns: retries, queues, circuit breakers, batch pattern | C4.1 | M6.5 reuses them for model services |
| Embedding-model choice, ANN indexes, pgvector | C7.3 | M5.8 trains an embedder into your pgvector index; M8.1 reuses ANN |
| Hybrid retrieval, rerankers (cross-encoders) | C7.4 | M5.8 contrasts bi- and cross-encoders in one line |
| Ranking metrics recall@k, precision@k, MRR, nDCG | C7.5 | M8.1 applies them to recommendations |
| LLM-as-judge, agent evals | C8.1–C8.2 | None |
| Tracing, OTel GenAI, MLflow 3 GenAI tracing, feedback capture | C8.3 | M6.2 uses the classic tracking and registry side of the same server |
| LLM release engineering: canary/A-B for LLM apps, data flywheel, migration | C8.4 | M6.7 adds bandits, interleaving and continual learning for trained models |
| Prompt injection, OWASP LLM/Agentic, guardrails, red teaming | C9.1–C9.4 | None |
| RAI principles, NIST AI RMF / 600-1, EU AI Act, model cards, lineage | C9.5 | M9 adds fairness metrics and explainability only |
| Prefill/decode, KV cache, continuous batching, speculative decoding | C10.1 | M7.1 adds the general GPU performance model |
| LLM quantization formats (GPTQ, AWQ, GGUF), quality vs memory | C10.2 | M7.2 implements the algorithms (PTQ, QAT, calibration) |
| Prompt/semantic caching | C10.3 | None |
| vLLM, self-host sizing, load testing, OpenAI-compatible endpoints | C10.4 | M6.6 runs vLLM as a KServe backend (read only) |
| Cost modelling, token budgets | C10.5 | M7.3 reuses your spreadsheet for GPU-hour estimates |
| When (not) to fine-tune; catastrophic forgetting | C11.1 | Reread your C11.1 decision note before M5.7 (10 min) |
| Full FT vs PEFT, LoRA/QLoRA, DPO, KTO, GRPO, distillation (concepts) | C11.2 | M5.7 implements them with TRL |
| Synthetic data, distillation data, chat dataset formats | C11.3 | M5.7 and M5.8 reuse your C11.3 datasets |
| Your Unsloth QLoRA run | C11.4 | M5.7 redoes it with raw TRL + PEFT and compares |
| VLMs, CLIP-style multimodal embeddings (use) | C12.1 | M4.5 trains and fine-tunes vision models |
| Classic language tasks: LLM vs specialised model | C12.3 | M5.5 trains the specialised model and benchmarks it against your C12.3 LLM baseline |
| Diffusion intuition | C12.4 | Optional Géron ch18 section in M4.5 |
| Requirements, SLAs, ROI, pattern choice for LLM products | C13.1 | M10.1 adds ML-objective framing |

## How to use this guide

1. **Take the diagnostic** below. It tells you which skip tests to attempt first.
2. **Each module opens with a skip test.** It is calibrated to you: you are strong on software engineering, CI, containers and agentic tooling, and new to statistics, model training and MLOps. Pass it cleanly, without docs, and you skip the reading. You still do the exercise and the self-check, because exercises build the capstones.
3. **One lab repo.** Create `mlcore-lab` (private GitHub) with one package per block (`m2_classical/`, `m6_mlops/` …). Every exercise lands there with a test or a script that proves its "done when" criterion. Use Claude Code to scaffold and review code, but write the model code yourself. The point is that your hands learn it.
4. **Resources are assigned once.** When a module says "Géron ch4", that chapter is read only there. Section names are given so you do not read whole books.
5. **Phases.** Phase 1 = M0 → M1 (just in time) → M2 → M3.1–M3.4. Phase 2 = M4 → M5 (M5.4 before M5.6; M5.5 after M5.2). Phase 3 = M3.5 → M6.1–M6.10 → M7; it can start in parallel once M2 is done. Phase 4 = M8 → M9 → M10. M3.6 and M7.4 are needed only before DBX-MLA, DBX-MLP or GCP-PMLE.
6. **Checkboxes.** Tick `reading` (or `skipped via skip test`) and `exercise + self-check` per module.

Exam codes in "Tested by" lines: MLA = AWS MLA-C02, 300 = Microsoft AI-300, PMLE = Google Professional ML Engineer (June-2026 guide), DMA / DMP = Databricks ML Associate / Professional, PGENL = NVIDIA NCP-GENL, ADS / PADS = NVIDIA NCA-ADS / NCP-ADS, AGENL / GENM = NVIDIA NCA-GENL / NCA-GENM (short notes in the AI repo's NV-AMAD file), AIF/901 = AWS AIF-C01 / Microsoft AI-901 (light). Exam facts live only in the path files.

## Diagnostic (1 h)

Answer on paper, without docs, in 60 minutes. A confident and correct answer sends you to that module's skip test. A shaky answer means you do the module in full.

| # | Question | Routes to |
|---|---|---|
| 1 | Z-score every column of a (1e6, 20) NumPy array without loops, then compute a 7-day rolling mean per customer with pandas | M0.1 |
| 2 | What does an eigenvector of a covariance matrix represent, and why does PCA use it? | M1.1 |
| 3 | Derive dL/dw for L = (σ(wx+b) − y)² | M1.2 |
| 4 | Test 99 % sensitive, 95 % specific, prevalence 1 %: what is P(disease \| positive)? | M1.3 |
| 5 | 1 % positive class, model has 99 % accuracy. What is wrong, which metric would you use, and how do you choose the threshold? | M2.2, M3.1 |
| 6 | Why must the scaler be fitted inside each CV fold? Give two other leakage sources | M2.6, M3.2 |
| 7 | Bagging vs boosting: which one reduces variance, and which one reduces bias? | M2.4 |
| 8 | What is a point-in-time join, and what is training–serving skew? | M3.5 |
| 9 | Write a PyTorch training loop with a DataLoader, AdamW, a cosine schedule and a correct eval mode | M4.3 |
| 10 | Loss goes to NaN at step 200. Name three likely causes and how you would check each | M4.4 |
| 11 | Write causal self-attention (one head) in about 20 lines | M5.1 |
| 12 | MLOps level 1 vs level 2 in one sentence each | M6.1 |
| 13 | KS test, chi-square or PSI: which one for which feature type? | M6.8 |
| 14 | What does DDP all-reduce, and when do you switch to FSDP? | M7.3 |
| 15 | Permutation importance vs SHAP: what does each one answer? | M9.2 |

Expected for you: most rows route to the full module. M0.2, M3.3, M6.5 and parts of M6.9 are where your backend background is likely to pass the skip test.

- [ ] Diagnostic done, routing noted in `mlcore-lab/README.md`

---

## M0 Setup

### M0.1 Python data stack for a Java dev
**4 h** · Prereqs: AI core · Tested by: none (foundation)

**Skip test** (20 min, no docs): (1) z-score a (1e6, 20) array with broadcasting and explain the shapes involved; (2) per-customer 30-day rolling spend with `groupby().rolling()` on a datetime index; (3) explain view vs copy and `SettingWithCopyWarning`; (4) draw a 2×2 subplot grid with the Matplotlib axes API.

- `ndarray`, dtypes, strides/views, broadcasting, vectorisation vs Python loops, boolean masks and fancy indexing
- pandas Series/DataFrame, index alignment, `groupby` + `agg`/`transform` (≈ SQL GROUP BY / window functions), merge/join, datetime index and resampling, categorical dtype, nullable ints
- Matplotlib figure/axes API

**Resource:** Géron [`handson-mlp`](https://github.com/ageron/handson-mlp) tool notebooks for NumPy, pandas and Matplotlib, plus the linear-algebra and differential-calculus math primers (skim the primers; M1 covers the intuition).

**Exercise:** `m0/taxi_eda.py`. Load one month of NYC TLC yellow-taxi Parquet (the same data MLOps Zoomcamp uses). Compute trip duration, filter to 1–60 min, aggregate by pickup hour and zone, and produce three plots. Implement one feature twice, once as a Python loop and once vectorised. **Done when:** the script runs in under 10 s for one month, prints a loop-vs-vectorised timing of at least 50×, and saves the plots.

**Self-check:** Why is `df.apply(axis=1)` slow? What dtype does an int column with a NaN become? When does slicing return a view?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M0.2 ML lab setup
**2 h** · Prereqs: C0.2 · Tested by: none

**Skip test:** your C0.2 env and MLflow server start with one command, and you can open a free Colab GPU runtime and print `torch.cuda.get_device_name()`. If both are true, do only the exercise.

- Extend the C0.2 uv project with ML extras (see [Tool setup](#tool-setup)); register the Jupyter kernel from the uv env
- Free GPUs: Colab and Kaggle sessions are ephemeral. Checkpoint to Drive or the HF Hub, and pin package versions in the first cell
- Rented GPU budget plan: M5.3 (about $10), optional M7.3 multi-GPU. Set a hard spend cap with your provider
- Seeds and determinism: `torch.manual_seed`, `PYTHONHASHSEED`, cuDNN determinism flags, and the cost of using them

**Resource:** none ([Tool setup](#tool-setup) is the checklist).

**Exercise:** `mlcore-lab` skeleton: `pyproject.toml` with pinned versions, a `uv run pytest` smoke test that trains a scikit-learn model and logs it to your C0.2 MLflow server, and `colab/bootstrap.ipynb` that clones the repo, installs it and runs one epoch of a small PyTorch model on GPU. **Done when:** the smoke test passes locally and appears in the MLflow UI, and the Colab notebook finishes on GPU in under 5 minutes.

**Self-check:** Where do Colab checkpoints go when the session dies? Which seeds do you set, and what does full determinism cost you in speed?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

## M1 Math

Do these just in time. M1.3 comes before M2. M1.1 and M1.2 can wait until just before M2.5 and M4.1.

### M1.1 Linear algebra
**3 h** · Prereqs: none · Tested by: none (foundation for M2.5, M5.1, M7.2)

**Skip test:** (1) What does a matrix with a zero eigenvalue do to space? (2) Why is QKᵀ a matrix of dot products, and what does it measure? (3) How many parameters does a rank-8 LoRA update of a 4096×4096 matrix have, compared with the full matrix? (4) Explain change of basis in one sentence.

- Vectors, linear combinations, span, basis; matrices as linear transforms; multiplication as composition; determinant; inverse, column space, null space, rank; dot product and duality; change of basis; eigenvectors and eigenvalues; low rank

**Resource:** 3Blue1Brown [*Essence of Linear Algebra*](https://www.3blue1brown.com/topics/linear-algebra). Watch about 9 videos: vectors; linear combinations/span/basis; linear transformations and matrices; matrix multiplication as composition; determinant; inverse/column/null space; dot products and duality; change of basis; eigenvectors and eigenvalues. Skip cross products, Cramer's rule and abstract vector spaces.

**Exercise:** `m1/linalg.ipynb`. Implement PCA from the covariance eigendecomposition in NumPy on correlated 2-D Gaussian data and on a 50-D dataset. Then do a low-rank SVD approximation of a grayscale image. **Done when:** your components equal scikit-learn's PCA up to sign (`np.allclose`), and the rank-k reconstruction-error curve is plotted.

**Self-check:** Why are covariance eigenvectors orthogonal? What does truncating an SVD throw away?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M1.2 Calculus and backprop
**1.5 h** · Prereqs: M1.1 (light) · Tested by: none (foundation for M4)

**Skip test:** compute the gradient in diagnostic Q3 by hand; explain why the update steps against the gradient; say what backprop caches and reuses.

- Derivatives, partial derivatives, chain rule, gradient descent, backprop as reverse-mode chain rule

**Resource:** 3Blue1Brown [Neural Networks](https://www.3blue1brown.com/topics/neural-networks) ch1–4, up to and including "Backpropagation calculus". Ch5–7 belong to C1.2.

**Exercise:** `m1/gradcheck.py`. Write a 2-layer NumPy network with hand-written backprop and a finite-difference gradient checker. **Done when:** analytic and numeric gradients agree to a relative error below 1e-6 on random inputs.

**Self-check:** Why do you use central differences? Which way does gradient-check error go if your epsilon is too small?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M1.3 Probability and statistics
**4.5 h** · Prereqs: none · Tested by: none directly (used by M2.2, M6.7, M6.8)

**Skip test:** (1) Why is minimising cross-entropy the same as MLE for a Bernoulli model? (2) Give the correct reading of a 95 % confidence interval. (3) A/B test with 1,000 users per arm, 5.0 % vs 5.6 % conversion: is it significant? Compute it. (4) Diagnostic Q4.

- Distributions (normal, binomial, Poisson), mean/variance/SD, sampling vs population, conditional probability, Bayes, MLE, confidence intervals and the bootstrap, p-values, hypothesis tests (t, chi-square), Type I/II error and power

**Resource:** [StatQuest](https://statquest.org/video_index.html) Statistics Fundamentals, about 20 videos (titles as listed on the index, 2026-10-31): "Histograms, Clearly Explained"; "The Main Ideas behind Probability Distributions"; "The Normal Distribution"; "Population and Estimated Parameters" + "Estimating the Mean, Variance and Standard Deviation"; "Conditional Probability"; "Bayes' Theorem"; "Probability vs Likelihood" + "Maximum Likelihood" (and its normal and binomial worked examples); "Bootstrapping" Parts 1–2; "Confidence Intervals"; "Hypothesis Testing and the Null Hypothesis"; "p-values: What they are…" + "How to Calculate p-values"; "Linear Models Part 2: t-tests and ANOVA"; "Statistical Power" + "Power Analysis". The section has no chi-square video: learn it from the SciPy `chi2_contingency` docs while doing the exercise. Use [*Mathematics for ML*](https://mml-book.github.io/) ch2, 5–7 only as a reference when a derivation blocks you.

**Exercise:** `m1/stats_toolkit.py` with bootstrap CI, two-proportion z-test, chi-square test (SciPy) and a power calculation. Validate it by simulation. **Done when:** under the null, 10,000 simulated A/B tests give a false-positive rate between 4 % and 6 %. You will reuse this toolkit in M2.1, M6.7 and M6.8.

**Self-check:** What does p = 0.03 not mean? Why does peeking at an A/B test inflate Type I error?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

## M2 Classical ML

Pick **one public tabular dataset with an imbalanced binary label** (churn or fraud; check its licence) now. It becomes Capstone A's dataset and is called "the capstone dataset" below.

### M2.1 End-to-end ML project
**9 h** · Prereqs: M0.1, M1.3 · Tested by: MLA, PMLE, DMA, ADS, AIF/901 (light)

**Skip test:** (1) Why do you set the test set aside before EDA? (2) When do you use a stratified split? (3) Build a `ColumnTransformer` with numeric impute + scale and categorical one-hot from memory. (4) Grid vs random search: when is random better?

- ML taxonomy (supervised/unsupervised, batch/online, instance/model-based); framing and performance measure; EDA; stratified splits and test-set hygiene
- EDA checks the vendor exams name explicitly: outlier rules (z-score > 3, 1.5 × IQR) and when to clip, drop or keep; comparing two features by type pair (num×num: correlation/scatter; cat×cat: chi-square (→ M1.3)/contingency table; num×cat: grouped box plots)
- Numeric transforms (normalisation, log, clipping, binning); categorical encodings (one-hot, hashing, embeddings as concept, feature crosses); custom transformers; `Pipeline`/`ColumnTransformer`
- Grid and random search; final evaluation with a confidence interval; overfitting and generalisation

**Resource:** Géron *Hands-On ML with Scikit-Learn and PyTorch* (2025): ch1 skim (1 h), ch2 in full, coding along. Google [ML Crash Course](https://developers.google.com/machine-learning/crash-course) modules "Datasets, generalization, and overfitting" (105 min stated), "Numerical data" (85 min stated) and "Categorical data" (50 min stated) ([durations page](https://developers.google.com/machine-learning/crash-course/prereqs-and-prework), as of 2026-10-31).

**Exercise:** Reproduce ch2 on California housing, then port it to NYC-taxi trip-duration regression. You need one `train.py` with a pipeline, `RandomizedSearchCV`, test RMSE with a bootstrap CI (M1.3), and MLflow logging. **Done when:** a clean clone reproduces the test RMSE within ±1 %, and the fitted pipeline is logged as an MLflow model.

**Self-check:** Why can't you tune on the test set "just once"? What does a feature cross capture that two one-hots can't?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M2.2 Classification and metrics
**6 h** · Prereqs: M2.1 · Tested by: MLA, PMLE, DMA, ADS

**Skip test:** (1) Diagnostic Q5. (2) When is PR-AUC better than ROC-AUC? (3) Describe a procedure to pick the threshold for precision ≥ 0.9. (4) Macro vs micro vs weighted F1. (5) What does log loss punish that accuracy doesn't? (6) RMSE vs MAE under outliers.

- Confusion matrix, precision, recall, F1/Fβ, ROC and AUC, PR curve, log loss, threshold selection, multiclass (OvR/OvO, macro/micro averaging), multilabel/multioutput
- Regression metrics: RMSE, MAE, R², and when each misleads

**Resource:** Géron ch3; MLCC ["Classification"](https://developers.google.com/machine-learning/crash-course/classification) module (70 min stated); StatQuest ROC/AUC and confusion-matrix videos only if stuck.

**Exercise:** `m2/metrics_report.py`. Given labels and scores, it outputs a confusion matrix at a chosen threshold, ROC and PR curves, the threshold that meets a target precision, and a cost-weighted threshold. Apply it to a baseline logistic regression on the capstone dataset. **Done when:** pytest checks your precision/recall/F1/ROC-AUC against scikit-learn on three fixtures, and the report picks a threshold that meets the stated precision target on validation data.

**Self-check:** Why does ROC-AUC look good on very imbalanced data? What changes if false negatives cost 20× false positives?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M2.3 Linear models and generalisation
**6 h** · Prereqs: M2.2, M1.2 · Tested by: MLA, PMLE, DMA, ADS

**Skip test:** (1) Ridge vs lasso: which one produces zero coefficients, and why? (2) Read a learning curve: high train and val error that are close together means what? (3) When is a single holdout acceptable instead of k-fold? (4) Batch vs mini-batch vs SGD trade-offs.

- Normal equation vs GD (batch, mini-batch, stochastic); polynomial features; learning curves; bias–variance; ridge, lasso, elastic net; early stopping; logistic and softmax regression; k-fold CV vs holdout

**Resource:** Géron ch4; StatQuest bias and variance, cross-validation, ridge, lasso and elastic-net videos (as needed).

**Exercise:** In NumPy, implement mini-batch GD for L2-regularised logistic regression, plus learning curves for under- and over-fitting polynomial models. **Done when:** your coefficients match `LogisticRegression` (same C, no intercept penalty) within 1e-3, and each learning curve is annotated with its diagnosis.

**Self-check:** Why do you scale features before ridge? What does C mean in scikit-learn's logistic regression?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M2.4 Trees and ensembles
**7 h** · Prereqs: M2.3 · Tested by: MLA, PMLE, DMA, ADS

**Skip test:** (1) Diagnostic Q7. (2) Why are single trees unstable? (3) Name four XGBoost/LightGBM parameters that control overfitting. (4) What is out-of-bag error?

- CART, Gini vs entropy, depth/leaf regularisation; bagging, random forests, OOB, extra-trees; impurity importance (its flaws → M9.2)
- Boosting: AdaBoost, gradient boosting, XGBoost/LightGBM (learning rate, estimators, depth/leaves, subsampling, column sampling, early stopping); stacking and voting

**Resource:** Géron ch5–6; StatQuest "Gradient Boost" parts 1–2 and "XGBoost" parts 1–2.

**Exercise:** Benchmark logistic regression, random forest, XGBoost and LightGBM on the capstone dataset with identical stratified folds and early stopping. **Done when:** a table shows CV mean ± std of PR-AUC and fit time per model, and `m2/README.md` justifies the chosen model in 5 lines or fewer.

**Self-check:** Why does a lower learning rate need more trees? Why is early stopping on the test set leakage?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M2.5 Unsupervised learning
**5.5 h** · Prereqs: M2.2, M1.1 · Tested by: MLA, PMLE, DMA, ADS

**Skip test:** (1) How do you choose the number of PCA components? (2) k-means vs DBSCAN: what kind of shapes does each find? (3) GMM vs k-means. (4) Two ways to detect anomalies without labels.

- Curse of dimensionality; PCA (explained variance), random projection, t-SNE/UMAP (for visualisation only); k-means (inertia, silhouette, k-means++), DBSCAN, GMM; anomaly and novelty detection (GMM density, Isolation Forest)

**Resource:** Géron ch7–8; StatQuest PCA if stuck.

**Exercise:** An unsupervised anomaly detector on the capstone dataset's features (labels hidden): Isolation Forest plus GMM log-likelihood, after PCA to 95 % variance. **Done when:** its PR-AUC against the held-out labels is reported next to the M2.4 supervised baseline, with one paragraph on when you would ship the unsupervised version.

**Self-check:** Why do you scale before PCA? Why is silhouette misleading for DBSCAN?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M2.6 Pipelines and leakage drills
**3 h** · Prereqs: M2.4 · Tested by: MLA, PMLE, DMA, ADS

**Skip test:** spot the leak in each case: (a) scaler fitted on all data; (b) target encoding fitted on the full train set; (c) a "days since account closed" feature; (d) the same patient in train and test.

- Pipelines inside CV, missing-value strategies, categorical handling, XGBoost in pipelines; leakage types: target, train–test contamination, temporal, group

**Resource:** Kaggle Learn [*Intermediate Machine Learning*](https://www.kaggle.com/learn/intermediate-machine-learning) lessons: Missing Values, Categorical Variables, Pipelines, Cross-Validation, XGBoost, Data Leakage. Hours are not stated on a page I could render; about 3–4 h is my estimate.

**Exercise:** `m2/leakage_lab.py`. Reproduce three leaks (pre-split scaling, non-CV target encoding, a future feature), measure the inflated metric, then fix each one. **Done when:** a table shows the three leaked-vs-fixed metric deltas, and a pytest asserts that every fit happens inside `cross_validate` on training indices only.

**Self-check:** Why does `GroupKFold` exist? Which of your capstone features could leak at serving time?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

## M3 Data and features

### M3.1 Training data
**3 h** · Prereqs: M2.2 · Tested by: MLA, PMLE, DMA, ADS, PADS

**Skip test:** (1) What is reservoir sampling, and when do you need it? (2) Natural labels and feedback-loop length: give an example of each. (3) Name three fixes for class imbalance, at the data, algorithm and decision levels. (4) Weak supervision vs active learning.

- Sampling: non-probability, simple random, stratified, weighted, reservoir, importance
- Labelling: hand labels, label multiplicity, data lineage, natural labels, feedback-loop length; weak supervision, semi-supervision, transfer, active learning
- Class imbalance: right metrics, resampling (under/over, SMOTE), class weights and cost-sensitive loss, focal loss; augmentation
- Tabular synthetic data, concept only: SMOTE-style interpolation vs generative synthesizers (Gaussian copula, CTGAN), and the two checks every synthetic set needs (fidelity to real marginals/correlations, no memorised real rows). LLM-era synthetic data stays in → C11.3

**Resource:** Huyen *Designing Machine Learning Systems* (DMLS) ch4 "Training Data".

**Exercise:** On the capstone dataset, compare class weights, random undersampling, SMOTE (imbalanced-learn) and threshold moving. Also implement reservoir sampling. **Done when:** a table shows PR-AUC and recall at your target precision for each method, and your reservoir sampler passes a chi-square uniformity test from the M1.3 toolkit.

**Self-check:** Why does SMOTE inside CV need a pipeline? When is threshold moving all you need?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M3.2 Feature engineering
**5 h** · Prereqs: M3.1, M2.6 · Tested by: MLA, PMLE, DMA, ADS, PADS

**Skip test:** (1) What are MCAR, MAR and MNAR, and what does each imply for imputation? (2) How do you do target encoding without leakage? (3) When do you prefer the hashing trick to one-hot? (4) Why can a time-based split change the winning model?

- Missing values and imputation; scaling, log transforms (if you train on `log1p(y)`, back-transform predictions with `expm1` before computing RMSE/MAE on the original scale), discretisation; one-hot vs hashing vs target encoding (CV-safe); feature crosses; time-window aggregates; mutual information; feature importance and generalisation; leakage sources (time, duplicates, groups, pre-split statistics); time-based splits

**Resource:** DMLS ch5 "Feature Engineering"; Kaggle Learn [*Feature Engineering*](https://www.kaggle.com/learn/feature-engineering) lessons: mutual information, creating features, clustering with k-means, PCA, target encoding (lesson list and "5 h" from a search snippet [unconfirmed]; the page is JS-rendered and did not render for verification).

**Exercise:** Add about 10 engineered features to the capstone dataset: time-window aggregates, CV-safe target encoding and one cluster feature. Rank them by MI, then evaluate on a random split and on a time-based split. **Done when:** an ablation table shows each feature group's delta under both splits, and the gap between the two splits is explained in 3 lines or fewer.

**Self-check:** Which of your features cannot be computed at serving time with the same freshness? How would you detect that?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M3.3 Data engineering for ML
**1.5 h** · Prereqs: M0.1 · Tested by: MLA, PMLE (processing-engine choice, light)

**Skip test** (you will probably pass it): row vs columnar formats; OLTP vs OLAP; ETL vs ELT; batch vs stream features; dataflow via databases, services and real-time transport.

- Data formats (row/column, text/binary, Parquet), data models, storage engines and processing, ETL/ELT, dataflow modes, batch vs stream processing for features

**Resource:** DMLS ch3 "Data Engineering Fundamentals". Skim it; read only the ML-specific angles.

**Exercise:** Convert 12 months of taxi CSV to month-partitioned Parquet with pyarrow, then compare size and the read time of a 2-column projection. **Done when:** a timing/size table is committed and the partitioned dataset is the input for the M3.5 and M6.4 pipelines.

**Self-check:** Why does columnar storage help feature computation, and when does it hurt?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M3.4 Data quality and validation
**3 h** · Prereqs: M3.2 · Tested by: MLA, PMLE, 300, ADS

**Skip test:** write five expectations for an incoming transactions table, and say where in the pipeline each one runs and what happens when it fails (halt or quarantine).

- Expectations (schema, types, ranges, uniqueness, nulls, distributions); data contracts between producers and ML consumers; validation points (ingest, pre-train, pre-serve); fail loudly vs quarantine

**Resource:** [Made With ML](https://madewithml.com/courses/mlops/) "Testing" lesson, data section only (the code and model sections → M6.9), and the "Data Engineering" lesson in the Production block.

**Exercise:** An expectation suite of at least 12 checks for the capstone raw data, using Great Expectations as in Made With ML, or plain pytest assertions if the GE API has drifted [verify current GE version]. Inject three corruptions (a schema change, a null spike, a range violation). **Done when:** each corruption fails with a readable message, and the suite is callable as a pipeline step (wired into CI in M6.9).

**Self-check:** Which checks belong to the producer's contract and which to your pipeline?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M3.5 Feature stores and skew
**4 h** · Prereqs: M3.2, M3.3 · Phase 3 · Tested by: MLA, PMLE, DMA, DMP

**Skip test:** (1) Diagnostic Q8. (2) Offline vs online store: what lives where, and why? (3) Give three ways training–serving skew sneaks in. (4) Why log features at serving time?

- Entities, feature views, offline/online stores, materialisation, point-in-time-correct joins, feature registry; training–serving skew (code paths, data freshness, join tables changing); logging served features; reusing code between train and serve

**Resource:** [Feast docs](https://docs.feast.dev/): Concepts pages (overview, entity, feature view, data ingestion, feature retrieval, point-in-time joins, feature repo) + Quickstart. Google [*Rules of ML*](https://developers.google.com/machine-learning/guides/rules-of-ml): read all 43 rules quickly, then Phase II (rules 16–37) closely, especially the training–serving skew rules.

**Exercise:** A Feast repo for the capstone. Entity = customer (or card); feature views = rolling aggregates from the M3.3 Parquet. Use `get_historical_features` with event timestamps to build the training set, materialise to the local online store, and serve via `get_online_features` in a FastAPI stub. **Done when:** a skew test finds zero mismatches between offline and online vectors for 100 entities, and a deliberately future-dated row is excluded from the training set.

**Self-check:** What breaks if you join on "latest value" instead of point-in-time? Which Rules of ML Phase II advice does Feast not enforce for you?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M3.6 Spark ML at scale (conditional)
**5 h** · Prereqs: M2.6 · Required before DBX-MLA, DBX-MLP, GCP-PMLE · Tested by: DMA, DMP, PMLE, MLA (EMR), ADS

**Skip test:** (1) Transformer vs Estimator vs PipelineModel. (2) Why use `VectorAssembler`? (3) CrossValidator vs TrainValidationSplit. (4) What is the status of the RDD-based `spark.mllib` API?

- DataFrame-based `spark.ml` pipelines; StringIndexer/OneHotEncoder/VectorAssembler; CrossValidator, ParamGridBuilder; persistence; distributed preprocessing. The RDD-based `spark.mllib` is in maintenance mode; `spark.ml` is primary (docs for Spark 4.3.0, as of 2026-10-31). Your JVM background makes the execution model cheap to learn

**Resource:** [Spark MLlib guide](https://spark.apache.org/docs/latest/ml-guide.html): "ML Pipelines"; "Extracting, transforming and selecting features" (the transformers you use only); "ML Tuning: model selection and hyperparameter tuning".

**Exercise:** Port the M2.1 pipeline to PySpark `local[*]` over 12 months of taxi Parquet, with a CrossValidator. **Done when:** the saved `PipelineModel` reloads and scores, metrics are within 2 % of the scikit-learn version, and a Spark UI screenshot of the stages is committed.

**Self-check:** Where does a shuffle happen in your pipeline? Why can't you use a scikit-learn transformer inside it?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

## M4 Deep learning

### M4.1 Neural nets from scratch
**12 h** · Prereqs: M1.2, M1.3 · Tested by: foundation for M4.3–M4.4 (PGENL indirectly)

**Skip test** (45 min): implement a micrograd-style `Value` with `+`, `*`, `tanh` and `backward()`, and explain why gradients accumulate with `+=`. Then explain negative log-likelihood for a bigram model.

- Computational graphs, reverse-mode autograd, neurons and MLPs, bigram language model, softmax and NLL, embedding tables, train/dev/test, minibatches, learning-rate search

**Resource:** Karpathy [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html) lectures 1 micrograd (2h25m), 2 makemore (1h57m), 3 makemore MLP (1h15m). Durations as stated on the page. Code along.

**Exercise:** Finish all three code-alongs, then extend micrograd with `exp`, `pow` and a 2-layer MLP on `make_moons`. **Done when:** a test shows your micrograd gradients equal PyTorch autograd on three random graphs, and your makemore-MLP dev loss is within about 5 % of the lecture's result.

**Self-check:** Why is softmax + NLL computed with log-sum-exp? What does the embedding table learn?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M4.2 Training dynamics
**9 h** · Prereqs: M4.1 · Tested by: foundation for M4.4 (PGENL indirectly)

**Skip test:** (1) Why does tanh saturation kill gradients? (2) What is Kaiming init scaling for? (3) BatchNorm in train vs eval mode. (4) What should the update-to-data ratio look like?

- Activation and gradient statistics, initialisation, saturation and dead units, BatchNorm (running stats), manual backprop through cross-entropy, BN and linear layers

**Resource:** Karpathy lecture 4 "Activations & Gradients, BatchNorm" (1h55m) and lecture 5 "Becoming a Backprop Ninja" (1h55m). Lecture 6 WaveNet (56m) is optional.

**Exercise:** `m4/diagnostics.py`. Hooks that plot per-layer activation stats, gradient stats and update:data ratio for any `nn.Module`. Apply them to a deliberately badly initialised MLP, then fix it. **Done when:** before/after plots show the saturation fixed and a better loss curve, and the lecture-5 manual-backprop comparisons all pass.

**Self-check:** Why does BatchNorm make the batch examples coupled? What replaces it in transformers?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M4.3 Idiomatic PyTorch and deep-net training
**11 h** · Prereqs: M4.2 · Tested by: MLA, PMLE, DMA, DMP, PGENL

**Skip test:** diagnostic Q9 in 15 minutes, including checkpoint/resume and gradient clipping, and an explanation of what `model.eval()` and `torch.no_grad()` each do.

- Tensors and devices; Dataset/DataLoader (workers, `pin_memory`, collate); `nn.Module`; losses; SGD+momentum, Adam, AdamW; LR schedules (warmup, cosine, one-cycle); dropout, weight decay, early stopping; gradient clipping; checkpointing; reproducibility

**Resource:** Géron ch9 (artificial neural networks), ch10 (neural nets with PyTorch), ch11 (training deep neural networks: init, activations, BN, optimizers, LR scheduling, regularisation).

**Exercise:** `m4/trainer.py`. A reusable loop with train/eval, AMP-ready, checkpoint/resume, early stopping and MLflow logging. Use it to train an MLP on the capstone dataset and a small CNN on Fashion-MNIST. **Done when:** a killed run resumes from its checkpoint and reproduces the same subsequent losses (seeded), and the MLP-vs-XGBoost comparison is logged in MLflow.

**Self-check:** Why AdamW and not Adam + L2? What happens if you forget `optimizer.zero_grad()`?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M4.4 Tuning and training troubleshooting
**4 h** · Prereqs: M4.3 · Tested by: MLA, PMLE, DMA, DMP, PGENL

**Skip test:** (1) Diagnostic Q10. (2) Scientific vs nuisance vs fixed hyperparameters. (3) TPE vs random search: what does TPE model? (4) What does a pruner do?

- Tuning strategy: incremental, scientific approach, batch size and LR coupling, choosing training steps; diagnosing divergence, overfitting and data bugs from curves
- Bayesian HPO (TPE), pruning (median, Hyperband/ASHA), search budgets; the managed equivalents (SageMaker AMT, Vertex, Ray Tune) belong to the vendor paths and M7.4

**Resource:** Google [*Deep Learning Tuning Playbook*](https://github.com/google-research/tuning_playbook) sections "Guide for starting a new project", "A scientific approach to improving model performance", "Determining the number of steps for each training run" and "Setting up experiment tracking". [Optuna tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html), "Key Features" 1–5 (Lightweight/platform-agnostic architecture; Pythonic search space; Efficient optimization algorithms; Easy parallelization; Quick visualization) [verify: not part of the research pass; section titles seen on the page 2026-10-31].

**Exercise:** An Optuna study (SQLite storage, MedianPruner) over the M4.3 trainer (LR, weight decay, dropout, width), logged to MLflow. Plus a "break it" drill: inject LR×100, shuffled labels, a missing `zero_grad` and eval in train mode, then diagnose each from its curves alone. **Done when:** the study has at least 40 trials, a parameter-importance plot exists, the best trial beats the defaults by a measured margin, and `m4/bugs.md` maps symptom → cause for all four bugs.

**Self-check:** Why tune LR on a log scale? Why can pruning bias a study?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M4.5 CNNs, CV and vision transformers
**7 h** · Prereqs: M4.3 · Tested by: PMLE, GENM, PGENL

**Skip test:** (1) Output size of a conv with stride 2 and padding 1 on 224×224. (2) Why does transfer learning freeze early layers? (3) Detection vs semantic vs instance segmentation. (4) How does ViT turn an image into tokens? (5) What objective trains CLIP?

- Convolutions, pooling, receptive field; classic architectures (ResNet and residuals); transfer learning and fine-tuning; detection and segmentation overview; ViT patches; CLIP-style contrastive image–text models (use-level VLMs → C12.1)

**Resource:** Géron ch12 (CNNs) and ch16 (vision and multimodal transformers). Backup: [CS231n notes](https://cs231n.github.io/) Module 2 (CNNs, visualising CNNs, transfer learning) only if ch12 feels thin. Optional (+2 h, only for NCA-GENM): the diffusion section of Géron ch18 (intuition → C12.4).

**Exercise:** On a small torchvision image dataset (for example Oxford-IIIT Pets), compare CLIP zero-shot (HF `transformers`) vs fine-tuned ResNet vs fine-tuned ViT (torchvision weights) on a Colab T4. **Done when:** a table shows accuracy, trainable parameters and training time for all three, with one paragraph on which you would ship and why.

**Self-check:** Why does ViT need more data than a CNN to train from scratch? What does CLIP zero-shot fail at?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M4.6 Sequence models (skim)
**1.5 h** · Prereqs: M4.3 · Tested by: none directly (background for M8.2)

**Skip test:** why do RNNs struggle with long dependencies, and what fixes do LSTM gates provide?

- RNN, LSTM/GRU, 1-D CNNs for sequences, windowed forecasting setup

**Resource:** Géron ch13, skim (about 1 h).

**Exercise:** LSTM vs 1-D CNN vs naive last-value forecast on a synthetic seasonal series. **Done when:** a MAE table exists, and if a neural model loses to naive, one sentence says why. M8.2 does forecasting properly.

**Self-check:** Why have transformers replaced RNNs for most sequence tasks?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

## M5 Transformers and LLM training

### M5.1 Build a GPT
**7 h** · Prereqs: M4.2, C1.2 · Tested by: PGENL, AGENL

**Skip test:** diagnostic Q11 with a causal mask and multi-head attention, and an explanation of pre-norm residual blocks and why attention is scaled by 1/√d.

- Attention as data-dependent weighted aggregation, causal mask, scaled dot-product, multi-head, residual streams, LayerNorm (pre-norm), MLP block, positional embeddings, dropout, character-level LM training (intuition stays in → C1.2)

**Resource:** Karpathy "Let's build GPT: from scratch, in code, spelled out" (1h56m), code along.

**Exercise:** Train nanoGPT-style on tiny Shakespeare, then swap learned positional embeddings for sinusoidal or RoPE (the Géron positional-encoding appendix is a reference) and compare. **Done when:** both variants train on a Colab T4 with validation loss reported, and a unit test proves the output at position t does not change when tokens after t are changed.

**Self-check:** What breaks without the 1/√d scale? Why do residual connections help depth?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M5.2 Tokenizers
**5 h** · Prereqs: M5.1, C1.4 · Tested by: PGENL, AGENL

**Skip test:** (1) Walk through one BPE merge step. (2) Byte-level BPE vs WordPiece vs Unigram. (3) What does doubling the vocab size do to embedding parameters and to sequence length? (4) Why do special tokens need care in fine-tuning?

- UTF-8 bytes, BPE training and merges, regex pre-tokenisation, special tokens; WordPiece, Unigram; normalisation; vocab-size trade-offs; training a domain tokenizer; offset mapping (pitfalls stay in → C1.4)

**Resource:** Karpathy "Let's build the GPT Tokenizer" (2h13m). [HF LLM Course](https://huggingface.co/learn/llm-course/chapter6/1) ch6 (training a new tokenizer from an old one, fast tokenizers, normalization and pre-tokenization, BPE, WordPiece, Unigram, building a tokenizer block by block).

**Exercise:** A pure-Python byte-level BPE trainer, plus an HF `tokenizers` BPE trained on a corpus of open-source Java code. That is your domain, so the results will be meaningful to you. **Done when:** your BPE round-trips encode/decode on 1,000 random Unicode strings, and a table compares tokens-per-character for your 8k and 32k vocabularies vs the GPT-2 tokenizer on Java source.

**Self-check:** Why do LLMs miscount characters? Why is Java code tokenised worse by a web-text tokenizer?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M5.3 Pretraining at small scale
**11 h** · Prereqs: M5.2, M4.3 · Tested by: PGENL, AGENL

**Skip test:** (1) What do bf16/TF32, `torch.compile` and flash attention each speed up? (2) Gradient accumulation math for a 0.5 M-token batch on one GPU. (3) Why use warmup + cosine decay? (4) What does HellaSwag measure?

- GPT-2 (124M) architecture, weight tying, initialisation; mixed precision; `torch.compile`; flash attention (SDPA); vocab padding; AdamW, warmup + cosine, gradient clipping; gradient accumulation; DDP via `torchrun` (intro; depth → M7.3); data shards; HellaSwag eval

**Resource:** Karpathy "Let's reproduce GPT-2 (124M)" (about 4 h video [length unconfirmed]) with [`build-nanogpt`](https://github.com/karpathy/build-nanogpt). The README says reproducing GPT-2 (124M) today "is a matter of ~1hr and ~$10" on a cloud GPU box (as of 2026-10-31). Optional capstone: [`nanochat`](https://github.com/karpathy/nanochat), whose README puts a GPT-2-capability speedrun on one 8×H100 node at about 1.5–2 h for about $48 (about $15 on spot) as of 2026-10-31. It is not in the hour total.

**Exercise:** Code along with a shrunken config on Colab or Kaggle. Optionally do the full rented run (about $10). **Done when:** a table records tokens/sec after each optimisation step (precision, compile, SDPA, vocab padding, fused AdamW). If you rented, the HellaSwag accuracy is logged in MLflow.

**Self-check:** Which optimisation gave the biggest jump, and why does Horace He's model (M7.1) predict that?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M5.4 Hugging Face ecosystem
**3 h** · Prereqs: M4.3 · Tested by: PGENL (tooling)

**Skip test:** load a model with `AutoTokenizer`/`AutoModelForSequenceClassification`, batch-tokenise with padding and truncation, run inference; stream a Hub dataset and `map` over it in batches.

- `pipeline`, AutoModel/AutoTokenizer, model heads, padding/attention masks; Hub; Datasets: loading, slicing, `map(batched=True)`, memory mapping, streaming big data, creating and sharing a dataset

**Resource:** HF LLM Course ch1–2 (skim) and ch5 "The 🤗 Datasets library". Skip ch5's FAISS semantic-search section (→ C7.3).

**Exercise:** Load, clean and push a private text dataset for Capstone C with batched `map`/`filter`, and stream a large Hub dataset. **Done when:** a streaming pass over at least 1 M rows stays under 2 GB RSS (measured), and a dataset card is written.

**Self-check:** Why does Arrow memory-mapping let you `map` over datasets larger than RAM?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M5.5 Classic NLP task heads
**7 h** · Prereqs: M5.2, M5.4 · Tested by: PGENL (light)

**Skip test:** (1) How do you align word-level NER labels to subword tokens? (2) What does domain-adaptive MLM buy you? (3) How does extractive QA handle contexts longer than the max length?

- Token classification/NER (label alignment, seqeval), masked-LM domain adaptation, translation and summarisation (seq2seq, collators), extractive QA (offsets, striding), causal LM from scratch. Choosing an LLM vs a specialised model → C12.3; this module builds the specialised side

**Resource:** HF LLM Course [ch7](https://huggingface.co/learn/llm-course/chapter7/1) "Classical NLP tasks": token classification, fine-tuning a masked language model, translation, summarization, training a causal LM from scratch, question answering.

**Exercise:** Fine-tune a small encoder for NER (CoNLL-2003) or a text classifier relevant to your work (non-proprietary data). Benchmark it against your C12.3 LLM-prompted baseline. **Done when:** a table shows task F1, p95 latency and cost per 1k requests for both, plus a one-line recommendation.

**Self-check:** At what volume does the fine-tuned encoder beat the LLM on total cost?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M5.6 Fine-tuning mechanics
**6 h** · Prereqs: M5.4 · Tested by: PGENL, MLA (FM tuning), 300

**Skip test:** (1) Dynamic padding with a collator: why? (2) What does `TrainingArguments` control that your M4.3 loop did by hand? (3) What does Accelerate abstract?

- Preprocessing and collators, Trainer API, `compute_metrics`, eval/save strategies, the same loop in raw PyTorch, Accelerate for multi-device, effective batch size

**Resource:** HF LLM Course [ch3](https://huggingface.co/learn/llm-course/chapter3/1) "Fine-tuning a pretrained model": "Processing the data", "Fine-tuning a model with the Trainer API", "A full training loop" (its "Supercharge your training loop with 🤗 Accelerate" section is the distributed part), "Understanding Learning Curves".

**Exercise:** Train the M5.5 task three ways: Trainer, your M4.3 loop, and an Accelerate script run with `accelerate launch` on 2 processes (CPU, or Kaggle's dual-T4 option [verify current Kaggle GPU options]). **Done when:** all three are within ±0.5 points on the metric, and the 2-process run logs the correct effective batch size.

**Self-check:** Why must LR sometimes change when you add processes?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M5.7 LLM fine-tuning and alignment implementation
**8 h** · Prereqs: M5.6, C11.1, C11.2, C11.4 · Tested by: PGENL, MLA (FM tuning), 300

**Skip test:** (1) How do you train only on assistant tokens with a chat template? (2) What do LoRA's r, alpha and target_modules each control? (3) What data does DPO need? (4) What does GRPO need instead of a reward model?

- Chat templates and loss masking, `SFTTrainer`, packing, LoRA/QLoRA config and merging, DPO on preference pairs, GRPO with programmatic rewards, evaluation and early stopping (concepts and the decision → C11.1–C11.2)

**Resource:** HF LLM Course [ch11](https://huggingface.co/learn/llm-course/chapter11/1) "Fine-tune Large Language Models" (chat templates, SFT with `SFTTrainer`, LoRA, evaluation) and [ch12](https://huggingface.co/learn/llm-course/chapter12/1) "Build Reasoning Models" (RL on LLMs, the DeepSeek-R1 paper, GRPO in DeepSeekMath, implementing GRPO in TRL, the GRPO practical exercise; the Unsloth exercise is optional). Ch12 is still unfinished: its last section is "Coming soon..." as of 2026-10-31.

**Exercise:** Redo your C11.4 task with raw TRL `SFTTrainer` + PEFT (no Unsloth), then run DPO on 200–500 preference pairs built with your C11.3 method. Optional: GRPO with a verifiable reward such as JSON-schema validity. **Done when:** a table compares base vs SFT vs SFT+DPO on your C3.3 eval set, and the merged model is served from Ollama or vLLM (C10.4).

**Self-check:** What did Unsloth hide that you now configure yourself? When did DPO hurt?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M5.8 Training embedding models
**4 h** · Prereqs: M5.6, C7.3, C7.5 · Tested by: PGENL, MLA, 300

**Skip test:** (1) Multiple-negatives-ranking loss and in-batch negatives. (2) Why do hard negatives matter? (3) Bi-encoder vs cross-encoder for retrieval vs reranking. (4) How is a two-tower recommender related?

- Contrastive learning, bi-encoders, MNR loss, hard-negative mining, unsupervised adaptation (TSDAE), two-tower models (bridge to M8.1); rerankers as consumers → C7.4

**Resource:** Alammar & Grootendorst *Hands-On Large Language Models* ch10 "Creating Text Embedding Models" + its [notebook](https://github.com/handsOnLLM/Hands-On-Large-Language-Models) (free, runs on a Colab T4).

**Exercise:** Fine-tune a small sentence-transformer on your C7 RAG corpus using (synthetic question, passage) pairs from C7.5 plus mined hard negatives, then re-index in pgvector. **Done when:** recall@10 (metric → C7.5) on held-out synthetic queries is reported before and after. If it doesn't improve, a note says why.

**Self-check:** Why can fine-tuning an embedder hurt out-of-domain retrieval?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M5.9 LLM benchmarking and LM metrics
**3 h** · Prereqs: M5.7 · Tested by: PGENL, AGENL

**Skip test:** (1) How do perplexity and cross-entropy relate? (2) Why can't you compare perplexity across tokenizers? (3) Loglikelihood vs generate-until tasks. (4) Two ways to detect benchmark contamination.

- Entropy, cross-entropy, bits-per-byte/character, perplexity; computing BLEU/ROUGE/BERTScore (meaning → C3.2); harness task types, `acc` vs `acc_norm`, few-shot settings; contamination and n-gram overlap; saturation; private vs public evals (→ C3.4)

**Resource:** Huyen *AI Engineering* ch3 section "Understanding Language Modeling Metrics" (you own the book from the AI core). [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) README plus one run.

**Exercise:** Run lm-eval on your M5.7 base vs SFT models (HellaSwag + one task, with `--limit` documented). Compute perplexity on held-out domain text yourself from logits. Compute ROUGE/BLEU for a summarisation sample. **Done when:** your perplexity equals exp(mean token NLL) from the model's loss within 1 %, and the results table records the limit and the few-shot settings.

**Self-check:** Why did SFT change HellaSwag (or not)?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

## M6 MLOps

The MLOps Zoomcamp is free and self-paced. Its README says there is no live cohort in 2026 ([repo](https://github.com/DataTalksClub/mlops-zoomcamp), as of 2026-10-31). Do each module's homework. Treat the AWS Kinesis/Lambda parts as examples, not syllabus.

### M6.1 MLOps concepts and maturity
**3 h** · Prereqs: M2.1 · Tested by: MLA, PMLE, 300, DMA, DMP, PADS

**Skip test:** (1) Diagnostic Q12. (2) What triggers continuous training? (3) Name four Sculley debt types and give an example of each from software you know.

- MLOps levels 0/1/2, continuous training, metadata store, model registry, pipeline triggers; technical-debt vocabulary: entanglement (CACE), hidden feedback loops, undeclared consumers, glue code, pipeline jungles, configuration debt

**Resource:** Google Cloud ["MLOps: Continuous delivery and automation pipelines in machine learning"](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) (whole). Sculley et al. ["Hidden Technical Debt in Machine Learning Systems"](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) (whole). MLOps Zoomcamp module 1 + homework.

**Exercise:** `docs/mlops-maturity.md` for Capstone A: current vs target level, CT triggers, metadata, registry, and five Sculley debt items with mitigations. **Done when:** the doc fits on one page and the module-1 homework passes.

**Self-check:** Which debt type does your Java publishing workflow already exhibit?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M6.2 Experiment tracking and registry
**6 h** · Prereqs: M6.1, M2.4 · Tested by: MLA, PMLE, 300, DMA, DMP, PADS

**Skip test:** (1) Backend store vs artifact store. (2) What does a model signature protect against? (3) Aliases vs the legacy stages: what do you promote with? (4) How do you record dataset lineage on a run?

- Runs, params, metrics, artifacts, tags; autolog; nested runs (`start_run(nested=True)`: one parent per tuning/CV study, a child per trial or fold, a separate run for the final refit); logging tables, figures and dicts (`log_table`, `log_figure`, `log_dict`); flavors, signatures, input examples; registered models, versions, aliases (champion/challenger); lineage (dataset hash, git SHA). MLflow 3's GenAI side → C8.3

**Resource:** MLOps Zoomcamp module 2 (experiment tracking and model management with MLflow) + homework.

**Exercise:** Refactor capstone training to log dataset hash, git SHA, params, metrics and signature. `promote.py` moves alias `@champion` only if the candidate beats the current champion on a frozen test set. **Done when:** `promote.py` refuses a worse model (tested), and a serving stub loads `models:/<name>@champion`.

**Self-check:** Why should promotion be a script and not a click?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M6.3 Offline evaluation discipline
**4 h** · Prereqs: M2.2 · Tested by: MLA, PMLE, 300, DMA, DMP, PADS

**Skip test:** (1) Three baselines every model must beat. (2) What is calibration, and how do you fix a miscalibrated model? (3) Perturbation vs invariance vs directional-expectation tests. (4) Why slice?

- Baselines (random, most-frequent, heuristic, existing system); perturbation, invariance and directional tests; calibration (reliability diagrams, Brier, Platt/isotonic); confidence; slice-based evaluation; model selection tips

**Resource:** DMLS ch6 "Model Development and Offline Evaluation". Read "Model Offline Evaluation" closely; skim the distributed training and AutoML parts (→ M7.3 / vendor paths).

**Exercise:** `m6/offline_eval.py` producing three baselines, a calibration curve with Brier before and after isotonic, slice metrics on two attributes, and perturbation and invariance tests. **Done when:** the report runs as a pipeline step, has all five sections, and at least one finding is filed as a GitHub issue.

**Self-check:** Why can a better-AUC model be worse for your business threshold?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M6.4 Pipelines and orchestration
**6 h** · Prereqs: M6.2, M3.4, M3.5 · Tested by: MLA, PMLE, 300, DMA, DMP, PADS

**Skip test** (you may pass it on the concepts): DAGs, retries, idempotency, backfills, parameterised schedules. You probably know these from Spring Batch. Skip the reading only if you can also already write and schedule a Prefect flow.

- Notebook → script → pipeline; tasks, DAGs, retries, caching, scheduling, parameters, artifacts; idempotent reruns and backfills; triggers (schedule, data arrival, drift from M6.8)

**Resource:** MLOps Zoomcamp module 3: "Introduction to ML Pipelines", "Turning the Notebook into a Python Script", "Using an Orchestrator" + homework. The current "Using an Orchestrator" lesson is tool-agnostic: it names Airflow and Prefect as common choices (Dagster, Kestra, Mage as others) and points to the Prefect and Mage examples from previous cohorts (checked 2026-10-31). **Use Prefect, and only Prefect**: module 5's batch monitoring also schedules with Prefect, and the [2022 cohort folder](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/cohorts/2022/03-orchestration) has worked Prefect material. (This tool pick is this guide's, not the course's.)

**Exercise:** The Capstone A DAG: ingest → validate (M3.4) → features (M3.5) → train → offline eval (M6.3) → register/promote (M6.2), with retries and a schedule. **Done when:** rerunning the same date creates no duplicate artifacts, and a failed validation stops the run before training.

**Self-check:** Which of your steps are not idempotent, and how did you fix them?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M6.5 Deployment patterns
**5 h** · Prereqs: M6.2 · Tested by: MLA, PMLE, 300, DMP, PGENL

**Skip test** (you will probably pass it on the service half): containerised HTTP service with health/readiness checks, model loaded at startup, schema validation. Then answer: (1) When is batch prediction better than online? (2) How do batch and online predictions diverge? (3) What edge constraints force compression?

- Batch vs online vs streaming prediction; unifying batch and streaming feature pipelines; pre/post-processing placement; containers and model loading; edge vs cloud; production patterns → C4.1

**Resource:** MLOps Zoomcamp module 4 + homework: 4.1 deployment modes, 4.2 web service with Flask and Docker (you build yours in FastAPI), 4.3 getting models from the model registry, 4.4 Kinesis/Lambda streaming (read only, it can create AWS charges), 4.5 batch scoring script. Skip 4.6 (batch scoring with Mage), which would add a second orchestrator. DMLS ch7 "Model Deployment and Prediction Service" (compression section: skim, → M7.2).

**Exercise:** A FastAPI + Docker service loading `@champion`, with a Pydantic request schema, `/health` and `/predict`, plus a batch-scoring CLI over Parquet that shares the same preprocessing module. **Done when:** online and batch predictions are identical for the same 1,000 rows (test), and p95 latency at 50 RPS is recorded with a simple async `httpx` load script.

**Self-check:** Where does your service read features from at serving time, and is it the M3.5 online store?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M6.6 Serving on Kubernetes
**4 h** · Prereqs: M6.5 · Tested by: MLA, PMLE, 300, DMP, PGENL

**Skip test:** (1) What is an InferenceService? (2) How does a KServe canary split traffic, and how do you roll back? (3) What does InferenceGraph add? (4) When is scale-to-zero a bad idea?

- InferenceService, predictors and runtimes (scikit-learn/XGBoost/ONNX), autoscaling and scale-to-zero (deployment-mode dependent), canary rollouts, InferenceGraph (pipelines and ensembles), LLM serving with vLLM/llm-d backends (vLLM itself → C10.4). TorchServe is archived; do not learn it

**Resource:** [KServe docs](https://kserve.github.io/website/) (v0.20, a CNCF incubating project, as of 2026-10-31): [Quickstart Guide](https://kserve.github.io/website/docs/getting-started/quickstart-guide); [Deploy Your First Predictive Model with InferenceService](https://kserve.github.io/website/docs/getting-started/predictive-first-isvc); [Canary Rollout Strategy](https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary) + its example page; [InferenceGraph overview](https://kserve.github.io/website/docs/model-serving/inferencegraph/overview); read only: [Deploy Your First LLM with InferenceService](https://kserve.github.io/website/docs/getting-started/genai-first-isvc) (vLLM backend). Note from the canary page: canary rollout "is only supported in serverless deployment mode" (Knative), not the quickstart's Standard mode.

**Exercise:** A local kind/minikube cluster via the quickstart, installed in **Knative (serverless) mode** so canary works. Deploy the Capstone A model from its MLflow artifact as an InferenceService, then send 10 % canary traffic to a challenger. **Done when:** per-revision request counts show the split, and rollback is one command, documented in `m6/kserve.md`.

**Self-check:** How would you do shadow traffic here, given that KServe canary is a split?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M6.7 Test in production and continual learning
**3 h** · Prereqs: M6.5, M1.3 · Tested by: MLA, PMLE, 300, DMP, PGENL

**Skip test:** (1) Shadow vs canary vs A/B: what does each one risk? (2) When is interleaving better than A/B? (3) Bandits vs A/B: what's the trade-off? (4) Stateless vs stateful retraining.

- Continual learning: stateless vs stateful retraining, how often to update, the value of data freshness; test in production: shadow, A/B, canary, interleaving, multi-armed and contextual bandits. LLM-app rollouts → C8.4

**Resource:** DMLS ch9 "Continual Learning and Test in Production".

**Exercise:** A simulator comparing Thompson sampling vs a fixed 50/50 A/B for two models with known conversion rates, using the M1.3 toolkit for the A/B decision. Plus a freshness experiment: train on window W, test on each following week, and plot the decay. **Done when:** a regret plot and a retraining-cadence recommendation, backed by the decay plot, are committed.

**Self-check:** Why do bandits complicate offline analysis later?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M6.8 Monitoring and drift
**7 h** · Prereqs: M6.5, M1.3 · Tested by: MLA, PMLE, 300, DMP, PADS

**Skip test:** (1) Diagnostic Q13. (2) Covariate vs label vs concept shift, with an example of each from your capstone. (3) How do you monitor when labels arrive 30 days late? (4) What is feature-attribution drift?

- Failure causes; covariate, label and concept drift; detection with KS (continuous), chi-square (categorical), PSI (binned, both), Wasserstein (continuous), Jensen-Shannon, total-variation and L-infinity/Chebyshev distances (categorical or binned); choosing by data type, sample size and window; feature-attribution drift (SHAP → M9.2); delayed labels and proxy metrics; retraining triggers; alert fatigue; dashboards. LLM quality drift → C8.3

**Resource:** DMLS ch8 "Data Distribution Shifts and Monitoring". MLOps Zoomcamp module 5 + homework: Evidently metrics stored in PostgreSQL and shown in Grafana, with Prefect scheduling the batch metric job (current lessons 5.1–5.10, which pin Evidently 0.6.7 and link a 0.7 example; checked 2026-10-31). Optional depth (+6 h, not counted): Evidently [*Open-source AI Observability Course*](https://www.evidentlyai.com/ml-observability-course) modules 1, 2 and 4.

**Exercise:** An Evidently report and a Grafana dashboard on Capstone A's prediction logs. Implement PSI and KS (SciPy) yourself and compare them with Evidently. When the drifted-feature share exceeds a threshold, trigger the M6.4 pipeline. **Done when:** replaying a shifted month triggers exactly one retraining run, and your drift decisions agree with Evidently's for every feature, or the difference is explained.

**Self-check:** Why does KS flag everything at n = 1 M? What do you do about it?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M6.9 CI/CD/CT, IaC and ML testing
**6 h** · Prereqs: M6.4, M3.4, C8.4 · Tested by: MLA, PMLE, 300, DMP

**Skip test** (you will probably pass the CI half): pre-commit, unit vs integration tests, GitHub Actions matrix, Terraform plan/apply. Then answer: (1) What is a model test vs a code test? (2) What runs in CI vs CT? (3) What are behavioural tests (minimum functionality, invariance, directional)?

- ML test pyramid; behavioural model tests; quality gates against the champion; reproducible training; CI vs CD vs CT; IaC basics (the eval-gate pattern itself → C8.4)

**Resource:** MLOps Zoomcamp module 6 (unit and integration tests, linting, Makefiles, pre-commit, GitHub Actions CI/CD, Terraform) + homework. Made With ML "Testing" lesson, the code and models sections.

**Exercise:** GitHub Actions for Capstone A: lint + unit tests → data suite (M3.4) → train-on-sample smoke → behavioural tests → quality gate vs `@champion` → Docker build. Add Terraform for the artifact bucket (LocalStack, as in module 6, or a free-tier bucket). **Done when:** a PR that degrades the model fails CI with a readable message, and `terraform plan` is clean.

**Self-check:** Which tests are flaky because of randomness, and how did you pin them?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M6.10 MLOps capstone (Capstone A)
**15 h** · Prereqs: M6.1–M6.9, M4.3 · Tested by: all of MLA, PMLE, 300, DMA, DMP, PADS

**Skip test:** none. This is the integration artifact every vendor ML path re-platforms.

**Resource:** MLOps Zoomcamp project brief and peer-review criteria (the 07-project folder of the repo). Upgrade it: add the M4.3 PyTorch MLP as a challenger model.

**Exercise:** See [Capstone A](#capstone-a-tabular-mlops-system-m610). **Done when:** a clean clone plus one command brings up the stack, and a recorded demo shows the full loop: drift → retrain → gate → canary → promote.

**Self-check:** Score yourself on every Zoomcamp project criterion, and fix anything under the maximum.

- [ ] Capstone A built
- [ ] self-scored against the project criteria

## M7 Performance and scale

### M7.1 GPU performance model and profiling
**1.5 h** · Prereqs: M5.3 · Tested by: PGENL, PMLE (3.3)

**Skip test:** (1) Compute-bound vs memory-bound vs overhead-bound: how do you tell which from a profile? (2) Why does operator fusion help elementwise ops? (3) What does `torch.compile` mostly remove?

- Three regimes, arithmetic intensity, fusion, overhead from Python/dispatch, CUDA graphs and compilation; profiler basics (`torch.profiler`). LLM-specific prefill/decode → C10.1

**Resource:** Horace He, ["Making Deep Learning Go Brrrr From First Principles"](https://horace.io/brrr_intro.html) (whole).

**Exercise:** Profile one training step of your M5.1 model on Colab, eager vs `torch.compile`. **Done when:** the trace is saved, and a table classifies the top 5 ops by regime with the measured speedup.

**Self-check:** Why does a bigger batch help a memory-bound layer?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M7.2 Model optimisation
**6 h** · Prereqs: M7.1, M5.5, M4.5 · Tested by: PGENL (AWS removed it from MLA-C02, see note)

**Skip test:** (1) Dynamic vs static PTQ: which one for transformers and which for CNNs, and why? (2) What does calibration choose? (3) When is QAT worth it? (4) Structured vs unstructured pruning on real hardware. (5) Distillation loss with temperature.

- ONNX export; dynamic and static INT8 PTQ, calibration (MinMax/Entropy/Percentile), QOperator vs QDQ; QAT; FP8 (concept); pruning; knowledge distillation; mixed precision (fp16/bf16, loss scaling). LLM formats (GPTQ/AWQ/GGUF) → C10.2
- ONNX Runtime facts (as of 2026-10-31): dynamic quantization is recommended for RNNs/transformers and static for CNNs; int4 block-wise weight-only quantization needs opset 21+; GPU int8 runs via the TensorRT EP on Tensor-Core GPUs ([source](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html))
- Exam note: AWS removed "reducing model size" (pruning/compression) and SageMaker Neo from MLA-C02 ([comparison page](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/mla-02-comparison.html), as of 2026-10-31). Learn this module for engineering value and for PGENL

**Resource:** PyTorch ["Export a PyTorch model to ONNX"](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html). It uses `torch.onnx.export(..., dynamo=True)`, the recommended exporter (checked 2026-10-31). [ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html). Géron ch17 "Speeding up transformers" + the mixed-precision/quantization appendix. Optional (+6 h): [MIT 6.5940](https://hanlab.mit.edu/courses/2024-fall-65940) lectures Quantization I–II, Pruning I, Knowledge Distillation.

**Exercise:** Export the M5.5 encoder and the M4.5 CNN to ONNX. Apply dynamic INT8 to the encoder and static INT8 with calibration to the CNN. Optionally distil the encoder into a smaller student. **Done when:** an FP32 parity test passes (atol documented), and a table shows size, CPU p50/p95 latency and accuracy delta per variant. ONNX Runtime also has [Java bindings](https://onnxruntime.ai/docs/get-started/with-java.html), which you may want for a JVM service.

**Self-check:** Which variant would you ship for a CPU-only Spring service, and at what accuracy cost?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M7.3 Distributed training
**10 h** · Prereqs: M5.3, M4.3, C10.5 · Tested by: PMLE (3.3), DMP, PGENL

**Skip test:** (1) Diagnostic Q14. (2) Memory per parameter for Adam with mixed precision, and where it goes. (3) ZeRO-1/2/3: what does each shard? (4) When do TP, PP, CP and EP each become necessary? (5) What does `torchrun` give you?

- Scale up (bigger or GPU node) vs scale out (more nodes): scale up when one machine can hold the work (single-node libraries, DL that fits one node); scale out when the work is data-parallel and the data or model exceeds one node
- Memory anatomy (params, grads, optimizer states, activations), activation checkpointing; DDP (all-reduce, bucketing); FSDP2/ZeRO sharding; tensor, pipeline, context and expert parallelism; communication overhead; `torchrun` and fault tolerance. PyTorch's guidance: start with DDP, move to FSDP2 when the model exceeds one GPU, and add TP/PP only when FSDP2 hits limits ([overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html), as of 2026-10-31)

**Resource:** PyTorch tutorials: Distributed overview → Getting started with DDP → fault-tolerant training with `torchrun` → FSDP2 getting started. HF [*Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook) ([static page](https://nanotron-ultrascale-playbook.static.hf.space/index.html); it states "Reading time: 2-4 days", so read only these sections): "First Steps: Training on One GPU" (memory anatomy, activation recomputation, gradient accumulation), "Data Parallelism" (incl. ZeRO), "Tensor Parallelism", "Context Parallelism", "Pipeline Parallelism", "Expert Parallelism", "5D Parallelism in a Nutshell", "Finding the Best Training Configuration" (section titles checked 2026-10-31). Reference only: Stas Bekman's [*ML Engineering* open book](https://github.com/stas00/ml-engineering) (training and debugging parts) for when something breaks.

**Exercise:** DDP with your M4.3 trainer on CPU (gloo, 2–4 processes). Then DDP vs FSDP2 on the M5.3 model with 2 GPUs (Kaggle dual GPU [verify], or rented). Write `m7/memcalc.py`, which predicts per-GPU memory for a config. **Done when:** DDP matches single-process loss at the same global batch (seeded), and memcalc is within about 20 % of `torch.cuda.max_memory_allocated`.

**Self-check:** Using your C10.5 spreadsheet: what would 1B tokens on your config cost?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M7.4 Ray (conditional)
**5 h** · Prereqs: M4.4, M7.3 · Required before DBX-MLP and GCP-PMLE · Tested by: PMLE, DMP

**Skip test:** (1) Ray tasks vs actors vs object store. (2) What does Ray Data stream that pandas can't? (3) `ScalingConfig` in Ray Train. (4) ASHA vs median pruning.

- Core (tasks, actors, objects); Ray Data streaming preprocessing and batch inference; Ray Train TorchTrainer; Ray Tune (search algorithms incl. an Optuna searcher, ASHA scheduler); Ray Serve (skim)

**Resource:** [Ray docs](https://docs.ray.io/en/latest/ray-overview/getting-started.html) (Ray 2.59 as of 2026-10-31): Getting started; "Scale ML workloads" quickstarts for Data, Train, Tune and Serve; Ray Train "Get started with PyTorch"; Ray Tune key concepts. Serve: skim only.

**Exercise:** Ray Tune with an Optuna searcher + ASHA on the M4.4 problem locally, plus Ray Data batch inference over the taxi Parquet. **Done when:** ASHA stops trials early (visible in results), and the batch job uses all local cores (utilisation evidence).

**Self-check:** When would you pick Ray over Spark for preprocessing?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

## M8 Recsys and time series

### M8.1 Recommender systems
**6 h** · Prereqs: M5.8, M2.4, C7.5 · Tested by: PMLE (implied), MLA (light)

**Skip test:** (1) Why do recommenders have retrieval and ranking stages? (2) Content-based vs collaborative filtering and cold start. (3) How is a softmax/two-tower model trained with implicit feedback? (4) What does re-ranking add?

- Candidate generation (content-based, CF/matrix factorisation, two-tower/DNN softmax), scoring, re-ranking (freshness, diversity, fairness); implicit feedback, negative sampling, position bias; offline/online × retrieval/ranking design (metrics → C7.5)

**Resource:** Google [*Recommendation Systems*](https://developers.google.com/machine-learning/recommendation) course ("Estimated Time: 4 hours", stated), all sections. Do its exercises in PyTorch instead of the provided framework. Eugene Yan, ["System Design for Recommendations and Search"](https://eugeneyan.com/writing/system-design-for-discovery/) (whole).

**Exercise:** A two-stage MovieLens recommender: MF or two-tower retrieval (PyTorch) with ANN lookup in pgvector (C7.3), then a LightGBM ranker. Evaluate on a time-based split. **Done when:** retrieval recall@100 and ranking nDCG@10 beat a popularity baseline, and the results table is in MLflow.

**Self-check:** Which features are only available at ranking time, and why?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M8.2 Time-series forecasting
**11 h** · Prereqs: M2.3, M1.3 · Tested by: PMLE (3.1), ADS, PADS

**Skip test:** (1) Why can't you use k-fold CV on time series, and what do you use instead? (2) Seasonal naive vs drift baselines. (3) What do ACF and PACF tell you for ARIMA orders? (4) MASE: why scale-free? (5) ETS vs ARIMA vs gradient boosting with lags: when does each win?

- Graphics, decomposition (STL), features; baselines (mean, naive, seasonal naive, drift); point and distributional accuracy, MASE, rolling-origin CV; exponential smoothing (ETS); stationarity, differencing, ARIMA/SARIMA; prediction intervals; practical issues (missing values, outliers, calendar effects); residual-based anomaly detection; ML forecasters with lag features; NN and foundation forecasters (skim)

**Resource:** [*Forecasting: Principles and Practice, the Pythonic Way*](https://otexts.com/fpppy/) (free; uses Nixtla `statsforecast`; "last updated October 31, 2026"): ch1–5 (getting started, graphics, decomposition, features, the forecaster's toolbox incl. evaluation and time-series CV), ch8 (exponential smoothing), ch9 (ARIMA), ch13 (practical issues). Skim ch14 (neural networks) and ch15 (foundation models).

**Exercise:** Forecast daily taxi trips per borough 14 days ahead: seasonal naive, ETS, ARIMA (statsforecast), and LightGBM with lag/calendar features, using rolling-origin CV. Flag anomalies from residuals. **Done when:** a MASE table per model per borough exists, the best model beats seasonal naive or the reason is documented, and the anomaly flags are checked against known holidays.

**Self-check:** Why can a model with lower RMSE have worse prediction intervals?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

## M9 Responsible ML

RAI principles, regulation, model cards and lineage are in → C9.5. This block adds only measurement.

### M9.1 Fairness
**3 h** · Prereqs: M2.2, M6.3 · Tested by: MLA, PMLE, 300

**Skip test:** (1) Name four bias types and the pipeline stage each enters at. (2) Demographic parity vs equalized odds vs equality of opportunity. (3) Why can't you satisfy all of them at once? (4) Pre-, in- and post-processing mitigations.

- Bias types (reporting, historical, automation, selection, group attribution, implicit), fairness metrics, metric incompatibility, mitigation (reweighting, constraints, per-group thresholds), slice monitoring
- Text and image datasets: the same metrics need a group label first, so measure representation per group (from metadata or annotation), per-group error rates on a labelled slice, and label-imbalance per group; for text also counterfactual token swaps (change only the group term, compare outputs)

**Resource:** Google MLCC ["Fairness"](https://developers.google.com/machine-learning/crash-course/fairness) module (110 min stated).

**Exercise:** On Capstone A, using a sensitive or proxy attribute (or the UCI Adult dataset if yours has none), compute demographic-parity and equal-opportunity differences yourself in pandas. Then apply per-group threshold post-processing. **Done when:** a fairness report shows before/after metrics and the accuracy cost, and it runs as an optional CI check.

**Self-check:** Which fairness metric fits your capstone's harm model, and why?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M9.2 Explainability
**4 h** · Prereqs: M2.4, M6.8 · Tested by: MLA, PMLE, 300, DMP

**Skip test:** (1) Diagnostic Q15. (2) Why does permutation importance mislead with correlated features? (3) PDP vs ICE vs ALE. (4) TreeSHAP vs KernelSHAP. (5) When would you choose an interpretable-by-design model instead?

- Global vs local, model-agnostic vs specific; permutation importance; PDP/ICE/ALE; Shapley values and SHAP; LIME; counterfactual explanations ("smallest feature change that flips the prediction"); attributions are correlational, not causal (causal-effect estimation is out of scope; recognise the term); interpretability requirements driving model choice; feature-attribution drift (feeds M6.8)

**Resource:** Kaggle Learn [*Machine Learning Explainability*](https://www.kaggle.com/learn/machine-learning-explainability): permutation importance, partial plots, SHAP values, advanced uses of SHAP values (lesson list and "4 h" from a search snippet [unconfirmed]; the page is JS-rendered). Reference: Molnar [*Interpretable Machine Learning*](https://christophm.github.io/interpretable-ml-book/) (3rd ed.) chapters on permutation feature importance, PDP, ALE, Shapley values, SHAP, LIME and counterfactual explanations, only when an output puzzles you.

**Exercise:** A SHAP report for the Capstone A champion, a PFI-vs-impurity disagreement demo using a duplicated correlated feature, and an attribution-drift metric (mean |SHAP| per feature, reference vs current window) added to the M6.8 dashboard. **Done when:** all three are committed and the drift metric shows up in Grafana.

**Self-check:** What would you tell a regulator your top feature "means"?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

## M10 ML system design

### M10.1 Framing and requirements
**4 h** · Prereqs: M6.1 · Tested by: PMLE, MLA (scenario framing)

**Skip test:** for a "reduce manual document routing" problem, state the business metric, ML objective, decoupled objectives, a non-ML baseline and three constraints in 10 minutes.

- When to use ML; research vs production; requirements (reliability, scalability, maintainability, adaptability); framing (task type, objective functions, decoupling objectives); the human side: UX consistency, team structure, responsible AI in practice (product discovery for LLM apps → C13.1)

**Resource:** DMLS ch1 "Overview of ML Systems", ch2 "Introduction to ML Systems Design", ch11 "The Human Side of Machine Learning".

**Exercise:** A 2-page design doc for Capstone B, or a non-proprietary work idea such as document routing in a publishing workflow. **Done when:** it has a measurable success metric, a non-ML baseline, a kill criterion and a data-availability check.

**Self-check:** What would make you not use ML for this?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

### M10.2 Design practice
**10 h** · Prereqs: M10.1, M6, M8.1 · Tested by: PMLE, MLA (scenario framing)

**Skip test:** run one 45-minute mock (below). If you cover all 7 steps with concrete numbers, do only 2 more mocks.

- A 7-step design framework (requirements → ML framing → data → model → evaluation → serving → monitoring); back-of-envelope sizing; trade-off articulation

**Resource:** Aminian & Xu, *Machine Learning System Design Interview* (ByteByteGo; paid, price [unconfirmed], no official free URL; "7-step framework, 10 case studies" is from search snippets [unconfirmed]): the framework chapter plus 4 of the 10 case studies (choose recommendation, ads/click prediction, search/visual search and harmful-content detection, by whatever titles the edition uses). The remaining 6 cases are optional (+8 h). **Free fallback:** the Evidently [ML/LLM system design database](https://www.evidentlyai.com/ml-system-design) ("800 case studies from 150+ companies", as of 2026-10-31). Read about 10 write-ups matching recsys, search/ranking, fraud, ads and forecasting. Self-test with Chip Huyen's free [*ML Interviews Book*](https://huyenchip.com/ml-interviews-book/) Part II questions.

**Exercise:** Four timed 45-minute mock designs. Use Claude Code as the interviewer with a rubric built from the 7 steps, and record yourself. Compare each one with the book's solution or a matching Evidently case. **Done when:** four written designs with self-scores are in `m10/`, and at least 30 Part II questions are answered.

**Self-check:** Which step do you consistently under-specify?

- [ ] reading / skipped via skip test
- [ ] exercise + self-check

---

## Capstone projects

Capstone A is required (built in M6.10). Pick **one** of B, C or D for the essential plan (about 8 h of integration on top of module exercises you have already done). Each vendor ML path re-platforms these same projects onto its managed services, so you learn the platform, not a new problem.

### Capstone A: Tabular MLOps system (M6.10)
Capstone dataset → Great-Expectations-style suite (M3.4) → Feast features (M3.5) → Optuna-tuned XGBoost champion + PyTorch MLP challenger (M2.4, M4.3–M4.4) → MLflow registry with aliases (M6.2) → orchestrated pipeline (M6.4) → FastAPI/Docker online + batch scoring (M6.5) → KServe canary (M6.6) → Evidently/Grafana drift with retraining trigger (M6.8) → GitHub Actions CI/CD/CT + Terraform (M6.9) → fairness + SHAP reports (M9, added later).
**Re-platform:** AWS-MLA (SageMaker Pipelines, Feature Store, Model Registry, Model Monitor, Clarify) · AZ-AI300 (AzureML pipelines, managed endpoints, drift monitors, RAI dashboard) · GCP-PMLE (Agent Platform Pipelines, Feature Store, Model Monitoring, explainability) · DBX-MLA / DBX-MLP (feature tables, MLflow in Unity Catalog, Lakeflow, Data Quality Monitoring) · NV-ADS (cuDF / cuML / GPU XGBoost rerun).
**Done when:** the M6.10 criteria are met.
- [ ] Capstone A

### Capstone B: Two-stage recommender
M8.1 recommender + M5.8 two-tower/embedder + Feast user/item features (M3.5) + KServe InferenceGraph retrieve → rank (M6.6) + offline nDCG and an online interleaving simulation (M6.7). With M7.4: Ray Data for candidate precomputation.
**Re-platform:** GCP-PMLE · DBX-MLP (Ray on Spark, feature serving) · AWS-MLA.
**Done when:** one request returns ranked items in under 100 ms p95 locally, and the design doc from M10.1 is updated with measured results.
- [ ] Capstone B

### Capstone C: Fine-tune → optimise → serve a transformer
M5.5/M5.6 fine-tuned encoder → ONNX INT8 (M7.2) → KServe canary vs the FP32 model (M6.6) → text-drift monitoring (M6.8) → benchmark sheet (M5.9) → cost comparison vs your C12.3 LLM baseline.
**Re-platform:** NV-GENL (TensorRT-LLM / Dynamo-Triton, Nsight) · AZ-AI300 (Foundry fine-tuning, managed endpoints) · AWS-MLA (SageMaker training + endpoints) · GCP-PMLE.
**Done when:** the INT8 model is served behind a canary with an accuracy/latency/cost table, and rollback is tested.
- [ ] Capstone C

### Capstone D: Batch demand forecasting
M8.2 forecasters → nightly batch pipeline (M6.4) with data contracts (M3.4) → forecast store (Parquet/Postgres) → forecast-accuracy monitoring (MASE on matured actuals) with retraining trigger (M6.8) → backfill command. With M3.6: per-series training in Spark.
**Re-platform:** GCP-PMLE (BigQuery ML) · NV-ADS (cuDF / cuML) · DBX-MLA (pandas Function API per-series training) · AWS-MLA.
**Done when:** a 30-day backfill runs idempotently, and an injected regime change triggers retraining.
- [ ] Capstone D

## Coverage matrix

Every module in this guide has a learning item (resource), a hands-on exercise with a "done when" criterion, a self-check and a skip test (M6.10 has none by design). The "Exams" column lists the vendor ML exams that test the concept.

| ID | h | Learning item | Exercise artifact | Capstone | Exams |
|---|---|---|---|---|---|
| M0.1 | 4 | Géron tool notebooks | `m0/taxi_eda.py` | all | — |
| M0.2 | 2 | setup checklist | lab skeleton + Colab bootstrap | all | — |
| M1.1 | 3 | 3B1B Linear Algebra | PCA via eigendecomposition | — | — |
| M1.2 | 1.5 | 3B1B NN ch1–4 | gradient checker | — | — |
| M1.3 | 4.5 | StatQuest stats | `stats_toolkit.py` | A, B | — |
| M2.1 | 9 | Géron ch1–2; MLCC data modules | taxi `train.py` | A | MLA PMLE DMA ADS |
| M2.2 | 6 | Géron ch3; MLCC Classification | `metrics_report.py` | A | MLA PMLE DMA ADS |
| M2.3 | 6 | Géron ch4; StatQuest | NumPy logistic GD | — | MLA PMLE DMA ADS |
| M2.4 | 7 | Géron ch5–6; StatQuest | model benchmark table | A | MLA PMLE DMA ADS |
| M2.5 | 5.5 | Géron ch7–8 | anomaly detector | A | MLA PMLE DMA ADS |
| M2.6 | 3 | Kaggle Intermediate ML | `leakage_lab.py` | A | MLA PMLE DMA ADS |
| M3.1 | 3 | DMLS ch4 | imbalance comparison + reservoir | A | MLA PMLE DMA ADS PADS |
| M3.2 | 5 | DMLS ch5; Kaggle FE | feature ablation | A | MLA PMLE DMA ADS PADS |
| M3.3 | 1.5 | DMLS ch3 | partitioned Parquet | A, D | MLA PMLE |
| M3.4 | 3 | Made With ML data testing | expectation suite | A, D | MLA PMLE 300 ADS |
| M3.5 | 4 | Feast; Rules of ML | Feast repo + skew test | A, B | MLA PMLE DMA DMP |
| M3.6 | 5* | Spark MLlib guide | PySpark pipeline | D | DMA DMP PMLE MLA ADS |
| M4.1 | 12 | Karpathy L1–L3 | micrograd + makemore | — | (PGENL) |
| M4.2 | 9 | Karpathy L4–L5 | `diagnostics.py` | — | (PGENL) |
| M4.3 | 11 | Géron ch9–11 | `trainer.py` | A | MLA PMLE DMA DMP PGENL |
| M4.4 | 4 | Tuning Playbook; Optuna | Optuna study + bug log | A | MLA PMLE DMA DMP PGENL |
| M4.5 | 7 | Géron ch12, 16 | CLIP vs ResNet vs ViT | C | PMLE GENM PGENL |
| M4.6 | 1.5 | Géron ch13 skim | LSTM vs CNN vs naive | D | — |
| M5.1 | 7 | Karpathy "build GPT" | nanoGPT + positional variant | — | PGENL AGENL |
| M5.2 | 5 | Karpathy tokenizer; HF ch6 | BPE on Java corpus | — | PGENL AGENL |
| M5.3 | 11 | Karpathy GPT-2 repro | optimisation throughput table | — | PGENL AGENL |
| M5.4 | 3 | HF ch1–2, ch5 | streamed dataset + card | C | PGENL |
| M5.5 | 7 | HF ch7 | NER/classifier vs LLM baseline | C | PGENL |
| M5.6 | 6 | HF ch3 | 3-way training parity | C | PGENL MLA 300 |
| M5.7 | 8 | HF ch11–12 | TRL SFT + DPO | — | PGENL MLA 300 |
| M5.8 | 4 | Hands-On LLMs ch10 | fine-tuned embedder | B | PGENL MLA 300 |
| M5.9 | 3 | AI Eng ch3 LM metrics; lm-eval | perplexity + harness run | C | PGENL AGENL |
| M6.1 | 3 | Google MLOps levels; Sculley; ZC m1 | maturity doc | A | MLA PMLE 300 DMA DMP PADS |
| M6.2 | 6 | ZC m2 | `promote.py` | A | MLA PMLE 300 DMA DMP PADS |
| M6.3 | 4 | DMLS ch6 | `offline_eval.py` | A | MLA PMLE 300 DMA DMP PADS |
| M6.4 | 6 | ZC m3 | Capstone A DAG | A, D | MLA PMLE 300 DMA DMP PADS |
| M6.5 | 5 | ZC m4; DMLS ch7 | FastAPI service + batch CLI | A | MLA PMLE 300 DMP PGENL |
| M6.6 | 4 | KServe docs | InferenceService + canary | A, B, C | MLA PMLE 300 DMP PGENL |
| M6.7 | 3 | DMLS ch9 | bandit sim + cadence | A, B | MLA PMLE 300 DMP PGENL |
| M6.8 | 7 | DMLS ch8; ZC m5 | drift monitor + trigger | A, C, D | MLA PMLE 300 DMP PADS |
| M6.9 | 6 | ZC m6; Made With ML testing | CI/CD/CT workflow | A | MLA PMLE 300 DMP |
| M6.10 | 15 | ZC project | Capstone A | A | all ML exams |
| M7.1 | 1.5 | Horace He | profile table | C | PGENL PMLE |
| M7.2 | 6 | ONNX export + ORT quant; Géron ch17 | INT8 benchmark | C | PGENL |
| M7.3 | 10 | PyTorch dist; Ultra-Scale | DDP/FSDP2 + `memcalc.py` | — | PMLE DMP PGENL |
| M7.4 | 5* | Ray docs | Tune + Ray Data job | B | PMLE DMP |
| M8.1 | 6 | Google Recsys; Eugene Yan | two-stage recommender | B | PMLE MLA |
| M8.2 | 11 | FPP Pythonic | forecasting CV table | D | PMLE ADS PADS |
| M9.1 | 3 | MLCC Fairness | fairness report | A | MLA PMLE 300 |
| M9.2 | 4 | Kaggle Explainability; Molnar | SHAP + attribution drift | A | MLA PMLE 300 DMP |
| M10.1 | 4 | DMLS ch1–2, 11 | design doc | B | PMLE MLA |
| M10.2 | 10 | Aminian & Xu / Evidently DB; Interviews Book | 4 mock designs | — | PMLE MLA |

\* conditional (M3.6, M7.4).

## Where this core leads (vendor ML paths)

Each path is a delta on this core. It also needs its AI-repo `<PREFIX>.1` platform module unless you already took that AI path. Exam facts (format, price, domains) live only in the path files.

| Path file | Certification | Extra prerequisites | Approx. minimum core subset for an exam-first plan |
|---|---|---|---|
| [paths/aws/mla-c02.md](paths/aws/mla-c02.md) | AWS Certified Machine Learning Engineer – Associate (MLA-C02). Beta now; GA delivery from 2027-01-14 per the [AWS blog](https://aws.amazon.com/blogs/training-and-certification/september-2026-new-offerings/), as of 2026-10-31 | [AWS-AIP.1](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/aws/aip-c01.md) | M0, M1.2–M1.3, M2, M3.1–M3.5, M4.3–M4.4, M5.4, M5.6–M5.8, M6, M8.1, M9, M10 (about 187 h) |
| [paths/azure/ai-300.md](paths/azure/ai-300.md) | Microsoft Certified: Machine Learning Operations Engineer Associate (AI-300) | [AZ-AI103.1](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/azure/ai-103.md) | M0, M1.2–M1.3, M2.1–M2.2, M3.4, M4.3–M4.4, M5.4, M5.6–M5.8, M6, M9 (about 132 h) |
| [paths/google/professional-ml-engineer.md](paths/google/professional-ml-engineer.md) | Google Cloud Professional Machine Learning Engineer (June-2026 guide; the guide says it "does not directly assess coding skills", [guide](https://services.google.com/fh/files/misc/professional_machine_learning_engineer_exam_guide_english_new.pdf), as of 2026-10-31) | [GCP-PAA.1](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/google/professional-agentic-architect.md); M3.6, M7.4 | M0, M1.2–M1.3, M2, M3, M4.3–M4.5, M6, M7.1, M7.3–M7.4, M8, M9, M10 (about 206 h) |
| [paths/databricks/ml-associate.md](paths/databricks/ml-associate.md) | Databricks Certified Machine Learning Associate | [DBX-GAI.1](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/databricks/genai-engineer-associate.md); M3.6 | M0, M1.2–M1.3, M2, M3.1–M3.2, M3.5–M3.6, M4.3–M4.4, M6.1–M6.4 (about 100 h) |
| [paths/databricks/ml-professional.md](paths/databricks/ml-professional.md) | Databricks Certified Machine Learning Professional | DBX-MLA; M7.4 | DBX-MLA subset + M6.5–M6.10, M7.3–M7.4, M9.2 (about 159 h) |
| [paths/nvidia/ncp-genl.md](paths/nvidia/ncp-genl.md) | NVIDIA NCP-GENL (Generative AI LLMs, Professional) | [NV-AMAD.1](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/nvidia/nca-amad.md) | M0, M1, M4.1–M4.5, M5, M6.5–M6.7, M7.1–M7.3 (about 142 h) |
| [paths/nvidia/nca-ads-ncp-ads.md](paths/nvidia/nca-ads-ncp-ads.md) | NVIDIA NCA-ADS / NCP-ADS (Accelerated Data Science), optional and lowest priority | NV-AMAD.1 (NGC only); M3.6 | M0, M1.3, M2, M3.1–M3.2, M3.4, M3.6, M6.1–M6.4, M6.8, M8.2 (about 100 h) |

The subset hours are my own sums of this file's module hours. Use them only for an exam-first plan. They do not include every in-core prerequisite: for example, M4.3 assumes M4.1–M4.2 and M3.5 assumes M3.3. Either pass those modules' skip tests or add their hours. Every path's capstone re-platforms Capstone A (M6.10). The DBX-MLA, NV-GENL and NV-ADS subsets stop short of M6.10, so before those capstones either add M6.5–M6.10 (40 h) or re-platform the partial pipeline you built.

## Schedule

### Default: about 7 h/week, about 42 weeks (290 h)

| Weeks | Block | Hours | Cumulative |
|---|---|---|---|
| 1 | Diagnostic + M0 | 7 | 7 |
| 2–3 | M1 (M1.3 first) | 9 | 16 |
| 3–6 | M2.1–M2.3 | 21 | 37 |
| 6–8 | M2.4–M2.6 | 15.5 | 52.5 |
| 8–10 | M3.1–M3.4 | 12.5 | 65 |
| 10–13 | M4.1–M4.2 | 21 | 86 |
| 13–16 | M4.3–M4.6 | 23.5 | 109.5 |
| 16–19 | M5.1–M5.3 | 23 | 132.5 |
| 19–24 | M5.4–M5.9 | 31 | 163.5 |
| 24–27 | M3.5, M6.1–M6.4 | 23 | 186.5 |
| 27–31 | M6.5–M6.9 | 25 | 211.5 |
| 31–33 | M6.10 (Capstone A) | 15 | 226.5 |
| 33–35 | M7.1–M7.3 | 17.5 | 244 |
| 35–38 | M8 | 17 | 261 |
| 38 | M9 | 7 | 268 |
| 38–41 | M10 | 14 | 282 |
| 41–42 | Capstone B, C or D | 8 | **290** |

Conditional M3.6 + M7.4 add 10 h (about 1.5 weeks) before the Databricks or Google ML paths. **Production-first variant:** if an AWS/Azure ML exam comes first, run Phase 3 (M3.5, M6.1–M6.9) right after M3.4 using a scikit-learn champion, then do M4.3 before M6.10. Total hours are unchanged.

### Compressed options
- **Same scope at about 14 h/week:** about 21 weeks. Pair one theory block (M4/M5) with one MLOps block (M6) each week, using the parallel track.
- **Exam-first:** do only the subset for your first path in the table above (about 100–206 h), then backfill the rest before your second path. Pass skip tests aggressively. You will probably pass M0.2, M3.3, the service half of M6.5 and the CI half of M6.9.

## Tool setup

One tool per job, shared with the AI core (C0.2).

| Job | Tool | Notes |
|---|---|---|
| Env | uv + Python 3.12 + Jupyter | Same project style as C0.2; add the extras below |
| Classical ML | scikit-learn, XGBoost, LightGBM, imbalanced-learn, SciPy | M2–M3 |
| DL | PyTorch (+ torchvision) | Géron 2025, Karpathy and HF all use it |
| HF stack | `transformers`, `datasets`, `tokenizers`, `peft`, `trl`, `accelerate`, `evaluate`, `sentence-transformers` | M5. Fine-tuning tool for the ML core = Trainer + TRL + Accelerate |
| HPO | Optuna (SQLite storage) | M4.4; Ray Tune wraps it in M7.4 |
| Tracking / registry | MLflow, the same server as C0.2 | Classic tracking/registry here; GenAI tracing stays C8.3 |
| Feature store | Feast (local provider) | M3.5 |
| Data validation | Great Expectations-style suites (Made With ML) or pytest | M3.4 [verify GE API version] |
| Orchestration | Prefect (module 3's lesson is tool-agnostic; module 5 schedules with Prefect) | M6.4, M6.8. Only one orchestrator; skip the Mage lesson 4.6 |
| Serving | FastAPI + Docker; KServe on kind/minikube | M6.5–M6.6. Not TorchServe (archived) |
| Optimisation | ONNX + ONNX Runtime | M7.2 |
| Drift | Evidently + PostgreSQL + Grafana (Zoomcamp module 5 stack) | M6.8 |
| Distributed | PyTorch DDP/FSDP2 + `torchrun`; Ray (M7.4); Spark (M3.6 only) | |
| Forecasting | `statsforecast` | M8.2 (FPP uses it) |
| Explainability | `shap` | M9.2 |
| Vector store | pgvector from C7.3 | M5.8, M8.1 |
| CI / IaC | GitHub Actions, pytest, pre-commit, Terraform | M6.9 |
| GPU | Colab / Kaggle free tier; rented GPU only for M5.3 and optional M7.3 | Budget ≤ about $50 |

## Skipped on purpose

| Item | Reason |
|---|---|
| Andrew Ng *ML* and *Deep Learning* Specializations | 95 h + 129 h, TF/Keras, DL last updated 2021; Géron + Karpathy cover the same ground faster in PyTorch |
| DeepLearning.AI / Imperial math specializations | 58–94 h for what 3B1B + StatQuest give in about 9 h |
| CS229, CS224n, CS231n lecture series | Research depth beyond an MLE's needs; CS231n notes kept only as an M4.5 backup |
| Stanford CS336 | Gold standard for LLM systems, but about 150–250 h (my estimate) plus GPU costs; out of scope unless you target LLM-training roles |
| fast.ai Part 1 / Part 2 | Overlaps Karpathy + Géron + HF; built on the high-level fastai library |
| d2l.ai | Reference-grade overlap; parts of its recsys chapter are MXNet-only (e.g. section 21.3) |
| *Understanding Deep Learning*, *Fairness and ML*, mlsysbook.ai | Good books but not needed; the assigned resources cover the tested depth |
| Made With ML full course | Pick-one overlap with the Zoomcamp; only its testing lessons are used (M3.4, M6.9) |
| ML Zoomcamp, Data Engineering Zoomcamp | Redundant with Géron + MLOps Zoomcamp, and with your backend data experience |
| Full Stack Deep Learning 2022, DLAI *ML in Production*, CS329S notes | Dated or superseded by DMLS (CS329S notes became DMLS) |
| TorchServe | Archived Aug 7, 2025, with no security patches ([repo](https://github.com/pytorch/serve)) |
| Géron ch14–15, ch18 (except optional diffusion), ch19 | Covered by Karpathy/HF; RL is not tested by the target exams |
| HF LLM Course ch4, ch8–10 | Sharing/help/demos are trivial for you; dataset-curation concepts are in → C11.3 |
| Evidently course module 3 (LLM monitoring) | LLM monitoring → C8.3 |
| HF Community CV course, `picotron`, `eugeneyan/applied-ml` | Specialist or reference-only; not needed for the capstones or exams |
| Exam dumps | Never |

## Sources

All accessed or confirmed on or before 2026-10-31. Items marked [unconfirmed] or [verify] in the text were not confirmed on a rendered page.

- AI core: https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/core-llm-app-engineering.md
- Géron `handson-mlp`: https://github.com/ageron/handson-mlp (book price and date [unconfirmed])
- 3Blue1Brown: https://www.3blue1brown.com/topics/linear-algebra · https://www.3blue1brown.com/topics/neural-networks
- StatQuest index: https://statquest.org/video_index.html
- *Mathematics for ML*: https://mml-book.github.io/
- Google MLCC: https://developers.google.com/machine-learning/crash-course · durations: https://developers.google.com/machine-learning/crash-course/prereqs-and-prework
- Kaggle Learn: https://www.kaggle.com/learn/intermediate-machine-learning · https://www.kaggle.com/learn/feature-engineering · https://www.kaggle.com/learn/machine-learning-explainability
- DMLS chapters: https://github.com/chiphuyen/dmls-book
- *Rules of ML*: https://developers.google.com/machine-learning/guides/rules-of-ml
- Made With ML: https://madewithml.com/courses/mlops/
- Feast: https://docs.feast.dev/
- Spark MLlib: https://spark.apache.org/docs/latest/ml-guide.html
- Karpathy: https://karpathy.ai/zero-to-hero.html · https://github.com/karpathy/build-nanogpt · https://github.com/karpathy/nanochat
- DL Tuning Playbook: https://github.com/google-research/tuning_playbook
- Optuna tutorial: https://optuna.readthedocs.io/en/stable/tutorial/index.html
- CS231n notes: https://cs231n.github.io/
- HF LLM Course: https://huggingface.co/learn/llm-course/chapter1/1 (ch3, ch5, ch6, ch7, ch11, ch12 under the same path)
- *Hands-On LLMs* notebooks: https://github.com/handsOnLLM/Hands-On-Large-Language-Models
- *AI Engineering* ToC: https://github.com/chiphuyen/aie-book
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- Google MLOps levels: https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- Sculley et al. 2015: https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
- MLOps Zoomcamp: https://github.com/DataTalksClub/mlops-zoomcamp · module 3: https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/03-orchestration
- KServe: https://kserve.github.io/website/
- Evidently: https://www.evidentlyai.com/ml-observability-course · https://www.evidentlyai.com/ml-system-design
- Horace He: https://horace.io/brrr_intro.html
- PyTorch ONNX export: https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- MIT 6.5940: https://hanlab.mit.edu/courses/2024-fall-65940
- PyTorch distributed: https://docs.pytorch.org/tutorials/beginner/dist_overview.html
- Ultra-Scale Playbook: https://huggingface.co/spaces/nanotron/ultrascale-playbook · static page with TOC: https://nanotron-ultrascale-playbook.static.hf.space/index.html
- ONNX Runtime Java: https://onnxruntime.ai/docs/get-started/with-java.html
- KServe canary (serverless mode only): https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary
- MLOps Zoomcamp modules 3–5 lesson lists: https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/03-orchestration · https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/04-deployment · https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/05-monitoring
- Stas Bekman *ML Engineering*: https://github.com/stas00/ml-engineering
- Ray: https://docs.ray.io/en/latest/ray-overview/getting-started.html
- Google Recommendation Systems: https://developers.google.com/machine-learning/recommendation
- Eugene Yan: https://eugeneyan.com/writing/system-design-for-discovery/
- FPP Pythonic: https://otexts.com/fpppy/
- Molnar: https://christophm.github.io/interpretable-ml-book/
- ML Interviews Book: https://huyenchip.com/ml-interviews-book/
- Aminian & Xu: ByteByteGo, no official free URL; edition and price [unconfirmed]
- TorchServe archive notice: https://github.com/pytorch/serve
- MLflow 3 launch: https://mlflow.org/blog/mlflow-3-launch/
- AWS MLA-C02 comparison: https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/mla-02-comparison.html · GA date: https://aws.amazon.com/blogs/training-and-certification/september-2026-new-offerings/
- Google PMLE guide (June 2026): https://services.google.com/fh/files/misc/professional_machine_learning_engineer_exam_guide_english_new.pdf
