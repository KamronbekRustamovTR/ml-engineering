# Databricks Certified Machine Learning Professional: study path (DBX-MLP)

**Who this is for / prerequisites.** A **delta on [DBX-MLA](ml-associate.md)** (finish it, incl. capstone DBX-MLA.10) for a senior Java backend engineer and Claude Code power user; Python-only exercises. Also needed: ML core M3.5, M3.6, M4.4, M6.1–M6.10, M7.3, M7.4, and, from the [Databricks AI path](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/databricks/genai-engineer-associate.md): **if you took the Databricks AI path, you already have these. Otherwise do just these modules from it: DBX-GAI.1 Platform foundations (6 h)**, which you already did before DBX-MLA. No other AI-path module is needed. Anything DBX-MLA or DBX-GAI.1 taught is cited, never repeated. New here: distributed tuning (Optuna `MlflowSparkStudy`, Ray on Spark), online features and feature serving, Lakeflow pipelines, MLOps Stacks and bundles for ML, Data Quality Monitoring (exam: Lakehouse Monitoring), rollout on Model Serving. A test-taker guide calls it "one of the more challenging Databricks certifications" ([Cole][cole]); plan a two-week classic-compute lab sprint.

## Contents

- [DBX-MLP.0 Exam at a glance and refresh](#dbx-mlp0-exam-at-a-glance-and-refresh)
- [Modules DBX-MLP.1–DBX-MLP.8](#modules)
- [DBX-MLP.9 Capstone](#dbx-mlp9-capstone-production-grade-m610-on-databricks)
- [DBX-MLP.10 Practice, readiness, exam day](#dbx-mlp10-practice-readiness-exam-day)
- [Coverage matrix](#coverage-matrix)
- [Schedule](#schedule)
- [Combining with other paths](#combining-with-other-paths)
- [Gaps for the core](#gaps-for-the-core)
- [Skipped on purpose](#skipped-on-purpose)
- [Sources](#sources)

---

## DBX-MLP.0 Exam at a glance and refresh

⏱ 1.5 h · prereqs: DBX-MLA

### Exam at a glance (as of 2026-10-31)

| Item | Value | Source |
|---|---|---|
| Name | Databricks Certified Machine Learning Professional | [cert page][mlp-page] |
| Status | Active. The current PDF (file name "august", posted Oct 2026, adds an "AI Prep Guide" section) still reads "Exam Published September 30, 2025 / Exam Guide Updated September 30, 2025". It replaced an April 2024 syllabus per a third party ([Cole][cole]; not confirmed by Databricks). | [guide PDF][mlp-pdf] |
| Questions | 59 scored multiple-choice, plus "up to 10 unscored items" | [guide PDF][mlp-pdf] |
| Duration | 120 min (time for unscored items is included) | [guide PDF][mlp-pdf] |
| Price | USD 200 plus applicable taxes. Each retake costs the full fee. | [guide PDF][mlp-pdf], [FAQ][faq] |
| Delivery | Online proctored or test center; no test aids | [guide PDF][mlp-pdf] |
| Language | **English only** | [cert page][mlp-page] |
| Passing score | **Not published** (FAQ). The ~70% figure is third-party and unconfirmed. | [FAQ][faq] |
| Retake | 14-day wait | [FAQ][faq] |
| Eligibility | No prerequisite. **1+ year** of hands-on experience recommended. | [cert page][mlp-page] |
| Validity | 2 years. Recertify by taking the full current exam. | [guide PDF][mlp-pdf], [FAQ][faq] |
| Weights | Model Development 44% · MLOps 44% · Model Deployment 12% | [cert page][mlp-page] |
| Official prep | Self-paced (free) and instructor-led: *Machine Learning at Scale*, *Advanced Machine Learning Operations*. Academy URLs and durations are unconfirmed. | [guide PDF][mlp-pdf] |
| Official practice | 10 sample questions in the guide only | [guide PDF][mlp-pdf] |

**Rename traps.** Answer in the exam's words. Already learned, not repeated: Delta Live Tables → Lakeflow pipelines and online tables → Online Feature Store (DBX-MLA.0/.1); Databricks Asset Bundles → Declarative Automation Bundles (DBX-GAI.1). New for this exam:

| Exam says | Docs now say |
|---|---|
| Lakehouse Monitoring | "Data profiling was formerly known as Lakehouse Monitoring", under Data Quality Monitoring ([docs][dqm]) |

### Already covered: objectives owned by a core

| Objectives ([matrix](#coverage-matrix) numbering) | Owner | Refresh action |
|---|---|---|
| 1.1–1.5 Spark ML pipelines, tuning, evaluators | M3.6 | Rebuild `Pipeline` + `CrossValidator` from memory |
| 1.9 Optuna concepts | M4.4 | Re-read your M4.4 notes |
| 1.10, 1.13 Ray basics, Ray vs Spark | M7.4 | Re-run your M7.4 Tune script |
| 1.12 data vs model parallelism | M7.3 | DDP vs FSDP/tensor parallel: when |
| 1.15–1.16 nested runs, custom logging | M6.2 | Drill in DBX-MLP.4 |
| 1.11 vertical vs horizontal scaling (concept) | M7.3 | One rule per workload type; Databricks framing in DBX-MLP.1 |
| 1.18 point-in-time correctness | M3.5 | Explain the as-of join with one example |
| 2.3–2.5 unit vs integration tests | M6.9 | Re-read your M6.9 test layout |
| 2.9 retraining triggers and cadence | M6.7, M6.8 | Scheduled vs drift-triggered vs performance-triggered |
| 2.10 metric for model selection | M2.2, M6.3 | Calibrated thresholds → **log loss** (sample Q6) |
| 2.11, 2.14, 2.16 drift tests, monitoring, baselines | M6.8 | Test per data type; PSI bins |
| 2.19 slicing | M6.3 | Slice-based evaluation |
| 3.1 blue-green, canary, shadow | M6.7, C8.4 | One line each on risk, cost and rollback |

**Carried over (no re-study; skip tests below check them):** runtimes, UC registry → DBX-MLA.1 · feature tables, PIT keys, `score_batch` → DBX-MLA.3 · deploy code vs models → DBX-MLA.5 · `applyInPandas` → DBX-MLA.7 · `spark_udf` scoring → DBX-MLA.8 · serving, traffic split → DBX-MLA.9 · pyfunc and its `load_context` pattern, MLflow Deployments, inference tables, route optimization (concept), bundles basics (`databricks.yml`, targets, `validate`/`deploy`) → DBX-GAI.1.

- [ ] Read the [DMP guide PDF][mlp-pdf]. Answer the 10 samples cold and log your score.
- [ ] Do the refresh actions above.

**Tested by:** DMP.

---

## Modules

Format as in DBX-MLA (pass the **Skip test** → do only exercise + self-check).

**Official courses:** *Machine Learning at Scale*, *Advanced Machine Learning Operations* (free in Databricks Academy; search by title, URLs unconfirmed).

### DBX-MLP.1 Classic compute, scaling choices and the lab sprint

⏱ 2.5 h · prereqs: DBX-MLA.1, M7.3

**Skip test**
1. Your single-node XGBoost run hits out-of-memory. Do you scale vertically or horizontally?
2. Spark ML training is too slow. Vertical or horizontal?
3. Which compute access modes allow Ray on Spark?
4. Why can't you do this exam's Ray lab on Free Edition?
5. When is a GPU single-node cluster better than an 8-worker CPU cluster?

<details><summary>Answers</summary>

1. Vertical: a bigger node, or a GPU node.
2. Horizontal: more workers, since the work is data-parallel.
3. Dedicated (formerly "single user") or No isolation shared, on all-purpose or jobs compute, DBR 12.2 LTS ML+ ([docs][ray-create], [limitations][ray]).
4. Free Edition is serverless only, and Ray on Spark "cannot be initiated on clusters using serverless-based runtimes" ([docs][ray]). (Ray on serverless GPU is the separate "Ray on AI Runtime" product; not what the exam means.)
5. For deep-learning training, and for single-node libraries that fit on one GPU node.
</details>

**Concepts**
- Single- vs multi-node clusters, autoscaling, driver vs worker sizing; vertical (bigger/GPU node) vs horizontal (more workers) per workload type.
- AI Runtime = separate serverless-GPU product (name recognition only).

**Lab environment.** Free Edition has no classic compute ([limits][fe-limits]). For classic-compute labs (.3, parts of .5/.7/.9) use an employer sandbox or the **free trial** ("Full Databricks platform access for two (2) weeks", "Up to $400 in credits", no credit card, "Work email address recommended"; [trial page][trial]); start it after .2/.4/.6/.7/.8 (see Schedule) and auto-terminate clusters at 20 min. Lakebase on Free Edition is **conflicting**: the limits page says "One Lakebase project per account, with scale-to-zero compute", while the try-databricks comparison lists Lakebase among Free Edition's "Locked features". Check in your workspace before the online-store lab (the Online Feature Store itself runs on serverless or DBR 16.4 LTS ML+, [docs][ofs]).

**Resources**
- Docs: [Databricks Runtime for ML][dbrml]: "Create a compute resource with Databricks Runtime for ML", "Compute access mode for Databricks Runtime ML" and the GPU-workloads note recommending AI Runtime.
- Academy *Machine Learning at Scale*: the scaling and cluster-configuration lesson.

**Exercise.** Write `sizing.md` for your capstone with a decision table: workload → node type, single or multi-node, CPU or GPU, and why. Then create the trial (or sandbox) cluster on DBR ML LTS in Dedicated access mode. **Done when:** the cluster runs `import ray, optuna, mlflow` successfully.

**Self-check**
- [ ] For four workload types, I can pick vertical vs horizontal with a reason.

**Tested by:** DMP 1.11.

### DBX-MLP.2 Spark ML at Professional depth

⏱ 3 h · prereqs: M3.6, DBX-MLA.7, DBX-MLA.8

**Skip test**
1. You have 430M rows with categorical and numeric features. What is the encoding pipeline? (sample Q1)
2. Tune a Spark ML model: what do `ParamGridBuilder`, `CrossValidator` and `parallelism` each do?
3. Which evaluator gives AUC, and which gives RMSE?
4. A model must answer in 50 ms per request. Do you use Spark ML or a single-node model?
5. How do you score a Spark ML model on a stream?

<details><summary>Answers</summary>

1. `StringIndexer` → `OneHotEncoder` → `VectorAssembler`.
2. The grid of parameter combinations; k-fold fitting over that grid; how many models train concurrently.
3. `BinaryClassificationEvaluator`; `RegressionEvaluator`.
4. Single-node. Spark ML fits batch and streaming; real-time serving of Spark ML models is not the exam's answer (whether Model Serving supports pyspark.ml is unverified).
5. `PipelineModel.transform` on a streaming DataFrame, or `spark_udf`.
</details>

**Concepts**
- Spark ML when data exceeds one node and MLlib has the algorithm; MLlib tuning parallelism; inference type per model family; MLflow logging of `PipelineModel`.

**Resources**
- Docs: [MLlib on Databricks][mllib]: the "GBT regression using MLlib pipelines" (bike sharing) notebook. The page has no separate tuning notebook; `ParamGridBuilder`/`CrossValidator` API → M3.6.
- Academy *Machine Learning at Scale*: the Spark ML pipeline and tuning lessons.

**Exercise.** Train a Spark ML `Pipeline` (indexer → OHE → assembler → GBT or linear regression) with `CrossValidator(parallelism=4)` on your capstone table, enlarged ×20 with `crossJoin` so it is "big enough". Log the run to MLflow, then score a stream with the `PipelineModel`. **Done when:** the CV metrics are in MLflow and a streaming query writes predictions.

**Self-check**
- [ ] I can write the 4-stage pipeline and the CV setup without docs.
- [ ] I can say which inference type each model family supports.

**Tested by:** DMP 1.1–1.7.

### DBX-MLP.3 Distributed tuning and training: Optuna on Spark, Ray on Spark, pandas UDFs

⏱ 4.5 h · prereqs: M4.4, M7.3, M7.4, DBX-MLA.6, DBX-MLP.1

**Skip test**
1. Distribute Optuna trials across a cluster with each trial logged to MLflow, with minimal setup. What do you use? (sample Q2)
2. Start a Ray cluster on Databricks. Which function and key parameters?
3. When should you choose Ray over Spark for training?
4. Train 5,000 per-SKU forecasters. Which API?
5. A 70B model doesn't fit one GPU. Which parallelism? What about a 1B model on a 10 TB dataset?

<details><summary>Answers</summary>

1. `MlflowStorage(experiment_id)` + `MlflowSparkStudy`. Distractor: `MLflowCallback` + async.
2. `ray.util.spark.setup_ray_cluster(max_worker_nodes=…, num_cpus_per_node=…, num_gpus_per_node=…)`.
3. Ray for dynamic, compute-heavy logical parallelism (tuning, RL, DL); Spark for data-parallel ETL and large-dataset processing.
4. `applyInPandas` grouped by SKU.
5. Model parallelism; data parallelism.
</details>

**Concepts**
- `MlflowStorage` + `MlflowSparkStudy` ("parallel Optuna studies using PySpark executors", [docs][optuna]); MLflow 3 preinstalled on DBR 17.0 ML+.
- Ray on Spark ([docs][ray-create]): DBR 12.2 LTS ML+, preinstalled from 15.0 ML; `setup_ray_cluster` / `shutdown_ray_cluster`; Ray Tune inside. Ray vs Spark ([docs][ray]): "logical parallelism" vs "data parallelism".
- TorchDistributor (DDP on Spark): name and fit only ([docs][torchdist]); theory → M7.3.

**Resources**
- Docs: [Optuna on Databricks][optuna]: the "Run Optuna optimization in parallel" section.
- Docs: [What is Ray on Databricks][ray] (whole) and [Start a Ray cluster][ray-create] (setup section).
- Academy *Machine Learning at Scale*: the distributed tuning and Ray lessons.

**Exercise (trial or sandbox cluster)**
1. Tune the same model three ways: Optuna `MlflowSparkStudy` (50 trials), Ray Tune on Ray-on-Spark (50 trials), and a single-node Optuna baseline.
2. Train one model per group with `applyInPandas`; reuse your DBX-MLA.7 code at ×100 groups.

**Done when:**
- a `tuning.md` table compares wall-clock time, best metric and setup lines for the three tuning runs;
- all trials are visible in MLflow.

**Self-check**
- [ ] I can write `MlflowSparkStudy` and `setup_ray_cluster` calls from memory.
- [ ] I can argue Ray vs Spark in two sentences.

**Tested by:** DMP 1.8–1.10, 1.12–1.14 (1.11 is DBX-MLP.1).

### DBX-MLP.4 Advanced MLflow and custom PyFunc serving

⏱ 3 h · prereqs: M6.2, DBX-GAI.1 (pyfunc, Deployments), DBX-MLA.5

**Skip test**
1. Structure runs for k-fold tuning followed by a final refit. (sample Q3)
2. Write a pyfunc that one-hot encodes a raw request and applies a saved scaler before predicting.
3. Register it with its artifacts in UC.
4. Query the endpoint from a Python pipeline with auth handled for you. (sample Q9)
5. Create an endpoint from code.

<details><summary>Answers</summary>

1. A parent run for the experiment, a nested child per hyperparameter set (aggregated CV metrics), and a separate run for the final model.
2. A `PythonModel` whose `load_context` reads `context.artifacts`, with feature engineering inside `predict`.
3. `log_model(..., artifacts={…}, signature=…, registered_model_name="cat.sch.m")` with the UC registry URI.
4. `get_deploy_client("databricks").predict(endpoint=…, inputs=…)`. Not `ai_query` (SQL), and never a token in the query string.
5. `create_endpoint(name, config={served_entities…})`, the REST `POST /api/2.0/serving-endpoints`, or the UI.
</details>

**Concepts**
- `start_run(nested=True)`; `log_table`/`log_figure`/`log_dict`.
- Custom `PythonModel` (subclassing and `load_context` vs `predict` → DBX-GAI.1). New here: fitted preprocessing objects passed as `artifacts={...}` at log time, real-time feature engineering inside `predict`, `code_paths`, signature, input example.
- Deploy via SDK/REST/UI; query via REST (the Deployments SDK → DBX-GAI.1).

**Resources**
- Nested runs: no new reading. The parent/child model is in the "SparkTrials and MLflow" section of Hyperopt concepts (read in DBX-MLA.6); `mlflow.start_run(nested=True)` → M6.2.
- Docs: [MLflow runs on Databricks][runs]: "Compare runs" only (how child runs appear when comparing in the UI).
- Docs: [Query custom model endpoints][score-ep]: "Querying methods and examples", REST tab only (the Deployments client was covered in DBX-MLA.9 / DBX-GAI.1).
- Academy *Advanced Machine Learning Operations*: the custom model lesson.

**Exercise.** Write `HousePriceModel(PythonModel)`. It accepts raw JSON rows, does the DBX-MLA.2 transformations inside `predict` with a fitted encoder artifact, and predicts. Log it with nested-run tuning lineage, register it to UC, and deploy it with `create_endpoint`. Query it via both REST (`requests`, bearer header) and `mlflow.deployments`. **Done when:** both queries return identical predictions for raw, unencoded input.

**Self-check**
- [ ] I can sketch the nested-run tree for sample Q3.
- [ ] I can write `load_context`/`predict` from memory.

**Tested by:** DMP 1.15–1.17, 3.3–3.5.

### DBX-MLP.5 Advanced feature store and Lakeflow pipelines

⏱ 4.5 h · prereqs: M3.5, DBX-MLA.3, DBX-MLA.8

**Skip test**
1. Name the online serving store for features today, and the exam's older term for it.
2. What is an on-demand feature, and how is it defined?
3. How does a feature serving endpoint differ from a model serving endpoint with automatic feature lookup?
4. Compute 5-minute rolling features from a Kafka-like stream into a feature table.
5. How do you guarantee point-in-time correct training sets?

<details><summary>Answers</summary>

1. The Online Feature Store (Lakebase); the exam says "online tables".
2. A feature computed at request time by a UC Python function (`FeatureFunction`), used identically in training and serving.
3. A feature serving endpoint returns features (a FeatureSpec) to any app; the model endpoint looks features up to feed its model.
4. A Lakeflow streaming table or a Structured Streaming job that runs `fe.write_table(mode="merge")`, then publishes to the online store.
5. A time-series key plus `timestamp_lookup_key` in `FeatureLookup` (→ DBX-MLA.3, M3.5).
</details>

**Concepts**
- Online Feature Store on Lakebase (Databricks' managed Postgres; product depth is in DBX-CTX.1 and not needed here): `fe.create_online_store(...)`, then `fe.publish_table(...)` with `publish_mode` TRIGGERED (default; needs Change Data Feed), CONTINUOUS or SNAPSHOT ([docs][ofs]). Requires `databricks-feature-engineering>=0.13.0` on DBR 16.4 LTS ML+ or serverless. Exam wording "online tables via Databricks SDK" = legacy `WorkspaceClient` spec (primary keys, triggered/continuous sync): recognise, don't build.
- `FeatureFunction` + `create_feature_spec`; feature serving endpoints.
- Lakeflow pipelines (owned here; DBX-MLA.8 used them only for inference): streaming tables vs materialized views, expectations, triggered vs continuous.

**Resources**
- Docs: [Online Feature Store][ofs]: "Create an online store" + "Publish a feature table to an online store".
- Docs: [On-demand features][ondemand] (whole).
- Docs: [Feature serving][featserve]: "Create a FeatureSpec", "Create an endpoint", "Query an endpoint using APIs".
- Docs: [Lakeflow pipeline concepts][ldp-concepts] (flows, streaming tables, materialized views, sinks) + one Python tutorial from the [tutorials index][ldp-tut]. (Landing page: [Lakeflow pipelines][ldp].)
- Academy *Machine Learning at Scale*: the feature store lessons.

**Exercise**
1. Build a Lakeflow pipeline (Python) that turns an append-only events table into rolling-window features. Add an expectation that drops nulls.
2. Write the features into a time-series UC feature table and add an on-demand `FeatureFunction` (e.g. `price_per_sqft`).
3. Publish to the online store and serve a FeatureSpec. If Lakebase is unavailable, read the publish code and do steps 1–2 only.

**Done when:** a feature-serving request for one key returns stored plus on-demand features, and a training set built with `timestamp_lookup_key` provably excludes future values (assert in a test).

**Self-check**
- [ ] I can map the four feature objectives to the API names.
- [ ] I can explain streaming tables vs materialized views.

**Tested by:** DMP 1.18–1.22.

### DBX-MLP.6 MLOps architecture: MLOps Stacks, bundles and testing

⏱ 5 h · prereqs: M6.9, DBX-MLA.5, DBX-GAI.1 (bundles basics)

**Skip test**
1. Sketch the deploy-code architecture across dev, staging and prod, saying what runs where.
2. In `databricks.yml`, which resource keys define an experiment, a registered model and a serving endpoint?
3. Where does Databricks recommend keeping Python functions and their unit tests?
4. A hyperparameter changes. Which integration tests must re-run? (sample Q5)
5. Map six Databricks features to lifecycle stages.

<details><summary>Answers</summary>

1. Dev builds the code. Staging runs CI integration tests on staging data. Prod trains, validates and deploys on prod data, and the same code moves between them.
2. `experiments`, `registered_models`, `model_serving_endpoints`, plus `jobs`, under per-environment `targets`.
3. Trick question: the docs list three approaches, each with "benefits and challenges", without one mandate. Functions and tests "outside of notebooks" (Python files + pytest) get the best tooling; the others are functions in one notebook with tests in a separate notebook, or both in the same notebook ([docs][testing]).
4. Training, evaluation and deployment tests; not feature engineering.
5. Repos/Git folders → dev; Jobs → training and retraining; UC registry + aliases → validation and promotion; Model Serving → deploy; inference tables + data profiling → monitor; bundles → CI/CD.
</details>

**Concepts**
- MLOps Stacks: model development "implemented, saved, and tracked as code" ([docs][stacks]); `databricks bundle init mlops-stacks`; a three-target layout (dev/staging/prod; `databricks.yml`, targets, `validate`/`deploy` → DBX-GAI.1); `bundle run`; the ML resource keys; workspaces or catalogs per environment.
- Tests: unit (workspace files + pytest) vs integration (Jobs) per environment; three ways to organise functions and tests ([docs][testing]).

**Resources**
- Docs: [MLOps Stacks][stacks] (whole).
- Docs: [Bundle resources][bundle-res]: the `experiment`, `registered_model (Unity Catalog)` and `model_serving_endpoint` sections (YAML keys are the plurals); skim `quality_monitor` (data profiling as a bundle resource, used in DBX-MLP.7).
- Docs: [Unit testing for notebooks][testing] (whole).
- Academy *Advanced Machine Learning Operations*: the environments, testing and CI/CD lessons.

**Exercise**
1. Turn your DBX-MLA.10 capstone into a bundle with `dev` and `staging` targets. On Free Edition, use two schemas as stand-in environments. Declare the experiment, registered model, endpoint and training Job.
2. Add pytest unit tests for `prep.py` and the pyfunc.
3. Add one integration Job (features → train → evaluate → deploy-to-staging → inference smoke test).
4. Add a GitHub Actions workflow that runs pytest plus `databricks bundle validate`.

**Done when:** `databricks bundle deploy -t staging` creates all resources from Git with no manual UI steps, and CI is green.

**Self-check**
- [ ] I can list which tests run in dev, staging and prod.
- [ ] I can write the bundle resource skeleton from memory.

**Tested by:** DMP 2.1–2.8.

### DBX-MLP.7 Monitoring: data profiling (ex-Lakehouse Monitoring) and endpoint health

⏱ 5 h · prereqs: M6.8, DBX-GAI.1 (inference tables), DBX-MLP.4

**Skip test**
1. Which three profile types exist, and which one fits a model's request log?
2. Which drift statistics apply to numeric columns, and which to categorical?
3. What do `drift_type` BASELINE and CONSECUTIVE compare?
4. Define a metric not in the default output.
5. How do you alert the team when PSI > 0.2?
6. Which endpoint health metrics do you watch?

<details><summary>Answers</summary>

1. Snapshot, time series, inference. The inference profile fits a request log.
2. Numeric: `ks_test`, `wasserstein_distance`, `population_stability_index`. Categorical: `chi_squared_test`, `tv_distance`, `l_infinity_distance`, `js_distance` ([docs][monitor-out]).
3. BASELINE compares the window to a baseline table; CONSECUTIVE compares it to the previous window.
4. A custom metric: aggregate, derived or drift ([docs][custom-metrics]).
5. A Databricks SQL alert on a query over the drift metrics table, with a notification destination ([docs][alerts]).
6. Latency, request rate, error rate, CPU and memory usage ([docs][ep-monitor]).
</details>

**Concepts** (beyond the answers above)
- Snapshot profiles have a 4 TB limit ([docs][profiling]). Inference profiles take a request log with model ID, inputs, prediction and an **optional** ground-truth label, plus a timestamp and problem type ([docs][profiling], [API][monitor-api]).
- Output: two metrics tables + an auto dashboard; `granularities`; `slicing_exprs` → `slice_key`/`slice_value`; baseline table.
- Performance over time = inference table joined with late labels. Pipeline: logging → drift → performance → health → alert.

**Resources**
- Docs: [Data profiling overview][profiling].
- Docs: [Create a profile via API][monitor-api]: the inference-profile example.
- Docs: [Monitor output tables][monitor-out]: the drift metrics table schema.
- Docs: [Custom metrics][custom-metrics] and [Alerts][alerts].
- Docs: [Monitor model serving endpoints][ep-monitor]: the metrics section.
- Academy *Advanced Machine Learning Operations*: the monitoring lessons.

**Exercise**
1. Join your endpoint's inference table (or `ml_lab.dma.predictions` plus a timestamp) with delayed labels.
2. Create an **inference** profile with a 1-day granularity, a slice on `region` and a baseline equal to the training set.
3. Add a custom derived metric (MAPE).
4. Inject a shifted batch (multiply one feature by 1.5) and refresh.
5. Write a SQL alert on PSI or KS.

**Done when:** the drift metrics table shows the shift for the shifted feature only, and the alert fires. **Fallback** if data profiling is unavailable to you: compute KS and PSI with Evidently (as in M6.8) on the same data, and write the SQL you *would* alert on against the documented schema.

**Self-check**
- [ ] Given a scenario, I can pick the profile type, test, granularity and slice in under a minute.
- [ ] I can list which tests return null for which column types.

**Tested by:** DMP 2.11–2.20.

### DBX-MLP.8 Automated retraining and rollout on Model Serving

⏱ 3 h · prereqs: M6.7, DBX-MLA.5, DBX-MLA.9, DBX-MLP.7

**Skip test**
1. Drift alert → retrain: how is it wired on Databricks?
2. The retrain produces a challenger. What gates promotion?
3. Roll out a new model on a high-traffic, business-critical endpoint. Which settings? (sample Q7)
4. Blue-green vs canary on Model Serving: how does each look?
5. What can't you change on an existing endpoint?

<details><summary>Answers</summary>

1. A SQL alert or monitoring check triggers the retraining Job (via a Jobs trigger, or `run_now` from a check task). Prefer drift-triggered retraining when the question says so.
2. Challenger vs `@champion` on the same holdout, using the business-aligned metric plus guardrail metrics; then the alias moves.
3. Compute scale-out, Route Optimization, and a second served entity whose traffic increases step by step while monitored.
4. Blue-green: a second entity at 0%, then switch to 100% with an instant rollback path. Canary: incremental percentages.
5. Route optimization: "You can only enable route optimization during endpoint creation, you can not update existing endpoints to be route optimized" ([docs][route-opt]).
</details>

**Concepts**
- Jobs triggers: Scheduled, Table update, File arrival, Continuous ([docs][triggers]); champion/challenger retraining job (deploy code); alias pinning for stable comparisons (sample Q4); served-entity traffic ramps, scale-out. Route optimization (what it is → DBX-GAI.1); new here: it can only be enabled at endpoint creation, and it belongs in high-traffic rollout answers.

**Resources**
- Docs: [Jobs triggers][triggers]: trigger types list.
- Docs: [Route optimization][route-opt] (whole).
- Traffic split → DBX-MLA.9 / DBX-GAI.1.
- Academy *Advanced Machine Learning Operations*: the retraining and deployment-strategy lessons.

**Exercise.** Write `canary.py`:
- It adds `@challenger` as a second served entity at 5%.
- It ramps 5 → 25 → 50 → 100% while checking the endpoint error rate and latency.
- It rolls back on breach.

Wire DBX-MLP.7's alert so that it starts the retraining Job. **Done when:** a forced error (a bad input schema in the challenger) triggers automatic rollback to 100% champion.

**Self-check**
- [ ] I can answer sample Q4, Q6 and Q7 with reasons, including why each distractor fails.

**Tested by:** DMP 2.9, 2.10, 3.1, 3.2.

---

## DBX-MLP.9 Capstone: production-grade M6.10 on Databricks

⏱ 8 h · prereqs: DBX-MLP.1–.8, DBX-MLA.10

Extend the DBX-MLA.10 re-platform of your **M6.10** project:

| Add | From |
|---|---|
| Bundle with dev/staging(/prod) targets, CI with pytest + `bundle validate` | DBX-MLP.6 |
| Spark ML or `MlflowSparkStudy` tuning with nested runs | DBX-MLP.2–.4 |
| Custom pyfunc with real-time feature engineering, plus online/on-demand features | DBX-MLP.4–.5 |
| Inference profile, custom metric, drift alert | DBX-MLP.7 |
| Drift-triggered retraining → challenger → canary → promote or rollback | DBX-MLP.8 |

- [ ] **Done when:** one scripted demo, deployed from Git, runs the loop: inject drifted data → alert → retraining Job registers a challenger → it beats the champion on the holdout → canary ramps → alias moves. The README maps every DMP section to the file that demonstrates it.

---

## DBX-MLP.10 Practice, readiness, exam day

⏱ 3 h + 3 h final-week review

**Practice strategy:** same method as DBX-MLA.11, applied to the DMP guide:
- the official samples cold, then again after .8;
- the AI Prep Guide process (owned by DBX-GAI);
- an objective drill of 94 self-written, docs-verified questions (MLOps has 20 bullets, half of them monitoring).

No third-party banks or dumps ([terms][terms]).

**Readiness signal to book**
- [ ] 10/10 on the samples with reasons.
- [ ] ≥85% on your drill, with no subsection below 75%.
- [ ] Capstone loop demoed.
- [ ] From memory: the drift-test-by-type table, the bundle resource keys, `MlflowSparkStudy`, `setup_ray_cluster`, and the pyfunc skeleton.

**Final-week review (3 h):** D1 monitoring (2.11–2.20) · D2 feature store + Lakeflow · D3 scaling/tuning · D4 missed drill questions · D5 samples + rename table.

**Exam-day tactics:** 59 + ≤10 items in 120 min ≈ 1.7 min each; read the stem's last sentence first; prefer the Databricks-native, Delta/UC-centric option; watch "minimal setup", "at a minimum", "high-traffic"; reschedule >24 h ahead online, >72 h test center ([FAQ][faq]).

**First-hand traps**
- [Correa, 2026-02-01][correa] (version unclear; quotes from the research notes, Medium body not re-fetchable on 2026-11-01): Lakehouse Monitoring "everywhere"; pick drift tests by data type; drift-triggered ≠ scheduled retraining; tests "ML engineering judgment".
- [Rosa, 2025-08-23][rosa] (previous version): "For the exam, they expect familiarity with using Delta tables" (his own project used pgvector).
- [Cole][cole]: "one of the more challenging Databricks certifications".
- Official-sample distractors: `MLflowCallback` (Optuna), `get_latest_version` (stable comparison), canary without Route Optimization, `ai_query` (SDK querying).

---

## Coverage matrix

| # | Task statement (abridged) | Core | DBX module |
|---|---|---|---|
| 1.1 | When Spark ML is recommended | M3.6 | DBX-MLP.2 |
| 1.2 | Construct a Spark ML pipeline | M3.6 | DBX-MLP.2 |
| 1.3 | Pick estimator/transformer for a use case | M3.6, M2.1 | DBX-MLP.2 |
| 1.4 | Tune a Spark ML model with MLlib | M3.6 | DBX-MLP.2 |
| 1.5 | Evaluate a Spark ML model | M3.6, M2.2 | DBX-MLP.2 |
| 1.6 | Score Spark ML in batch or streaming | DBX-MLA.8 (Structured Streaming) | DBX-MLP.2 |
| 1.7 | Spark ML vs single-node by inference type | M6.5 | DBX-MLP.2 |
| 1.8 | Scale training with Spark ML + pandas Function API | — | DBX-MLA.7, DBX-MLP.3 |
| 1.9 | Distributed Optuna + MLflow | M4.4 | DBX-MLP.3 |
| 1.10 | Distributed tuning with Ray | M7.4 | DBX-MLP.3 |
| 1.11 | Vertical vs horizontal scaling | M7.3 | DBX-MLP.1 |
| 1.12 | Model vs data parallelism | M7.3 | DBX-MLP.3 |
| 1.13 | Ray vs Spark for training | M7.4, M3.6 | DBX-MLP.3 |
| 1.14 | pandas Function API group training + inference | — | DBX-MLA.7, DBX-MLP.3 |
| 1.15 | Nested runs | M6.2 | DBX-MLP.4 |
| 1.16 | Programmatic custom logging | M6.2 | DBX-MLA.5, DBX-MLP.4 |
| 1.17 | Custom model objects with real-time FE | DBX-GAI.1 | DBX-MLP.4 |
| 1.18 | Point-in-time correctness | M3.5 | DBX-MLA.3, DBX-MLP.5 |
| 1.19 | Automated feature pipelines with FE Client | — | DBX-MLP.5 |
| 1.20 | Online tables via SDK | — | DBX-MLP.5 |
| 1.21 | Streaming feature ingestion | — | DBX-MLP.5 |
| 1.22 | On-demand features via feature serving | — | DBX-MLP.5 |
| 2.1 | Lifecycle architecture, deploy-code strategy | M6.1 | DBX-MLA.5, DBX-MLP.6 |
| 2.2 | Map Databricks features to lifecycle | — | DBX-MLP.6 |
| 2.3 | Unit tests for notebook functions | M6.9 | DBX-MLP.6 |
| 2.4 | Test types per environment | M6.9 | DBX-MLP.6 |
| 2.5 | Integration test across pipelines | M6.9 | DBX-MLP.6 |
| 2.6 | Organising functions and unit tests | — | DBX-MLP.6 |
| 2.7 | Scalable Databricks ML environments | — | DBX-MLP.6 |
| 2.8 | DAB ML assets (endpoints, experiments, models) | DBX-GAI.1 | DBX-MLP.6 |
| 2.9 | Retraining on drift or degradation | M6.7, M6.8 | DBX-MLP.8 |
| 2.10 | Select top model during retraining | M2.2, M6.3 | DBX-MLP.8 |
| 2.11 | Statistical tests in the drift metrics table | M6.8 | DBX-MLP.7 |
| 2.12 | Table type + monitoring feature for a need | — | DBX-MLP.7 |
| 2.13 | Build a snapshot/time-series/inference monitor | — | DBX-MLP.7 |
| 2.14 | Monitoring pipeline components | M6.8 | DBX-MLP.7 |
| 2.15 | Alerting on drift thresholds | — | DBX-MLP.7 |
| 2.16 | Drift vs baseline or consecutive windows | M6.8 | DBX-MLP.7 |
| 2.17 | Performance trends from inference table | DBX-GAI.1 | DBX-MLP.7 |
| 2.18 | Custom metrics | — | DBX-MLP.7 |
| 2.19 | Granularities and slicing | M6.3 | DBX-MLP.7 |
| 2.20 | Endpoint health metrics | C8.3 (concept) | DBX-MLP.7 |
| 3.1 | Blue-green vs canary for high traffic | M6.7, C8.4 | DBX-MLP.8 |
| 3.2 | Rollout on Model Serving | DBX-GAI.1 | DBX-MLP.8 |
| 3.3 | Register custom PyFunc + artifacts in UC | DBX-GAI.1 | DBX-MLP.4 |
| 3.4 | Query via REST or Deployments SDK | DBX-GAI.1 | DBX-MLP.4 |
| 3.5 | Deploy via SDK, REST or UI | DBX-GAI.1 | DBX-MLP.4 |

Every row is practised in its module's exercise and integrated in DBX-MLP.9.

---

## Schedule

| Module | .0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 | .8 | .9 | .10 | Final review | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h | 1.5 | 2.5 | 3 | 4.5 | 3 | 4.5 | 5 | 5 | 3 | 8 | 3 | 3 | **46 h** |

**Default (~7–9 h/week, ~7 weeks):** W1 .0–.2 · W2 .4, .6 · W3 .7, .8 (serverless parts and fallbacks) · W4 start the 2-week trial; .3, .5 · W5 .9 + any classic-compute parts of .7/.8 (trial ends) · W6 .10 + review · W7 sit. Doing .4, .6, .7 and .8 first keeps all classic-compute work inside the two trial weeks (W4–W5).

**Compressed (~16 h/week, 3 weeks):** W1 .0–.2, .4, .6 · W2 (trial) .3, .5, .7, .8 · W3 .9, .10 + review.

Combined with DBX-MLA: 39 h + 46 h = **85 h**, about 12 weeks at 7 h/week. Without the Databricks AI path, add DBX-GAI.1 (6 h): **91 h**.

---

## Combining with other paths

- **Chain:** ML core → DBX-GAI.1 (6 h; skip if you took the Databricks AI path) → DBX-MLA → DBX-MLP. The full DBX-GAI and DBX-CTX exams are independent.
- **With [GCP-PMLE](../google/professional-ml-engineer.md), [AWS-MLA](../aws/mla-c02.md), [AZ-AI300](../azure/ai-300.md):** they share the M3/M6/M7 theory; each adds only its platform modules plus a ≤1 h core refresh. AZ-AI300 is a natural next step (AzureML is MLflow-native).

## Gaps for the core

- None open. Resolved 2026-10-31: MLflow nested runs and `log_table`/`log_figure` are now in M6.2; scale-up vs scale-out is now in M7.3; Structured Streaming is owned by DBX-MLA.8 (Databricks-only concept).

## Skipped on purpose

| Item | Reason |
|---|---|
| Instructor-led *ML at Scale* / *Advanced ML Operations* | Paid (price unconfirmed); the same courses are free self-paced |
| Academy Labs (USD 200/yr) | The free trial plus Free Edition cover the labs |
| Legacy online tables hands-on | Superseded by the Online Feature Store; recognising the exam wording is enough |
| TorchDistributor / DeepSpeed labs | One parallelism bullet; theory is in M7.3 |
| Third-party practice banks, community quiz repos | Unofficial, accuracy unknown |
| Exam dumps | Forbidden; the penalty is loss of all Databricks certifications |

## Sources

All accessed 2026-10-31.

[mlp-page]: https://www.databricks.com/learn/certification/machine-learning-professional
[mlp-pdf]: https://www.databricks.com/sites/default/files/2026-10/databricks-certified-machine-learning-professional-exam-guide-august.pdf
[faq]: https://www.databricks.com/learn/certification/faq
[terms]: https://www.databricks.com/learn/certification/terms-and-conditions
[fe-limits]: https://docs.databricks.com/aws/en/getting-started/free-edition-limitations
[trial]: https://www.databricks.com/try-databricks
[dbrml]: https://docs.databricks.com/aws/en/machine-learning/databricks-runtime-ml
[bundle-res]: https://docs.databricks.com/aws/en/dev-tools/bundles/resources
[stacks]: https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-stacks
[testing]: https://docs.databricks.com/aws/en/notebooks/testing
[dqm]: https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/
[profiling]: https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/
[monitor-api]: https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/create-monitor-api
[monitor-out]: https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/monitor-output
[custom-metrics]: https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/custom-metrics
[alerts]: https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/monitor-alerts
[ep-monitor]: https://docs.databricks.com/aws/en/machine-learning/model-serving/monitor-diagnose-endpoints
[ldp]: https://docs.databricks.com/aws/en/ldp/
[ldp-concepts]: https://docs.databricks.com/aws/en/ldp/concepts/
[ldp-tut]: https://docs.databricks.com/aws/en/ldp/tutorials
[ofs]: https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store
[ondemand]: https://docs.databricks.com/aws/en/machine-learning/feature-store/on-demand-features
[featserve]: https://docs.databricks.com/aws/en/machine-learning/feature-store/feature-function-serving
[mllib]: https://docs.databricks.com/aws/en/machine-learning/train-model/mllib
[optuna]: https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna
[ray]: https://docs.databricks.com/aws/en/machine-learning/ray/
[ray-create]: https://docs.databricks.com/aws/en/machine-learning/ray/ray-create
[torchdist]: https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor
[runs]: https://docs.databricks.com/aws/en/mlflow/runs
[score-ep]: https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints
[route-opt]: https://docs.databricks.com/aws/en/machine-learning/model-serving/route-optimization
[triggers]: https://docs.databricks.com/aws/en/jobs/triggers
[correa]: https://medium.com/@gema.correa/how-i-prepared-for-and-passed-the-databricks-certified-machine-learning-professional-exam-c741cfca259e
[rosa]: https://community.databricks.com/t5/community-articles/databricks-machine-learning-professional-preparation/td-p/129420
[cole]: https://www.alexcole.net/databricks-ml-professional-certification-guide-2025/

Link definitions above are the sources (exam facts: cert page, guide PDF, FAQ, terms; first-hand reports are cited for traps only, not as study material).
