# Databricks Certified Machine Learning Associate: study path (DBX-MLA)

**Who this is for / prerequisites.** You have finished the [ML core](../../core-ml-engineering.md), including M3.6 (Spark ML) and M6.10 (MLOps capstone), and the [AI core](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/core-llm-app-engineering.md). You also need one module of the [Databricks GenAI Engineer path](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/databricks/genai-engineer-associate.md). **If you took the Databricks AI path, you already have these. Otherwise do just these modules from it: DBX-GAI.1 Platform foundations (6 h).** DBX-GAI.1 teaches the workspace and Free Edition limits, Unity Catalog, models in UC (aliases and tags), MLflow on Databricks, the pyfunc flavour and its `load_context` pattern, Model Serving (traffic split, scale-to-zero, route optimization as a concept), MLflow Deployments, inference tables, Secrets, bundles basics (exam name DAB) and `ai_query`. No other AI-path module is needed; the ML exams' rename traps are in DBX-MLA.0/.1, not in the GenAI rename table. This file teaches only what the ML Associate exam adds on top: ML runtimes, Spark data processing on Databricks, Feature Engineering in UC, AutoML, Hyperopt, pandas Function API, batch and streaming inference, and the exam's framing. It is written for a senior Java backend engineer who uses Claude Code daily. Exercises are Python only. The follow-on delta is [ml-professional.md](ml-professional.md) (DBX-MLP).

## Contents

- [DBX-MLA.0 Exam at a glance and core refresh](#dbx-mla0-exam-at-a-glance-and-core-refresh)
- [Modules DBX-MLA.1–DBX-MLA.9](#modules)
- [DBX-MLA.10 Capstone](#dbx-mla10-capstone-re-platform-m610-on-databricks)
- [DBX-MLA.11 Practice, readiness, exam day](#dbx-mla11-practice-readiness-exam-day)
- [Coverage matrix](#coverage-matrix)
- [Schedule](#schedule)
- [Combining with other paths](#combining-with-other-paths)
- [Gaps for the core](#gaps-for-the-core)
- [Skipped on purpose](#skipped-on-purpose)
- [Sources](#sources)

---

## DBX-MLA.0 Exam at a glance and core refresh

⏱ 1.5 h · prereqs: ML core, AI core, DBX-GAI.1

### Exam at a glance (as of 2026-10-31)

| Item | Value | Source |
|---|---|---|
| Name | Databricks Certified Machine Learning Associate | [cert page][mla-page] |
| Status | Active. The current PDF (file name "August-2026", posted Oct 2026, adds an "AI Prep Guide" section) still reads "Exam Published March 1, 2025 / Exam Guide Updated March 1, 2025", so the outline is the March 2025 one. | [guide PDF][mla-pdf] |
| Questions | 48 scored multiple-choice and multiple-selection, plus "up to 10 unscored items" | [guide PDF][mla-pdf] |
| Duration | 90 min (time for unscored items is included) | [guide PDF][mla-pdf] |
| Price | USD 200 plus applicable taxes. Each retake costs the full fee; there are no free retake vouchers. | [guide PDF][mla-pdf], [FAQ][faq] |
| Delivery | Online proctored or test center; no test aids | [cert page][mla-page] |
| Languages | English, Japanese, pt-BR, Korean | [cert page][mla-page] |
| Passing score | **Not published**: "set through statistical analysis and are subject to change". The often-quoted ~70% is third-party and unconfirmed. | [FAQ][faq] |
| Retake | 14-day wait between attempts | [FAQ][faq] |
| Eligibility | No prerequisite. 6+ months of hands-on experience recommended. | [cert page][mla-page] |
| Validity | 2 years. To recertify, take the full current live exam. | [guide PDF][mla-pdf], [FAQ][faq] |
| Weights | Databricks Machine Learning 38% · Data Processing 19% (the web page calls it "ML Workflows") · Model Development 31% · Model Deployment 12% | [cert page][mla-page], [guide PDF][mla-pdf] |
| Official practice | Only the 10 sample questions with explanations in the guide. Databricks: "Databricks doesn't offer complete practice tests." | [guide PDF][mla-pdf], [community][nopractice] |
| Discount | Learning Festivals give 50% off (a USD 100 voucher). The Sep 16–Oct 14 2026 vouchers expire "before 17 January 2027". **The next window is not announced.** | [festival FAQ][fest] |

**Version drift you must handle.** The exam still tests features the platform has changed. Answer in the exam's vocabulary and study in the docs' vocabulary.
- "Hyperopt is not included in Databricks Runtime for Machine Learning after 16.4 LTS ML." ([docs][hyperopt])
- "In Databricks Runtime 18.0 ML or above, AutoML is not included as a built-in library." ([docs][automl])
- Delta Live Tables is now **Lakeflow pipelines**.
- "Online tables": current docs serve features through the **Online Feature Store** on Lakebase ([docs][ofs]), and Free Edition lists online tables as unsupported ([Free Edition limits][fe-limits]). A dated deprecation notice for legacy online tables was not found (**unconfirmed**).

### Already covered: exam objectives owned by a core (light refresh only)

| Exam objectives (numbering as in the [coverage matrix](#coverage-matrix)) | Owner | Refresh action (≤1 h total) |
|---|---|---|
| 3.1 algorithm choice | M2.1–M2.5 | Re-read your M2 notebook headings; say for each algorithm which data it suits |
| 3.2 class imbalance (the exam answer is **cost-sensitive learning**) | M3.1 | Re-do one `class_weight` vs resampling comparison |
| 3.8–3.10 CV vs holdout, CV in fitting, model count | M2.3, M2.6 | Drill: grid 3×2×3 with 5-fold CV = **90** fits (+1 refit if `refit=True`) |
| 3.11–3.13 classification and regression metrics, metric choice | M2.2 | Flash-card F1 / log loss / ROC-AUC / RMSE / MAE / R² ("when does it mislead?") |
| 3.14 exponentiate log-target predictions | M3.2 | `np.expm1(pred)` before RMSE when you trained on `log1p(y)` |
| 3.15 bias–variance | M2.3 | Low train error with high validation error = high variance → regularise or simplify |
| 2.5–2.9 imputation, one-hot (and when not: tree models, high cardinality), log transform | M3.2 | Re-read your M3.2 encoding notes |
| 2.2–2.4 outliers, plots, comparing features | M2.1, M1.3 | IQR rule, z-score rule; correlation vs chi-square vs box-plot by type pair |
| 1.10 online vs offline features (concept) | M3.5 | Recall the offline/online split and training–serving skew |
| 1.1 MLOps strategy (concept), 1.11–1.13 MLflow tracking basics | M6.1, M6.2 | Skim your M6.2 homework |
| 4.1 batch vs real-time vs streaming | M6.5 | The exam's default: a nightly full-table score is **batch** |
| 3.3–3.4 estimators vs transformers, Spark pipelines | M3.6 | `fit()` returns a Model (transformer); `transform()` adds columns |
| 3.6 grid / random / Bayesian search | M4.4 | TPE in one sentence |

The AI core is not tested here. Its MLflow GenAI items (C2.2, C3.3, C8.3) belong to the DBX-GAI exam.

- [ ] Read the [DMA guide PDF][mla-pdf] in full: outline plus the 10 sample questions (40 min). Answer the samples cold and record your score as a baseline.
- [ ] Do the refresh actions above.

**Tested by:** DMA.

---

## Modules

Module format: time · prereqs · **Skip test** (pass it → skip the reading, still do the exercise and self-check) · concepts · resource · exercise (done when) · self-check · tested by.

**Lab environment.** Use Databricks Free Edition (set up in DBX-GAI.1; its general limits, such as serverless only, no GPU serving and no endpoint batch inference, are listed there). ML-specific limits:
- There is a maximum of 5 concurrent job tasks per account. ([Free Edition limits][fe-limits])
- The [AutoML docs][automl] list supported access modes as Dedicated or No isolation shared (classic compute) and offer a serverless path only for forecasting, so AutoML regression/classification on Free Edition is **probably unavailable** (inferred, not tested). Whether `SparkTrials` runs on serverless is **unconfirmed**. Each module gives a fallback.
- Lakeflow pipelines: one active pipeline per pipeline type ([Free Edition limits][fe-limits]).

**Official course.** The guide recommends four self-paced Databricks Academy courses, all free (Academy made "all self-paced courses for free", [blog][freeed]):
- *Data Preparation for Machine Learning*
- *Machine Learning Model Development*
- *Machine Learning Model Deployment*
- *Machine Learning Operations*

Their exact Academy URLs and durations are **unconfirmed**: the catalog renders dynamically. Search the course titles in Databricks Academy. Hosted lab notebooks need Academy Labs (USD 200/yr, [community][labs]). Use Free Edition instead.

### DBX-MLA.1 Databricks for ML: runtimes, compute, UC-level assets

⏱ 2.5 h · prereqs: DBX-GAI.1, M0.2

**Skip test**
1. Name three things Databricks Runtime ML gives you that the standard runtime doesn't.
2. Why is a feature table in Unity Catalog better than a workspace-local one?
3. Why is a model in the UC registry better than one in the workspace registry? (Recall from DBX-GAI.1's self-check.)
4. Which two exam-tested ML features are no longer built into current ML runtimes?
5. Your team needs Ray on Spark. Can you run it on Free Edition?

<details><summary>Answers</summary>

1. Pre-installed and version-tested ML libraries (scikit-learn, XGBoost, PyTorch, MLflow), GPU builds, plus Ray preinstalled from 15.0 ML.
2. and 3. Account-level governance shared across workspaces: one ACL model, lineage, discovery, and cross-workspace reuse.
4. Hyperopt (after 16.4 LTS ML) and AutoML (from 18.0 ML).
5. Not Ray **on Spark**: it needs all-purpose or jobs compute in Dedicated or No isolation shared mode and "cannot be initiated on clusters using serverless-based runtimes" ([docs][ray]). Ray on serverless GPU is a separate product (Ray on AI Runtime); LinkedIn-verified Free Edition accounts get only "limited access to AI Runtime". See DBX-MLP.1.
</details>

**Concepts**
- DBR ML vs standard DBR. Reasons to choose it: pinned ML library stack, GPU variants, MLflow integration.
- LTS vs non-LTS. Serverless notebooks vs classic all-purpose compute; single-node vs multi-node.
- Account-level UC feature tables vs the workspace-scoped legacy feature store: the same argument as models in UC vs the workspace registry (→ DBX-GAI.1).
- Exam-vs-platform rename table, ML subset: Hyperopt → Optuna / Ray Tune; built-in AutoML → `databricks-automl-runtime`; DLT → Lakeflow pipelines; online tables → Online Feature Store.

**Resources**
- Docs: [Databricks Runtime for ML][dbrml] (overview + "libraries included" section).
- Docs: [Feature tables in UC][fe-uc], the section "Share a feature table in Unity Catalog across workspaces or accounts" only (the page has no separate benefits intro).
- Academy *Data Preparation for Machine Learning*: the platform intro lesson only.

**Exercise.** In Free Edition, create the catalog/schema `ml_lab.dma` and a UC volume, then load your M6.10 project dataset (fallback: scikit-learn `fetch_california_housing(as_frame=True)` with a `house_id` row-number column and a `region` column = latitude rounded to 1 degree) into a Delta table with Python. **Done when:** `spark.table("ml_lab.dma.raw")` returns rows and the table shows lineage in Catalog Explorer.

**Self-check**
- [ ] I can list four ML-runtime benefits and two deprecations from memory.
- [ ] I can state why UC beats the workspace registry in one sentence.

**Tested by:** DMA 1.2, 1.5, 1.15.

### DBX-MLA.2 Spark data processing for ML on Databricks

⏱ 3 h · prereqs: M3.2, M3.6, DBX-MLA.1

**Skip test**
1. Name two ways to get summary statistics for a Spark DataFrame on Databricks.
2. Write the IQR outlier filter in PySpark without collecting to pandas.
3. Which Spark ML class imputes with the median?
4. Which test compares two categorical columns in Spark?
5. Why should you not one-hot encode a 10,000-level ID for a gradient-boosted tree?

<details><summary>Answers</summary>

1. `df.summary()` and `dbutils.data.summarize(df)` (or the `display()` data profile tab).
2. `q1, q3 = df.approxQuantile(c, [0.25, 0.75], 0.01)`, then filter to `[q1-1.5·IQR, q3+1.5·IQR]`.
3. `pyspark.ml.feature.Imputer(strategy="median")`.
4. `pyspark.ml.stat.ChiSquareTest` (use `Correlation` for continuous pairs).
5. Sparse explosion, and trees split poorly on it. Use target/frequency encoding or native categorical handling (→ M3.2).
</details>

**Concepts**
- `.summary()`, `.describe()`, `dbutils.data.summarize`, the `display()` profile and visualisation tabs.
- Outliers by standard deviation and by IQR with `approxQuantile`.
- `Correlation.corr` vs `ChiSquareTest`.
- `Imputer` (mean / median / mode) and the "inspect the distribution first" rule (sample Q2).
- `StringIndexer` → `OneHotEncoder` → `VectorAssembler`.
- `F.log1p` for right-skew.

**Resources**
- Academy *Data Preparation for Machine Learning*: the data exploration, cleaning/imputation and feature engineering lessons.
- Spark API names → M3.6 (already read).

**Exercise.** Write `prep.py`, a function `clean(df) -> df` that does these steps on your raw table:
- profiles the table;
- removes outliers on two numeric columns (one with the std rule, one with IQR);
- median-imputes;
- one-hot encodes two categoricals;
- log-transforms the skewed target.

**Done when:** a pytest run on a 20-row fixture (local `pyspark`) passes, and the cleaned table is written to `ml_lab.dma.clean`.

**Self-check**
- [ ] For each type pair (cat×cat, num×num, cat×num), I can name the comparison method and the plot.
- [ ] I can explain when mean, median and mode imputation fit.

**Tested by:** DMA 2.1–2.9.

### DBX-MLA.3 Feature Engineering in Unity Catalog

⏱ 4 h · prereqs: M3.5, DBX-MLA.2

**Skip test**
1. How do you create a UC feature table: `FeatureStoreClient.register_table`, `CREATE TABLE … AS FEATURE STORE`, or `FeatureEngineeringClient.create_table`?
2. What does `write_table(..., mode="merge")` do?
3. Which object joins features to labels for training?
4. What does `log_model(..., training_set=...)` enable at scoring time?
5. How does a time-series feature table prevent leakage?
6. What is the difference between offline and online feature tables on Databricks?

<details><summary>Answers</summary>

1. `FeatureEngineeringClient.create_table`. The SQL clauses do not exist (sample Q1).
2. Upserts on the primary key.
3. `FeatureLookup` + `create_training_set` → `load_df()`.
4. `score_batch` looks features up automatically from just the keys.
5. A timestamp key gives a point-in-time ("as of") join.
6. The offline table is a Delta table for training and batch. The online store is low-latency serving (Lakebase Online Feature Store; depth in DBX-MLP.5).
</details>

**Concepts**
- `from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup`.
- `create_table` (primary keys, optional time-series key), `write_table`.
- `create_training_set(df, feature_lookups, label, exclude_columns)`.
- `fe.log_model` with a feature spec; `fe.score_batch`.
- Any Delta table with a primary key in UC is usable as a feature table.
- Point-in-time joins (concept → M3.5).

**Resources**
- Docs: [Feature tables in UC][fe-uc]: create, write and update sections.
- Docs: [Train models with feature tables][fe-train]: whole page.
- Docs: [Point-in-time / time-series feature tables][fe-ts]: "How time series feature tables work" and "Create a training set with a time series feature table".
- Academy *Machine Learning Model Development*: the feature store lesson.

**Exercise.** Split your clean table into `ml_lab.dma.house_features` (keyed on `house_id`) and a label table. Train scikit-learn on `create_training_set(...).load_df()`, log it with `fe.log_model`, then score with `fe.score_batch` on a keys-only DataFrame. **Done when:** `score_batch` returns predictions given only `house_id`.

**Self-check**
- [ ] I can write `create_table`, `write_table`, `FeatureLookup`, `create_training_set` and `score_batch` from memory, with argument names close.
- [ ] I can explain why a keys-only scoring call avoids training–serving skew.

**Tested by:** DMA 1.5–1.10.

### DBX-MLA.4 AutoML

⏱ 2 h · prereqs: DBX-MLA.3

**Skip test**
1. What three problem types does AutoML support?
2. What artifacts does AutoML generate besides the model?
3. How does it help with feature and model selection?
4. What changed in DBR 18.0 ML?

<details><summary>Answers</summary>

1. Classification, regression, forecasting.
2. A data-exploration notebook, a notebook for the best trial (all trials for forecasting), and an MLflow experiment with every trial.
3. It runs a time-boxed search over algorithms and hyperparameters, giving a baseline plus editable code ("glass box").
4. AutoML is no longer built in; use the `databricks-automl-runtime` package.
</details>

**Concepts**
- AutoML UI vs `databricks.automl.classify / regress / forecast(dataset, target_col, timeout_minutes)`.
- Generated notebooks as a starting baseline.
- Feature-table joins in AutoML.
- Exam framing: speed to baseline, transparency, MLflow tracking built in.

**Resources**
- Docs: [AutoML][automl]: overview, "How does AutoML work?", "Requirements" (access modes) and "Trial notebook generation"; then [Train regression models with the AutoML Python API][automl-api].
- Academy *Machine Learning Model Development*: the AutoML lesson.

**Exercise.** Run an AutoML regression on `ml_lab.dma.clean` with a 10-min timeout. Open the generated best-trial notebook and change one feature step. **Done when:** you have the best trial's RMSE and the generated notebook re-runs after your edit. **Fallback** if AutoML is unavailable on Free Edition serverless (probable, see Lab environment): run it later on the DBX-MLP.1 trial cluster (Dedicated access mode, DBR ML 17.x or earlier, where AutoML is still built in). If you will not take DBX-MLP, read the regression API page instead and write a 1-page outline of the best-trial notebook (data load, preprocessing, training, SHAP section, MLflow logging); done when the outline names each section the docs mention.

**Self-check**
- [ ] I can give two advantages of AutoML and one limitation.

**Tested by:** DMA 1.3, 1.4.

### DBX-MLA.5 MLflow and MLOps on Databricks: exam framing

⏱ 3 h · prereqs: M6.1, M6.2, DBX-GAI.1 (models in UC, aliases, tags)

**Skip test**
1. Fetch the run with the lowest `val_rmse` using the client.
2. Promote version 7 to production the recommended way.
3. Remove a tag from a registered model.
4. Give two pieces of information visible in the MLflow run UI that are not visible in a model version page.
5. When does Databricks prefer "deploy code" over "deploy models", and vice versa?

<details><summary>Answers</summary>

1. `MlflowClient().search_runs([exp_id], order_by=["metrics.val_rmse ASC"], max_results=1)[0]`.
2. `set_registered_model_alias(name, "champion", 7)`. Stages are deprecated (sample Q6).
3. `delete_registered_model_tag` (or `delete_model_version_tag` for a version).
4. Parameters and metric charts, the run comparison and parallel-coordinates views, artifacts, source notebook and Git commit.
5. Deploy code is the default, best when prod data is restricted or retraining is automated. Deploy models fits when training is expensive, everything is in a single workspace, or there is no CI/CD.
</details>

**Concepts**
- Recall only (taught in DBX-GAI.1): `mlflow.set_registry_uri("databricks-uc")`, three-level model names, aliases, model vs version tags. Drill them here in the exam's question style.
- Recall only (taught in M6.2): `log_metric / log_param / log_artifact / log_model` with a signature.
- `MlflowClient().search_runs(..., order_by=[...])` as the exam's "best run" answer (sample Q7).
- **Deploy code vs deploy models** (this file owns the concept; only Databricks tests it).
- Databricks MLOps reference flow across dev / staging / prod.

**Resources**
- Docs: [MLOps deployment patterns][deploy-patterns]: whole page, 20 min.
- Docs: [MLOps workflows on Databricks][mlops-wf]: the "recommended workflow" diagrams.
- Academy *Machine Learning Operations*: the MLOps strategy lesson.
- MLflow generics → M6.2; UC registry mechanics → DBX-GAI.1. Do not reread them.

**Exercise.** Write `promote.py`:
- It finds the best run in an experiment and registers it to `ml_lab.dma.house_price`.
- It sets alias `challenger`, compares challenger vs `champion` on a holdout set, and moves `champion` if the challenger is better. It tags the losing version `status=rejected`.

**Done when:** running it twice is idempotent, and the alias moves only when the metric improves.

**Self-check**
- [ ] I can draw deploy-code vs deploy-models across three environments.
- [ ] I can say which artefacts move between environments in each pattern.

**Tested by:** DMA 1.1, 1.11–1.18.

### DBX-MLA.6 Hyperparameter tuning on Databricks (Hyperopt, exam version)

⏱ 3 h · prereqs: M4.4, DBX-MLA.5

**Skip test**
1. Write a minimal `fmin` call with TPE.
2. What must the objective function return?
3. `Trials` vs `SparkTrials`: when do you use each, and why not `SparkTrials` with Spark MLlib?
4. How do you turn `fmin`'s returned indices back into values for an `hp.choice`?
5. What does the `parallelism` setting trade off?

<details><summary>Answers</summary>

1. `fmin(fn=obj, space=space, algo=tpe.suggest, max_evals=50, trials=SparkTrials(parallelism=4))`.
2. `{"loss": x, "status": STATUS_OK}`, minimising loss.
3. `SparkTrials` parallelises single-node models across workers. Distributed algorithms (MLlib, Horovod) need `Trials`, because they already use the cluster.
4. `space_eval(space, best)`.
5. Higher parallelism is faster, but adaptive TPE learns less from each finished trial.
</details>

**Concepts**
- `hp.uniform / quniform / loguniform / choice`; TPE vs random (`rand.suggest`).
- `SparkTrials` for parallel single-node models and auto-logging to MLflow.
- Current replacement: Optuna single-node (concept → M4.4). Distributed Optuna is in DBX-MLP.3.

**Resources**
- Docs (1 h): [Hyperopt concepts][hyperopt-concepts] (whole: `fmin()` arguments, the `SparkTrials` class, "SparkTrials and MLflow" parent/child runs); [Hyperopt + MLflow example][hyperopt] ("Part 1" search space and algorithm, "Part 2" distributed tuning); [Hyperopt best practices][hyperopt-bp] ("Best practices" section only).
- Academy *Machine Learning Model Development*: the hyperparameter tuning lesson.

**Exercise.** Tune your XGBoost or RandomForest model with `fmin` (`%pip install hyperopt` on serverless). Use `Trials()` on Free Edition, and `SparkTrials` if you have classic compute. Log the best run.

**Done when:** these work:
- `space_eval` prints readable best parameters;
- MLflow shows ≥20 trials;
- you can state from memory how many models a grid search with CV would have trained instead.

**Self-check**
- [ ] I can write `fmin` with three `hp` types without docs.
- [ ] I can explain the `SparkTrials` vs MLlib rule.

**Tested by:** DMA 3.5–3.7.

### DBX-MLA.7 Spark ML and pandas Function API on Databricks

⏱ 3.5 h · prereqs: M3.6, DBX-MLA.6

**Skip test**
1. Train one scikit-learn model per store across 2,000 stores in parallel. Which API do you use?
2. What are `applyInPandas` and `mapInPandas` used for?
3. When does Spark ML beat single-node scikit-learn plus a pandas UDF?
4. How is a Spark ML `Pipeline` logged to MLflow?

<details><summary>Answers</summary>

1. `groupBy("store").applyInPandas(train_fn, schema)`.
2. Grouped map (one function per group, flexible output) vs map (iterator of pandas DataFrames, e.g. batch inference).
3. When data doesn't fit one node and the algorithm exists in MLlib.
4. `mlflow.spark.log_model` (or autolog), then load it as pyfunc for scoring.
</details>

**Concepts**
- `applyInPandas` (grouped map), `mapInPandas` (map), cogrouped map.
- Arrow batch sizing.
- Spark ML `Pipeline` on DBR ML logged with MLflow.
- Choosing distributed algorithms vs many single-node models.

**Resources**
- Docs: [pandas function APIs][pandas-fn]: grouped map and map sections.
- Docs: [MLlib on Databricks][mllib]: overview + the binary classification notebook (the GBT regression notebook is left for DBX-MLP.2).
- Academy *Machine Learning Model Development*: the Spark ML / pandas API lessons.

**Exercise.** Grouping by `region` (or any group column in your data), train one model per group with `applyInPandas`. Log each model's metric into the returned DataFrame. **Done when:** the returned DataFrame has one row per group with RMSE, and the run is ≤2 min on Free Edition.

**Self-check**
- [ ] I can write the `applyInPandas` signature, including the output schema.

**Tested by:** DMA 3.3, 3.4, 3.7.

### DBX-MLA.8 Batch and streaming inference

⏱ 3 h · prereqs: DBX-MLA.3, DBX-MLA.7

**Skip test**
1. Score a 50M-row Delta table with a UC model.
2. Score a 10k-row pandas DataFrame in a notebook.
3. Events arrive at tens of thousands per second and compute must autoscale. Which design? (sample Q5)
4. Why not call a serving endpoint per row from a streaming job?

<details><summary>Answers</summary>

1. `mlflow.pyfunc.spark_udf(spark, "models:/…@champion")` in `withColumn`, or `fe.score_batch`.
2. `mlflow.pyfunc.load_model(uri).predict(pdf)`.
3. A DLT (Lakeflow) pipeline that applies the model as a Spark UDF.
4. Network overhead, endpoint limits and cost. The in-cluster UDF scales with the pipeline.
</details>

**Concepts**
- `spark_udf` (with `env_manager`); batch jobs on a schedule.
- pandas batch inference.
- Structured Streaming `readStream` → UDF → `writeStream`.
- The same UDF inside a Lakeflow pipeline table definition. **The Lakeflow product itself is taught in DBX-MLP.5. Here only the inference pattern.**
- `ai_query` for SQL batch against an endpoint → DBX-GAI.1.

**Resources**
- Docs: [Log, load, and register MLflow models][mlflow-models]: "Log and load models" (the `spark_udf` batch/streaming example) and "Log model dependencies" (`env_manager="virtualenv"`). The [batch inference landing page][inference] now covers only AI Functions (`ai_query`, → DBX-GAI.1); skip it.
- Structured Streaming API: no extra reading; use the `readStream`/`writeStream` pattern from the exercise (this file owns it; see Gaps).
- Academy *Machine Learning Model Deployment*: batch and streaming lessons.

**Exercise.** Write `score_batch.py`, which writes `ml_lab.dma.predictions` using `spark_udf` with the `@champion` alias. Then write a streaming variant that reads a Delta table appended in micro-batches (`trigger(availableNow=True)` so it fits serverless). **Done when:** promoting a new champion changes predictions on the next run without a code change.

**Self-check**
- [ ] Given a latency/volume scenario, I can pick batch, streaming or real-time in under 30 s.

**Tested by:** DMA 1.9, 4.1, 4.3, 4.4.

### DBX-MLA.9 Real-time serving for a classic model: exam drill

⏱ 1.5 h · prereqs: DBX-GAI.1 (Model Serving, traffic split, MLflow Deployments)

**Skip test.** If you did DBX-GAI.1's Model Serving lab, answer these three and skip the reading:
1. Deploy a UC sklearn model to a CPU endpoint with scale-to-zero.
2. Split traffic 90/10 between two versions.
3. Query it with `mlflow.deployments`.

**Concepts.** Only the exam framing:
- A served entity per model version; traffic percentages.
- Workload size and scale-to-zero.
- Payload formats `dataframe_split` / `dataframe_records`.

**Resources**
Only if the skip test fails:
- Traffic split: re-read DBX-GAI.1's "Serve multiple models" doc (owned there; do not assign it again). DBX-GAI.1's exercise (step 4) serves and queries a single version only, so this module's exercise is your first hands-on split.
- Docs: [Query custom model endpoints][score-ep]: "Querying methods and examples" (Deployments client tab) and "Supported scoring formats → Pandas DataFrame" (the classic-model payload formats DBX-GAI.1 does not cover).

**Exercise.** Serve `house_price@champion`, add `@challenger` at 10%, then send 200 requests. **Done when:** the endpoint metrics show both entities receiving traffic. Delete the endpoint afterwards, because Free Edition limits active endpoints.

**Self-check**
- [ ] I can write the endpoint config JSON (served entities + traffic) from memory.

**Tested by:** DMA 4.2, 4.5, 4.6.

---

## DBX-MLA.10 Capstone: re-platform M6.10 on Databricks

⏱ 6 h · prereqs: DBX-MLA.1–.9

Port your **M6.10 MLOps capstone** (the MLOps Zoomcamp project) to Databricks. Keep the model and the problem; replace the tooling.

| M6.10 component | Databricks replacement (module) |
|---|---|
| Local Parquet/CSV + pandas prep | UC Delta tables + `prep.py` (DBX-MLA.2) |
| Feast feature repo | UC feature table + `create_training_set` (DBX-MLA.3) |
| Baseline model | AutoML baseline (DBX-MLA.4), then your own model |
| Optuna/HPO | Hyperopt `fmin` (exam API) (DBX-MLA.6) |
| MLflow server + registry aliases (M6.2) | UC registry, champion/challenger via `promote.py` (DBX-MLA.5) |
| Orchestrated batch job | Databricks Job: prep → train → promote → `score_batch.py` (DBX-MLA.8) |
| FastAPI/Docker service | Model Serving endpoint with a traffic split (DBX-MLA.9) |

- [ ] **Done when:** one Job run (≤5 tasks, which fits the Free Edition limit) goes from raw table to updated predictions. The repo has a README mapping each M6.10 part to its Databricks counterpart. One deliberate challenger that is worse than the champion is rejected automatically.

DBX-MLP.9 extends this capstone; keep the code in Git.

---

## DBX-MLA.11 Practice, readiness, exam day

⏱ 3 h · prereqs: all modules

**Practice strategy (official first)**
1. Use the 10 guide sample questions twice: cold in DBX-MLA.0, again after DBX-MLA.9. For every distractor, write why it is wrong.
2. Run the Databricks AI Prep Guide process. That resource is owned by [DBX-GAI](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/databricks/genai-engineer-associate.md): you used Steps 2–3 in DBX-GAI.1, and the Step 4–5 loop is DBX-GAI.10's method (the DMA guide also points to the PDF). You do not need DBX-GAI.10 itself; the steps below are the DMA version.
   - Prime Claude with the DMA guide PDF and generate the "renamed product traps" list and the "hands-on minimum" task list. Diff both against this file.
   - Databricks' own warning: AI mock exams "aren't a calibrated readiness check".
3. **Objective drill.** Write two scenario questions per outline bullet: 96 for 48 bullets. Ask Claude Code to draft them with WebFetch limited to `docs.databricks.com`, then verify each answer against the docs yourself. Answer them 3+ days later.
4. Do not use third-party "practice exams" or dumps. The certification terms forbid "brain-dump material", and the penalty is loss of all Databricks certifications ([terms][terms]).

**Readiness signal to book**
- [ ] 10/10 on the sample questions, with every distractor explained.
- [ ] ≥85% on your own objective drill, and no section below 75%.
- [ ] Capstone done.
- [ ] From memory: the rename table, the `fmin` call, the five `FeatureEngineeringClient` calls, and the alias API.

**Booking.** Check whether a Learning Festival window is open (50% voucher, [festival FAQ][fest]). Re-check the guide two weeks before the exam date, as the guide instructs.

**Final-week review (3 h, scheduled separately)**
- Day 1–2: redo every self-check box.
- Day 3: Section 1 (38%) API recall: write each call on paper.
- Day 4: your drill questions you got wrong.
- Day 5: sample questions plus the rename table.
- Day 6: rest.

**Exam-day tactics**
- 48 scored + ≤10 unscored in 90 min ≈ 1.5 min per item. Flag anything over 2 min.
- Multiple-selection items say how many to pick.
- Answer in the exam's vocabulary: "Delta Live Tables", "online tables", Hyperopt and AutoML are still right answers here.
- Reschedule more than 24 h before an online exam (72 h for a test center) ([FAQ][faq]).

**First-hand traps.** No current-version first-hand report was found: Reddit was inaccessible to the research and the community forum had none. These traps come from the official sample answers:
- `AS FEATURE STORE` SQL doesn't exist.
- Stages are deprecated; use aliases.
- `SimpleImputer` doesn't auto-choose a strategy.
- Use cost-sensitive learning for imbalance.
- A DLT + Spark UDF answers the autoscaling-streaming question.
- `search_runs` + `order_by`, not `get_run`.
- A nightly job means batch.

Some third-party sites say 45 questions; the official count is 48, so treat those sites as unreliable.

---

## Coverage matrix

Every DMA task statement maps to its core owner and this file's module. "Ex" is where it is practised.

| # | Task statement (abridged) | Core | DBX module | Ex |
|---|---|---|---|---|
| 1.1 | MLOps strategy best practices | M6.1 | DBX-MLA.5 | DBX-MLA.5, .10 |
| 1.2 | Advantages of ML runtimes | — | DBX-MLA.1 | DBX-MLA.1 |
| 1.3 | AutoML facilitates model/feature selection | — | DBX-MLA.4 | DBX-MLA.4 |
| 1.4 | AutoML advantages | — | DBX-MLA.4 | DBX-MLA.4 |
| 1.5 | UC account-level feature tables vs workspace | — | DBX-MLA.1, DBX-MLA.3 | DBX-MLA.3 |
| 1.6 | Create a UC feature table | M3.5 | DBX-MLA.3 | DBX-MLA.3 |
| 1.7 | Write to a feature table | — | DBX-MLA.3 | DBX-MLA.3 |
| 1.8 | Train with feature-table features | M3.5 | DBX-MLA.3 | DBX-MLA.3 |
| 1.9 | Score with feature-table features | — | DBX-MLA.3, DBX-MLA.8 | DBX-MLA.3 |
| 1.10 | Online vs offline feature tables | M3.5 | DBX-MLA.3 | DBX-MLA.3 |
| 1.11 | Best run via the MLflow Client API | M6.2 | DBX-MLA.5 | DBX-MLA.5 |
| 1.12 | Manually log metrics/artifacts/models | M6.2 | DBX-MLA.5 | DBX-MLA.5 |
| 1.13 | Information in the MLflow UI | M6.2 | DBX-MLA.5 | DBX-MLA.5 |
| 1.14 | Register a model to UC via the Client | DBX-GAI.1 | DBX-MLA.5 | DBX-MLA.5 |
| 1.15 | UC registry vs workspace registry | DBX-GAI.1 | DBX-MLA.1, DBX-MLA.5 | DBX-MLA.5 |
| 1.16 | Promote code vs promote models | — | DBX-MLA.5 | DBX-MLA.5 |
| 1.17 | Set/remove a model tag | DBX-GAI.1 | DBX-MLA.5 | DBX-MLA.5 |
| 1.18 | Challenger → champion via aliases | M6.2, DBX-GAI.1 | DBX-MLA.5 | DBX-MLA.5, .10 |
| 2.1 | `.summary()` / dbutils summaries | — | DBX-MLA.2 | DBX-MLA.2 |
| 2.2 | Outliers by std or IQR | M2.1 | DBX-MLA.2 | DBX-MLA.2 |
| 2.3 | Visualise categorical/continuous | M2.1 | DBX-MLA.2 | DBX-MLA.2 |
| 2.4 | Compare two features appropriately | M1.3 | DBX-MLA.2 | DBX-MLA.2 |
| 2.5 | Mean vs median vs mode imputation | M3.2 | DBX-MLA.2 | DBX-MLA.2 |
| 2.6 | Impute missing values | M3.2 | DBX-MLA.2 | DBX-MLA.2 |
| 2.7 | One-hot encoding | M3.2, M3.6 | DBX-MLA.2 | DBX-MLA.2 |
| 2.8 | When one-hot is (in)appropriate | M3.2, M2.4 | DBX-MLA.2 | DBX-MLA.2 skip test Q5 |
| 2.9 | When to log-transform | M3.2 | DBX-MLA.2 | DBX-MLA.2 |
| 3.1 | Select an algorithm | M2.1–M2.5 | DBX-MLA.0 | DBX-MLA.11 drill |
| 3.2 | Mitigate class imbalance | M3.1 | DBX-MLA.0 | DBX-MLA.11 drill |
| 3.3 | Estimators vs transformers | M3.6 | DBX-MLA.7 | DBX-MLA.7 |
| 3.4 | Develop a training pipeline | M2.6, M3.6 | DBX-MLA.7 | DBX-MLA.2, .7 |
| 3.5 | Hyperopt `fmin` | — | DBX-MLA.6 | DBX-MLA.6 |
| 3.6 | Grid/random/Bayesian search | M2.1, M4.4 | DBX-MLA.6 | DBX-MLA.6 |
| 3.7 | Parallelise single-node models for tuning | — | DBX-MLA.6, DBX-MLA.7 | DBX-MLA.6, .7 |
| 3.8 | CV vs train/validation split | M2.3 | DBX-MLA.0 | DBX-MLA.11 drill |
| 3.9 | Cross-validation in fitting | M2.3, M2.6 | DBX-MLA.0 | DBX-MLA.11 drill |
| 3.10 | Number of models in grid × CV | M2.3 | DBX-MLA.0 | DBX-MLA.6 done-when |
| 3.11 | Classification metrics | M2.2 | DBX-MLA.0 | DBX-MLA.11 drill |
| 3.12 | Regression metrics | M2.2 | DBX-MLA.0 | DBX-MLA.10 |
| 3.13 | Choose the metric for the objective | M2.2 | DBX-MLA.0 | DBX-MLA.11 drill |
| 3.14 | Exponentiate log-target predictions | M3.2 | DBX-MLA.0 | DBX-MLA.10 |
| 3.15 | Complexity and bias–variance | M2.3 | DBX-MLA.0 | DBX-MLA.11 drill |
| 4.1 | Batch vs real-time vs streaming | M6.5 | DBX-MLA.8 | DBX-MLA.8 |
| 4.2 | Deploy a custom model to an endpoint | DBX-GAI.1 | DBX-MLA.9 | DBX-MLA.9 |
| 4.3 | pandas batch inference | — | DBX-MLA.8 | DBX-MLA.8 |
| 4.4 | Streaming inference with DLT | — | DBX-MLA.8 | DBX-MLA.8 |
| 4.5 | Deploy and query real-time | DBX-GAI.1 | DBX-MLA.9 | DBX-MLA.9 |
| 4.6 | Split traffic between endpoints | DBX-GAI.1 | DBX-MLA.9 | DBX-MLA.9 |

---

## Schedule

| Module | h |
|---|---|
| DBX-MLA.0 | 1.5 |
| .1 | 2.5 |
| .2 | 3 |
| .3 | 4 |
| .4 | 2 |
| .5 | 3 |
| .6 | 3 |
| .7 | 3.5 |
| .8 | 3 |
| .9 | 1.5 |
| .10 capstone | 6 |
| .11 practice + readiness | 3 |
| Final-week review | 3 |
| **Total** | **39 h** |

Without the Databricks AI path, add DBX-GAI.1 (6 h): **45 h**.

**Default (~7 h/week, 6 weeks)**
- W1: .0, .1, .2
- W2: .3, .4
- W3: .5, .6
- W4: .7, .8, .9
- W5: .10
- W6: .11 + final-week review, then book.

**Compressed (~18 h/week, 2 weeks)**
- W1: .0–.6
- W2: .7–.11

Take the exam at the start of week 3.

---

## Combining with other paths

- **Before:** DBX-GAI.1 (6 h; already done if you took the Databricks AI path). Nothing else from the AI path is required. The full DBX-GAI exam is optional and shares UC, MLflow and Model Serving with this path.
- **After:** [DBX-MLP](ml-professional.md) is a pure delta on this file. Book it within ~2 months while the APIs are fresh.
- **With other ML paths:**
  - [AWS-MLA](../aws/mla-c02.md), [AZ-AI300](../azure/ai-300.md) and [GCP-PMLE](../google/professional-ml-engineer.md) cover the same objectives behind this file's DBX-MLA.0 table, mapped to the same M modules. Each costs only its own ~1 h refresh.
  - MLflow habits transfer directly to AZ-AI300 (MLflow in AzureML) and to MLflow on SageMaker in AWS-MLA.
  - Spark skills transfer to EMR and Dataproc framings.
- **Overlap:** no content is repeated. Only the core-refresh table is re-done per path.

## Gaps for the core

- None open for the core. Resolved 2026-10-31: outlier rules and feature-pair comparison are now in M2.1; log-target back-transformation is in M3.2.
- Spark Structured Streaming (`readStream`/`writeStream`/triggers) is tested only by Databricks, so by the ownership rule it belongs to this file (DBX-MLA.8, pattern level); DBX-MLP cites DBX-MLA.8.

## Skipped on purpose

| Item | Reason |
|---|---|
| Instructor-led "Machine Learning with Databricks" | Paid (price not confirmed for this course); the same four courses are free self-paced |
| Academy Labs (USD 200/yr) | Free Edition covers these labs; buy only if you need the hosted notebooks |
| Hyperopt beyond `fmin`/`SparkTrials` | Removed from runtimes after 16.4 LTS ML; exam depth only |
| AutoML forecasting deep dive | One outline mention; the overview is enough |
| Third-party practice banks (Udemy, SkillCertPro, etc.), community quiz repos, Alex Cole guides | Not official, accuracy unknown, some suggest leaked items |
| Exam dumps | Forbidden by Databricks terms; penalty is loss of all certifications |
| Apache Spark Developer Associate / Data Engineer Associate | Not ML certifications; the Spark needed here is in M3.6 and this file |
| Databricks genai-cookbook | Officially deprecated |
| Accreditations (GenAI / Agent / Databricks Fundamentals) | Covered in the DBX-GAI note; redundant here |

## Sources

All accessed 2026-10-31.

[mla-page]: https://www.databricks.com/learn/certification/machine-learning-associate
[mla-pdf]: https://www.databricks.com/sites/default/files/2026-10/Databricks-Certified-Machine-Learning-Associate-Exam-Guide-August-2026.pdf
[faq]: https://www.databricks.com/learn/certification/faq
[terms]: https://www.databricks.com/learn/certification/terms-and-conditions
[nopractice]: https://community.databricks.com/t5/certifications/looking-for-practice-material-for-databricks-certified/td-p/119641
[fest]: https://community.databricks.com/t5/databricks-academy-learners/faq-for-advanced-learning-festival-september-16-october-14-2026/m-p/166272
[labs]: https://community.databricks.com/t5/training-offerings/200-per-year-lab-subscription/td-p/155497
[freeed]: https://www.databricks.com/blog/introducing-databricks-free-edition
[fe-limits]: https://docs.databricks.com/aws/en/getting-started/free-edition-limitations
[dbrml]: https://docs.databricks.com/aws/en/machine-learning/databricks-runtime-ml
[fe-uc]: https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc
[fe-train]: https://docs.databricks.com/aws/en/machine-learning/feature-store/train-models-with-feature-store
[fe-ts]: https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series
[automl]: https://docs.databricks.com/aws/en/machine-learning/automl
[automl-api]: https://docs.databricks.com/aws/en/machine-learning/automl/regression-train-api
[hyperopt]: https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/hyperopt-spark-mlflow-integration
[hyperopt-concepts]: https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/hyperopt-concepts
[hyperopt-bp]: https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/hyperopt-best-practices
[ofs]: https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store
[ray]: https://docs.databricks.com/aws/en/machine-learning/ray/
[mlflow-models]: https://docs.databricks.com/aws/en/mlflow/models
[deploy-patterns]: https://docs.databricks.com/aws/en/machine-learning/mlops/deployment-patterns
[mlops-wf]: https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-workflow
[pandas-fn]: https://docs.databricks.com/aws/en/pandas/pandas-function-apis
[mllib]: https://docs.databricks.com/aws/en/machine-learning/train-model/mllib
[inference]: https://docs.databricks.com/aws/en/machine-learning/model-inference/
[score-ep]: https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints

Link definitions above are the sources: exam facts from the cert page, guide PDF, FAQ and terms; costs from the festival FAQ, the Academy Labs thread and the Free Edition blog; lab limits from the Free Edition limitations page; everything else from the Databricks docs linked per module.
