# Google Cloud Professional Machine Learning Engineer (PMLE): certification path

**Who this is for / prerequisites.** This path is for a senior backend engineer who has finished the [ML core](../../core-ml-engineering.md) (M0–M10) and the [AI core](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/core-llm-app-engineering.md) (C0–C13), and who wants the Google PMLE exam (June-2026 guide). It teaches only what is specific to Google Cloud: product names, the 2026 Agent Platform renames, decision rules for choosing a product, and labs. Every ML or LLM concept is cited by its core module ID and is not taught again here. M3.6 (Spark) and M7.4 (Ray) are required before you start. **Google Cloud basics** are taught once, in the [Google Cloud AI path](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/google/professional-agentic-architect.md). **If you took the Google Cloud AI path, you already have these. Otherwise do just these modules from it:**
- **GCP-PAA.1** Platform foundations, whole module (IAM and service accounts, the Agent Platform map, Model Garden and models as a service (MaaS), the Gemini API, Model Armor basics, Cloud Logging/Trace): 4.5 h, before GCP-PMLE.1.
- **GCP-PAA.8**, Gen AI evaluation service only: from its concept bullet the Gen AI evaluation service, online monitors and quality alerts with their three linked pages, plus the exercise's second checkbox run on your C8.2 cases instead of an ADK eval set: 1.5 h, before GCP-PMLE.5.
- **GCP-PAA.9**, concept bullet only: Cloud Run vs GKE criteria, quotas, the Provisioned Throughput page, caching and batch for cost: 0.5 h, before GCP-PMLE.3.
- **GCP-PAA.10**, the Sensitive Data Protection item and its de-identification page only: 0.25 h, before GCP-PMLE.4.

Total 6.75 h. Exercises are Python only. Hours below do not include these 6.75 h.

## Contents

- [GCP-PMLE.0 Exam at a glance and core refresh](#gcp-pmle0-exam-at-a-glance-and-core-refresh)
- [Modules](#modules) (GCP-PMLE.1 – GCP-PMLE.12)
- [Coverage matrix](#coverage-matrix)
- [Schedule](#schedule)
- [Practice, readiness and exam day](#practice-readiness-and-exam-day)
- [Combining with other paths](#combining-with-other-paths)
- [Gaps for the core](#gaps-for-the-core) · [Skipped on purpose](#skipped-on-purpose) · [Sources](#sources)

---

## GCP-PMLE.0 Exam at a glance and core refresh

**Time:** 2 h · **Prereqs:** ML core, AI core · **Tested by:** PMLE (all sections)

**Skip test:** none. The cold sample-question run below is itself the diagnostic, and every later module's skip test is calibrated against it.

### Exam at a glance (as of 2026-10-31)

| Item | Value | Source |
|---|---|---|
| Name / code | Professional Machine Learning Engineer. Google publishes no exam code | [cert page][cert] |
| Guide version | "Certification exam guide as of June 1, 2026". Updated for "the transition from Vertex AI to Gemini Enterprise Agent Platform" and "prioritizes Google Cloud native solutions" | [guide][guide], [cert page][cert] |
| Status | Active. No 2026 PMLE beta found in research **[unconfirmed: absence of evidence; the third-party testero write-up of the June-2026 changes does not mention a beta]** | [testero] (third party) |
| Price | $200 plus tax | [cert page][cert] |
| Format | 50–60 multiple-choice questions in 2 hours. Third parties also report multiple-select questions **[unconfirmed]** | [cert page][cert] |
| Delivery | Online-proctored (Pearson OnVUE) or at a test center | [cert page][cert], [delivery FAQ][deliv] |
| Languages | English, Japanese | [cert page][cert] |
| Passing score | Not published; results are pass/fail **[unconfirmed: no official 2026 page]** | — |
| Retake waits | Historically 14 / 60 / 365 days **[unconfirmed]**. The official page says only that "the waiting period between attempts still applies" | [delivery FAQ][deliv] |
| Eligibility | No prerequisites. Recommended: 3+ years of industry experience, including 1+ year designing and managing solutions on Google Cloud | [cert page][cert] |
| Validity / renewal | 2 years. The cert page offers "Renewal via Google Skills" ("designated courses and/or skill badges in the renewal learning path", adds 1 year). **Conflict:** the renewal FAQ lists continuing-education renewal only for CDL, ACE, PCA and PDE, and for "all other … professional certifications" gives renewal by exam only (window opens 60 days before the inactive date, adds 2 years) **[unconfirmed which applies to PMLE]** | [cert page][cert], [renewal FAQ][renew] |
| Coding | "does not directly assess coding skill". Reading-level Python and SQL is enough | [cert page][cert], [guide][guide] |
| Section weights | S1 low-code AI ~13% · S2 data and models across teams ~16% · S3 scaling prototypes ~21% · S4 serving and scaling ~20% · S5 pipelines ~18% · S6 monitoring ~13% (previous guide: 13/14/18/20/22/13, from the research diff; the Wayback copy rate-limited a re-check on 2026-10-31) | [guide][guide], [2025 guide (Wayback)][guide25] |
| Official prep | [Sample questions][sq] (the form shows "Page 1 of 10"; Google says they "do not represent the range of topics or level of difficulty"); [learning path 17][lp17] (17 activities, ~57.75 nominal h) | [cert page][cert] |

**What changed in June 2026 (from a diff of the two guides):** **removed:** TFX, TensorBoard, Spark kernels, Dataproc notebooks, Reduction Server, Horovod, hosting MLflow, hybrid/multicloud pipelines, Jenkins, RAG via Agent Builder (RAG and agents now belong to the Agentic Architect exam (PAA)); the old standalone 5.3 "Tracking and auditing metadata" is folded into 2.3. **Added:** fine-tuning Gemini with BigQuery, Imagen and Veo, cost/latency/availability optimisation of Gemini apps, LLM-as-a-judge, Model Armor and malicious prompting, Ray on Agent Platform, Managed Service for Apache Airflow, CI/CD/CT.

- [ ] Read the [guide][guide] end to end (0.5 h) and print the bullet list as your checklist.
- [ ] Take the [sample questions][sq] cold (0.5 h). Note every miss. This is your baseline, so don't study for it.
- [ ] Skim the "Already covered" table and do only the listed refresh actions (1 h total).

**Done when** `misses.csv` holds every baseline sample-question miss, each tagged with its guide § and the core or GCP-PMLE module that covers it.

### Already covered by the cores (light refresh only)

Each row is an exam objective whose concept a core module owns. The refresh action is the only thing you do here. The Google product side of each row is taught in the GCP-PMLE module named in the [coverage matrix](#coverage-matrix).

| Exam objective (guide §) | Core owner | Refresh action (≤10 min each) |
|---|---|---|
| Model type choice incl. ARIMA vs DNN vs LLM (3.1) | M10.1, M8.2, C1.5 | Re-read your M10.1 framing notes and FPP ARIMA chapter summary |
| Interpretability-driven modelling (3.1), explainability (6.1) | M9.2 | Recall when SHAP vs permutation importance vs PDP applies |
| Data types, EDA, ingestion, engine by scale (2.1, 3.2) | M2.1, M3.1, M3.3, M3.6 | List row vs column formats, batch vs stream; when Spark beats pandas |
| Feature engineering / selection (1.1) | M3.2 | Re-run one encoding drill |
| PII and privacy (2.1) | C9.4, C9.5 | Redaction vs masking vs pseudonymisation |
| Feature store, PIT joins, skew (2.1, 4.2, 5.1) | M3.5 | Explain a PIT join in two sentences |
| Experiment tracking, lineage, registry (2.3, 4.1) | M6.2 | MLflow run vs model version vs alias |
| Predictive metrics (2.3) | M2.2, M6.3 | PR-AUC vs ROC-AUC under imbalance |
| GenAI evaluation, LLM-as-a-judge, GenAI monitoring (2.3, 6.2) | C3.2, C8.1–C8.3 | Re-read your judge-calibration notes |
| Training failures, HPO (3.2, 4.2) | M4.3, M4.4 | Tuning Playbook "scientific approach" summary |
| When to fine-tune foundation models (1.2, 3.2) | C11.1, C11.2 | One-paragraph prompt → RAG → FT decision |
| CPU/GPU/TPU, parallelism (3.3, 4.2) | M7.1, M7.3 | Sketch DDP vs FSDP vs tensor-parallel memory |
| Serving modes, containers, pre/post-processing, autoscaling (4.1, 4.2) | M6.5, M6.6, C10.4 | Batch vs online vs streaming trade-offs |
| Canary, A/B, shadow (4.1) | M6.7 | What each one measures |
| Validation, pipelines, CI/CD/CT, retraining (5.1, 5.2) | M3.4, M6.1, M6.4, M6.7, M6.9, M7.4 | Redraw your M6.10 DAG; MLOps level 0/1/2 |
| Drift types, skew, attribution drift (6.2) | M6.8 | Statistic per data type (PSI/KS/chi-square) |
| Bias monitoring, fairness (6.1) | M9.1 | Demographic parity vs equalized odds |
| Malicious prompting, exfiltration (6.1) | C9.1, C9.2, C9.4 | OWASP LLM01/LLM02 notes |
| Gemini app cost/latency/availability (1.2) | C4.1, C4.2, C10.3, C10.5 | Re-read your C10.5 cost sheet |
| Vision, document, translation, image/video generation (1.2) | C12.1, C12.3, C12.4 | One line each: VLM vs parser for documents; LLM vs specialised API for translation |

---

## Modules

Each module opens with a **Skip test**. If you pass it, skip the reading and do only the exercise and the self-check. Lab costs: Google Skills labs use credits. Starter = 35 free credits/month, Pro = $29/month ([plans][skills-plans], as of 2026-10-31). New Google Cloud customers get $300 in free credit ([free tier][free], as of 2026-10-31; the page does not state how long the credit lasts, so check at sign-up **[unconfirmed: commonly reported as 90 days]**). **Before your first exercise, set a project budget alert (I suggest $50; this is my estimate, not a Google figure), and undeploy every endpoint after each exercise.**

### GCP-PMLE.1 Platform foundations for ML on Agent Platform

**Time:** 4 h · **Prereqs:** GCP-PAA.1 (or the Google Cloud AI path) · **Tested by:** PMLE all sections

**Skip test:** (1) Give the 2026 names of Vertex AI Workbench, Vertex AI Prediction, Cloud Composer and Dataproc Serverless. (2) Which service account does a custom training job run as by default, and why should you override it? (3) In which region-scoped resources must the training data bucket, BigQuery dataset and pipeline root live, and why?

- **Concepts:** the ML surface of Agent Platform (the agent surface is GCP-PAA.1). Rename table you must read fluently:

  | Old name | 2026 name |
  |---|---|
  | Vertex AI Workbench | Agent Platform Workbench |
  | Vertex AI Prediction | Agent Platform Inference |
  | Vertex AI Pipelines / Feature Store / Model Registry / Experiments / ML Metadata / Model Monitoring | "Agent Platform …" plus the same component name |
  | Cloud Composer | Managed Service for Apache Airflow ([docs][airflow]) |
  | Dataproc Serverless | Managed Service for Apache Spark (serverless) ([docs][spark]) |

  Other concepts: project/region layout for ML (GCS bucket, BigQuery dataset, Artifact Registry repo, pipeline root in one region); a user-managed service account per workload (training, pipeline, serving), with roles via IAM (→ GCP-PAA.1); quotas for GPUs/TPUs; the Python SDK (still `pip install google-cloud-aiplatform` / `from google.cloud import aiplatform` in the renamed Agent Platform docs, as of 2026-10-31), `google-cloud-bigquery`, `kfp`; the CLI group is still `gcloud ai …`; cost hygiene (endpoints bill while deployed).
- **Resource:** Agent Platform docs: [training overview][train] (first section only) and the [Model Registry introduction][registry]. Learning path 17: "A Tour of Google Cloud Hands-on Labs" (45 min).
- **Exercise:** `bootstrap.py` (uses `subprocess` + `gcloud`) creates the bucket, BigQuery dataset, Artifact Registry repo and three service accounts with least-privilege roles, then sets the budget alert. **Done when** `python check_env.py` prints project, region, SAs with their roles, bucket and budget, and exits 0.
- **Self-check:** [ ] Translate five pre-2026 blog headings into 2026 names. [ ] Explain why a pipeline SA needs `iam.serviceAccountUser` on the training SA.

### GCP-PMLE.2 Low-code ML: BigQuery ML, AutoML and Tabular Workflows

**Time:** 7 h · **Prereqs:** .1, M2, M3.2, M8.2 · **Tested by:** PMLE 1.1, 3.1, 3.2

**Skip test:** (1) Write `CREATE MODEL` for a logistic regression with a `TRANSFORM` clause, from memory. (2) Which BigQuery ML (BQML) model type forecasts many time series at once with holidays? (3) When does BQML beat AutoML, and when is it the other way round? (4) How does a BQML model reach an online endpoint?

- **Concepts:** BQML model families: internal (linear/logistic, k-means, matrix factorization, PCA, contribution analysis, ARIMA_PLUS / ARIMA_PLUS_XREG; plus the built-in TimesFM model called through `AI.FORECAST`, in Preview, with no `CREATE MODEL`), externally trained via Agent Platform (DNN, Wide & Deep, autoencoder, boosted trees, random forest, AutoML), remote models, and imported ONNX/TF/TFLite/XGBoost ([BQML intro][bqml]); `TRANSFORM` makes feature engineering travel with the model; `ML.EVALUATE`, `ML.PREDICT`, `ML.FORECAST`, `ML.EXPLAIN_PREDICT`; registering BQML models in Model Registry (`MODEL_REGISTRY` option, with `VERTEX_AI_MODEL_VERSION_ALIASES`); AutoML tabular (classification/regression/forecasting) vs Tabular Workflows (managed pipelines with control over architecture search and cost). Exam heuristic: data already in BigQuery + SQL team + standard model type → BQML; best accuracy with no ML code → AutoML; custom architecture → custom training (.6).
- **Resource:** BigQuery docs: [BQML intro][bqml] (whole), [TRANSFORM][bqml-transform], [inference overview][bq-infer], [model management with Agent Platform][bq-registry], [end-to-end journeys][bq-e2e]; Agent Platform [AutoML tabular overview][automl] + [Tabular Workflows overview][tabwf]. Learning path 17: skill badges "Create ML Models with BigQuery ML" and "Engineer Data for Predictive Modeling with BigQuery ML" (do the challenge labs, ~2 h total).
- **Exercise:** `bqml_lab.py` (Python + `google-cloud-bigquery`) on a public dataset trains logistic regression and boosted trees with `TRANSFORM`, forecasts one series with ARIMA_PLUS, runs `ML.EXPLAIN_PREDICT`, and registers the best model in Model Registry. Optional: one AutoML tabular run on the same table, capped at the minimum training budget. **Done when** a Markdown table compares AUC, training time and cost for BQML vs AutoML, and the model is visible in Model Registry.
- **Self-check:** [ ] State the product you would pick for "analysts, data in BigQuery, churn model by Friday". [ ] Explain why `TRANSFORM` prevents training–serving skew (→ M3.5).

### GCP-PMLE.3 Google AI APIs and foundation-model tuning (Gemini, Imagen, Veo)

**Time:** 6 h · **Prereqs:** .1, GCP-PAA.1 (Model Garden, Gemini API), GCP-PAA.9 (quotas, Provisioned Throughput), C11.1–C11.2, C12.1, C12.4 · **Tested by:** PMLE 1.1 (Gemini via BigQuery), 1.2, 3.2 (FM tuning)

**Skip test:** (1) Invoices in 12 layouts, fields must be extracted: Document AI processor or Gemini prompt? Give the reasons. (2) Write the BigQuery SQL that fine-tunes a Gemini remote model. (3) Name the three hyperparameters of Gemini supervised tuning. (4) Give four levers that cut Gemini app cost or latency without retraining.

- **Concepts:**
  - Pre-trained API selection: Document AI processors are Digitize (Enterprise OCR), Extract (Form Parser, layout parser, custom extractor including the generative one, pretrained parsers) and Classify (custom classifier, custom splitter) ([Document AI][docai]). Vision API and Translation API fit when the task is standard and determinism, cost or latency matter more than flexibility (→ C12.3).
  - Gemini supervised fine-tuning: JSONL data, and the hyperparameters adapter size, epochs and learning-rate multiplier ([SFT][sft], [use SFT][sft-use]). Preference tuning ([docs][pref]) and open-model tuning in Model Garden ([docs][openft]). Fine-tuning Gemini from BigQuery: `CREATE MODEL … REMOTE WITH CONNECTION … AS SELECT` plus `ML.EVALUATE` (BLEU/ROUGE → C3.2) ([tune-evaluate][bq-tune]).
  - Imagen and Veo: generation and customization are product choices, and the concepts are in C12.4. Know the names: Imagen "Customize images" (subject customization, style customization, controlled and instruct customization) ([Imagen][imagen], still under the pre-rename "Generative AI on Vertex AI" docs path) and Veo text/image/reference-to-video, extend and edit ([Veo][veo]). Skim the navigation only (~15 min).
  - Gemini app optimisation (1.2) as an exam framing:

    | Lever | Concept owner | Google product name to recognise |
    |---|---|---|
    | Smaller model tier | C3.4 | Gemini Flash / Flash-Lite vs Pro tiers |
    | Prompt/context caching | C10.3 | Context caching |
    | Batch inference | C4.1 | Batch inference with Gemini |
    | Reserved capacity | C10.4 | Provisioned Throughput |
    | Availability | C4.2 | Global vs regional endpoints **[verify name]** |
    | Tuning instead of long prompts | C11.1 | Supervised tuning |

    The SDK call and global/regional endpoints are in GCP-PAA.1; quotas, Provisioned Throughput, and caching and batch for cost are in GCP-PAA.9.
- **Resource:** the pages linked above. Learning path 17: skill badge "Prepare Data for ML APIs on Google Cloud" (do the challenge lab only, ~1 h).
- **Exercise:** `doc_pipeline.py` sends 5 sample PDFs through the Document AI Form Parser and translates one extracted field with the Translation API. Then do the [tune-evaluate][bq-tune] tutorial with ≤100 rows and `max_iterations` lowered (watch the cost). **Done when** the extracted JSON validates against a Pydantic schema (→ C2.3) and an `ML.EVALUATE` table shows the tuned model vs the base model.
- **Self-check:** [ ] One-line rule: Document AI vs Gemini for extraction. [ ] When would you tune Gemini instead of adding few-shot examples (cost per request vs tuning cost)?

### GCP-PMLE.4 Data preparation at scale and Feature Store

**Time:** 6 h · **Prereqs:** .1, M3.3–M3.6, GCP-PAA.10 (Sensitive Data Protection) · **Tested by:** PMLE 2.1, 3.2, 4.2, 5.1

**Skip test:** (1) 5 TB of clickstream needs nightly preprocessing, streaming too: which service? (2) What does Beam `MLTransform` guarantee across training and serving? (3) Name the offline store of the current Feature Store, the online serving type to choose today, and why not the other one. (4) Where does PII redaction happen in a Dataflow pipeline?

- **Concepts:** engine choice by scale and complexity: BigQuery SQL (data already in the warehouse; [BigQuery DataFrames][bqdf] for pandas-style code), Dataflow/Beam (batch + streaming, `RunInference`, `MLTransform` for identical preprocessing in training and serving, enrichment transforms) ([Dataflow ML][dfml]), Managed Service for Apache Spark (existing Spark code, serverless batches) ([docs][spark]), in-memory pandas (small data). Organising data: tabular in BigQuery; images, audio and video in GCS with manifest tables. PII: Sensitive Data Protection inspection/de-identification of the training data before training (the service and its de-identification methods → GCP-PAA.10). Feature Store (latest): BigQuery is the offline store; feature groups and features; feature views synced to an online store; historical feature values for point-in-time training sets ([Feature Store][fs]). Online serving types ([docs][fs-online]): **Bigtable online serving** (TB-scale, scheduled and continuous sync) is the current choice; **Optimized online serving is deprecated** (no new features from 2026-05-17, APIs sunset 2027-02-17), so expect older material and possibly exam distractors to use it.
- **Resource:** [Dataflow ML][dfml] (overview, `MLTransform`, `RunInference` pages); [Feature Store overview][fs] + [online serving types][fs-online] + "Create a feature view instance" + "Serve historical feature values" in its left nav. Learning path 17: "MLOps with Agent Platform: Manage Features" (labs, ~3 h of the 5 h nominal).
- **Exercise:** `features/pipeline.py`: a Beam pipeline with `MLTransform` (scale + vocabulary) runs locally on DirectRunner, then once on Dataflow, and writes to BigQuery. Register the table as a feature group, create a Bigtable online store and a feature view, fetch online features for 3 entity IDs, then delete the online store (it bills while it exists). **Done when** the same `MLTransform` artifact is loaded by both training and serving code, and an online fetch returns values identical to the offline table.
- **Self-check:** [ ] A decision table of 4 scenarios → engine. [ ] Explain how Feature Store prevents label leakage (→ M3.5).

### GCP-PMLE.5 Notebooks, experiments, lineage and model evaluation

**Time:** 5 h · **Prereqs:** .1, M6.2, M6.3, C8.1, GCP-PAA.8 (Gen AI evaluation service) · **Tested by:** PMLE 2.2, 2.3

**Skip test:** (1) Colab Enterprise vs Workbench: name two deciding factors. (2) Three notebook security controls an enterprise expects. (3) What does ML Metadata record that Experiments doesn't? (4) Where do model bias metrics live on Agent Platform?

- **Concepts:** Colab Enterprise (managed, collaborative, runtime templates) vs Agent Platform Workbench (customisable JupyterLab instances) ([comparison][nb-compare]); security: service account vs end-user credentials, VPC Service Controls, CMEK, idle shutdown, no public IP; frameworks in notebooks (PyTorch, sklearn; JAX → Gaps); prototyping with Model Garden models in notebooks (→ GCP-PAA.1 for Model Garden itself). Experiments (runs, params, metrics, time series, autologging) vs ML Metadata (artifacts, executions, lineage graph) vs Pipelines (repeatable runs) as the "environment for experimentation" choice ([Experiments][exp], [ML Metadata][mlmd]). Model evaluation on Agent Platform for predictive models, incl. [data and model bias metrics][bias]; Gen AI evaluation service and custom LLM-judge metrics (→ GCP-PAA.8, concepts C8.1).
- **Resource:** the pages above, plus [model evaluation introduction][eval]. Learning path 17: "MLOps with Agent Platform: Model Evaluation" (2.5 h; do the labs, skim the videos).
- **Exercise:** in Colab Enterprise, `experiments.ipynb` logs 3 runs (sklearn, XGBoost, small PyTorch MLP) to Experiments, uploads the best to Model Registry, and runs a model evaluation with a bias slice. **Done when** the Experiments comparison view shows 3 runs and the lineage graph links dataset → run → model.
- **Self-check:** [ ] Map MLflow concepts (M6.2) one-to-one onto Experiments/ML Metadata/Model Registry. [ ] Pick a notebook option for a regulated team that needs CMEK and VPC-SC.

### GCP-PMLE.6 Custom training, tuning and hardware at scale

**Time:** 7 h · **Prereqs:** .1, .5, M4.3–M4.4, M7.1, M7.3, M7.4 · **Tested by:** PMLE 3.2, 3.3, 5.1 (Ray)

**Skip test:** (1) Prebuilt vs custom container for training: when is custom required? (2) How does a training script report the metric to the hyperparameter tuning service? (3) Name the worker-pool roles of a distributed job. (4) Give two signals that favour TPU over GPU and two that favour GPU. (5) A job dies after 9 h on Spot VMs: what should already have been in place?

- **Concepts:** serverless custom training jobs: prebuilt containers (your Python package) vs custom containers (any framework or dependency) ([training overview][train]); worker pools (primary, workers, parameter server, evaluator) and machine/accelerator specs; Spot VMs and reservations; persistent resources for iteration; checkpoints to GCS so preemption is survivable. Hyperparameter tuning service: Bayesian optimisation by default (grid and random search selectable), parallel vs sequential trials trade-off, metric reporting with the `cloudml-hypertune` package ([HPO][hpo]). Distributed training on GPUs/TPUs ([distributed][dist], [TPU training][tpu]); Kubeflow on GKE as the unmanaged alternative; Ray on Agent Platform: cluster create, BigQuery integration, deploy ([Ray][ray]). Troubleshooting: quota errors, OOM (batch size or sharding → M7.3), stragglers, data-loader bottlenecks (→ M7.1), container start failures.
- **Resource:** the pages above (overview + "configure" subsections only). Learning path 17: "Production Machine Learning Systems", only the module on high-performance/distributed training (~2 h). The [course page][pmls] (16 h, "updated 2 months ago") confirms it covers "distributed training options" and is TensorFlow-centric, so read it for Google concepts only **[module title not shown on the page; unconfirmed]**.
- **Exercise:** `train/task.py` (PyTorch) is packaged for a prebuilt container and submitted with `aiplatform.CustomJob` on a Spot VM with checkpointing. Then run a 6-trial HPO job (2 parallel) on learning rate and weight decay. Optional: one Ray cluster running a Ray Tune sweep from M7.4, deleted right after. **Done when** the best trial's model artifact is in GCS and in Model Registry, and you resumed one deliberately killed run from its checkpoint.
- **Self-check:** [ ] Fill the CPU/GPU/TPU table for: XGBoost on 10 GB, ViT fine-tune, 7B LLM SFT, nightly sklearn retrain. [ ] Name the first two things to check when HPO trials all report the same metric.

### GCP-PMLE.7 Serving and scaling models

**Time:** 6 h · **Prereqs:** .2, .6, M6.5–M6.7, C10.4, GCP-PAA.9 (Cloud Run vs GKE) · **Tested by:** PMLE 4.1, 4.2

**Skip test:** (1) How do you run a canary between two model versions on one endpoint? (2) When do you use batch inference instead of an endpoint? (3) Private endpoint options and why. (4) A PyTorch model needs tokenisation before inference and thresholding after: which container type? (5) What sets autoscaling limits?

- **Concepts:** online inference on endpoints (shared public, dedicated public, dedicated private via Private Service Connect, plus older private-services-access endpoints) ([online][online], [dedicated][dedicated], [PSC][psc]); batch inference from BigQuery/GCS with no endpoint ([batch][batch]); prebuilt serving containers (sklearn, XGBoost, PyTorch, TF) vs custom containers carrying pre/post-processing ([prebuilt][prebuilt], [custom][custom]); traffic split across deployed models for canary/A-B, plus rolling deployments ([rolling][rolling]); min/max replicas and utilisation targets ([autoscaling][autoscale]); GPU/TPU serving; Model Garden one-click deploys of open models on vLLM-style containers (concept → C10.4); Cloud Run or GKE when you need full control (services → GCP-PAA.1 compute map; Cloud Run vs GKE criteria → GCP-PAA.9); edge = export to a portable format (TFLite/ONNX → M7.2); Model Registry versions and aliases (default alias) ([registry][registry]); online features from Feature Store (.4).
- **Resource:** the pages above.
- **Exercise:** `serve/` reuses the FastAPI container from M6.5 as a custom serving container (with pre/post-processing). Deploy v1 and v2 of the .6 model to one endpoint at 90/10, send 500 requests with a Python load script, then run a batch inference job from a BigQuery table. Undeploy everything. **Done when** the endpoint metrics show the 90/10 split, the batch output table exists, and `gcloud ai endpoints list` / `describe` (the CLI group the current docs still use) shows no deployed models.
- **Self-check:** [ ] Say which option is cheapest for 1M nightly predictions with no latency SLA. [ ] Explain why aliases beat hard-coded version IDs in pipelines.

### GCP-PMLE.8 Pipelines, orchestration and CI/CD/CT

**Time:** 8 h · **Prereqs:** .4–.7, M6.1, M6.4, M6.9 · **Tested by:** PMLE 5.1, 5.2, 2.3 (Pipelines as experiment env)

**Skip test:** (1) Agent Platform Pipelines vs Managed Airflow vs Ray: which one for an ML DAG, for a cross-system data DAG, and for distributed Python? (2) What are Google Cloud Pipeline Components? (3) How does a pipeline gate model registration on evaluation? (4) What does Cloud Build do in CT vs CD?

- **Concepts:** Agent Platform Pipelines run KFP v2 pipelines: components, artifacts, caching ([caching][pipe-cache]), conditional steps, schedules via the scheduler API ([schedule][pipe-sched]), templates and the template gallery ([intro][pipes], [build][pipes-build], [GCPC][gcpc], [templates][pipe-tmpl]); lineage lands in ML Metadata. Managed Service for Apache Airflow for cross-system scheduling, where it triggers pipelines rather than replacing them ([docs][airflow]). Ray on Agent Platform for distributed Python steps (.6). Validation steps: data checks before training (→ M3.4), evaluation-threshold gates before registering. Consistent preprocessing: the same `TRANSFORM`/`MLTransform` artifacts (.2, .4). CI/CD/CT with Cloud Build: triggers on git push, `cloudbuild.yaml` steps run tests, build the container into Artifact Registry, compile and submit the pipeline ([Cloud Build][cb]). Retraining policy (schedule vs drift-triggered vs performance-triggered → M6.7) wired to Model Monitoring alerts (.9) through a Pub/Sub-triggered pipeline run ([trigger with Pub/Sub][pipe-pubsub]).
- **Resource:** the pages above. Learning path 17: "Machine Learning Operations (MLOps): Getting Started", labs only (~1.5 h).
- **Exercise:** `pipeline.py` (KFP v2) runs validate data → train (the `CustomJob` from .6) → evaluate → `dsl.If(auc > threshold)` → register with alias `candidate`. Add a `cloudbuild.yaml` triggered on push that runs pytest, builds the image, compiles and submits the pipeline, and a weekly schedule. **Done when** a git push produces a pipeline run that registers a new version, and a second run with poisoned data stops at the gate.
- **Self-check:** [ ] Map MLOps level 0/1/2 onto these services. [ ] Explain why caching can hide a data change, and how to disable it per step.

### GCP-PMLE.9 Monitoring, explainability, responsible AI and security

**Time:** 5 h · **Prereqs:** .7, .8, M6.8, M9.1–M9.2, C9.1–C9.4, GCP-PAA.1, GCP-PAA.8, GCP-PAA.10 (slices listed at the top) · **Tested by:** PMLE 6.1, 6.2

**Skip test:** (1) Model Monitoring v2: which objectives can it watch, what baseline does each need, and when must you use v1 instead? (2) Which attribution method suits a tree ensemble, and which suits an image CNN? (3) What is the status of Vertex Explainable AI? (4) Three layers that stop a user from extracting PII through a Gemini app.

- **Concepts:** Model Monitoring has two offerings ([overview][mm], [setup][mm-setup]): **v1** is GA and is configured on Agent Platform endpoints (Google says to use it when you need production-level support for an endpoint-deployed model); **v2** is in Preview, attaches monitors to a model version, and runs scheduled or on-demand jobs. v2 objectives: input feature drift and output inference drift (Jensen-Shannon divergence; L-infinity for categorical features) and feature attribution (SHAP values). Skew = drift measured against the training data as baseline; drift = against an earlier production window. Thresholds and alerts. Explainability: Sampled Shapley (non-differentiable models, tabular), Integrated Gradients (differentiable), XRAI (images), example-based explanations ([XAI][xai]). **Trap:** Google's page says "Vertex Explainable AI is deprecated as of March 16, 2026… no longer available on or after March 16, 2027" ([XAI][xai], as of 2026-10-31), yet guide §6.1 still lists "Model explainability on Agent Platform". Learn the methods for the exam, and in practice prefer BQML `ML.EXPLAIN_PREDICT` ([BQ XAI][bq-xai]) or SHAP in your container (→ M9.2). Bias: model/data bias metrics (.5) plus slice monitoring. GenAI risks: regex/output filters, Gemini safety filters, Model Armor (→ GCP-PAA.1) and Sensitive Data Protection (→ GCP-PAA.10) (concepts → C9.1, C9.4). GenAI monitoring and continuous evaluation → C8.3 + GCP-PAA.8 (online monitors, quality alerts).
- **Resource:** the pages above. Learning path 17: "Responsible AI for Developers: Interpretability & Transparency" and "…: Privacy & Safety", Google-tool labs only (~1.5 h total).
- **Exercise:** `monitor.py` redeploys the .7 model, creates a monitor for skew + drift (v1 on the endpoint, or v2 on the model version; note which you chose and why), then sends a shifted request batch generated with NumPy. Add an `ML.EXPLAIN_PREDICT` report for the BQML model from .2. **Done when** the monitoring job flags the shifted feature, a one-page note says which retraining trigger (.8) fires, and the endpoint is undeployed again.
- **Self-check:** [ ] Pick metric + baseline for skew vs drift. [ ] Name the explainability option you'd ship after 2027-03-16.

### GCP-PMLE.10 Product-selection drills (exam framing)

**Time:** 4 h · **Prereqs:** .2–.9 · **Tested by:** PMLE 3.1 and cross-section scenario questions

**Skip test:** score ≥ 90% on 20 random cards from a colleague's (or your own earlier) deck without notes.

- **Concepts:** the guide "prioritizes Google Cloud native solutions" ([cert page][cert]), so the right answer is usually the **most managed service that meets the constraint**. Read the constraint words in each question ("minimal code", "lowest latency", "data already in BigQuery", "explainable", "least operational overhead", "regulated"). Core decisions: model type (ARIMA vs DNN vs LLM → M10.1) × product (BQML vs AutoML vs custom vs Gemini/API) × deployment (batch vs endpoint vs Cloud Run/GKE) × interpretability (linear/BQML/explanations vs black box).
- **Resource:** your own notes from .2–.9. No new reading.
- **Exercise:** `drills/cards.yaml` (≥ 60 scenario → answer → why-not-the-others cards, one per guide bullet minimum) + `quiz.py` (random 20, records misses to `misses.csv`). **Done when** two consecutive runs score ≥ 90%.
- **Self-check:** [ ] Every guide bullet has at least one card. [ ] Every "why not" cites a constraint, not a feeling.

### GCP-PMLE.11 Capstone: re-platform the M6.10 project on Google Cloud

**Time:** 12 h · **Prereqs:** .1–.9, M6.10 · **Tested by:** PMLE 2–6 end to end

**Skip test:** none. This is the readiness evidence.

Take your M6.10 MLOps capstone (the version upgraded with a DL model) and move it onto Google Cloud services without changing the model logic:

| M6.10 piece | Google replacement |
|---|---|
| Local files / Postgres | BigQuery tables + GCS |
| Feature code | `MLTransform` or BQML `TRANSFORM` + Feature Store feature view (.4) |
| MLflow tracking/registry | Experiments + Model Registry with aliases (.5, .7) |
| Orchestrator from M6.4 | KFP v2 on Agent Platform Pipelines (.8) |
| Training script | `CustomJob` + HPO on Spot (.6) |
| FastAPI/KServe serving | Endpoint with 90/10 canary + batch inference (.7) |
| Evidently | Model Monitoring (v1 or v2) → retraining trigger (.9) |
| GitHub Actions | Cloud Build trigger (.8). Keep GitHub as the repo |

- [ ] **Done when** a git push retrains, gates, registers and canaries a new version end to end; injected drift raises an alert; `README.md` shows the architecture, the cost of one full run, and a teardown script that leaves no billable resources.
- [ ] Optional: add a BQML baseline (.2) and justify, in one paragraph, whether it would have been enough.

### GCP-PMLE.12 Practice exams and final review

**Time:** 6 h · **Prereqs:** .10, .11 · **Tested by:** PMLE all

**Skip test:** none. **Exercise:** a timed run (below) logged to `misses.csv`. **Done when** the readiness signal holds. See [Practice, readiness and exam day](#practice-readiness-and-exam-day). Hours: second pass of the sample questions with reasoning (1 h), the miss-driven re-read of guide bullets (3 h), two timed drill sessions of 50 cards in 100 minutes (2 h).

---

## Coverage matrix

Every bullet of the June-2026 guide → where the concept lives (core) and where the Google side is taught.

| § | Guide bullet (abridged) | Core | GCP-PMLE |
|---|---|---|---|
| 1.1 | Build BQML/AutoML models by problem | M2, M8.2 | .2 |
| 1.1 | Feature engineering/selection in BQML | M3.2 | .2 |
| 1.1 | Predictions with BQML | M6.5 | .2 |
| 1.1 | Train with AutoML | M2.1 | .2 |
| 1.1 | Fine-tune Gemini with BigQuery | C11.1–C11.2 | .3 |
| 1.2 | Select model from Model Garden | C3.4, C1.5 | GCP-PAA.1, .3 |
| 1.2 | Industry APIs (Document AI, Vision, Translate) | C12.1, C12.3 | .3 |
| 1.2 | Tune Gemini/Imagen/Veo/MaaS | C11.2, C12.4 | .3 |
| 1.2 | Optimise Gemini apps: cost/latency/availability | C10.3, C10.5, C4.1–C4.2 | .3, GCP-PAA.1, GCP-PAA.9 |
| 2.1 | Organise/explore data types | M2.1, M3.3 | .4 |
| 2.1 | Preprocessing tool by scale | M3.3, M3.6 | .4 |
| 2.1 | Feature Store features | M3.5 | .4 |
| 2.1 | Data privacy / PII | C9.4, C9.5 | .4, GCP-PAA.10 |
| 2.2 | Notebook collaboration/security | — | .5 |
| 2.2 | Develop in Workbench/Colab Enterprise (PyTorch, sklearn, JAX) | M4.3, M2 | .5 |
| 2.2 | Model Garden models in notebooks | C1.5 | .5, GCP-PAA.1 |
| 2.3 | Choose experimentation environment | M6.2, M6.4 | .5, .8 |
| 2.3 | Evaluate predictive and gen AI (LLM-as-judge) | M2.2, M6.3, C8.1 | .5, GCP-PAA.8 |
| 2.3 | Track artifacts, versions, lineage | M6.2 | .5 |
| 3.1 | Choose model type (ARIMA, DNN, LLM) | M10.1, M8.2 | .10 |
| 3.1 | Choose product (AutoML, BQML, Pipelines) | — | .2, .10 |
| 3.1 | Choose deployment strategy | M6.5 | .7, .10 |
| 3.1 | Modelling for interpretability | M9.2 | .9, .10 |
| 3.2 | Organise training data (GCS, BigQuery) | M3.1 | .4 |
| 3.2 | Ingest structured/unstructured data | M3.3 | .4 |
| 3.2 | Train via custom training, Kubeflow on GKE, AutoML, Tabular Workflows | M4.3 | .6, .2 |
| 3.2 | Troubleshoot training failures | M4.4 | .6 |
| 3.2 | Hyperparameter tuning | M4.4 | .6 |
| 3.2 | Fine-tune foundation models; when to tune | C11.1–C11.2 | .3 |
| 3.3 | CPU/GPU/TPU evaluation | M7.1 | .6 |
| 3.3 | Distributed training, data/model parallelism | M7.3 | .6 |
| 4.1 | Batch and online inference (Agent Platform, Model Garden, Cloud Run, GKE) | M6.5, C10.4 | .7, GCP-PAA.1, GCP-PAA.9 |
| 4.1 | Prebuilt/custom containers (PyTorch, XGBoost) | M6.5 | .7 |
| 4.1 | Model Registry versioning | M6.2 | .7 |
| 4.1 | A/B and canary rollout | M6.7 | .7 |
| 4.1 | Inference pre/post-processing | M6.5 | .7 |
| 4.2 | Serve features from Feature Store | M3.5 | .4, .7 |
| 4.2 | Public and private endpoints | — | .7 |
| 4.2 | Serving hardware incl. edge | M7.1, M7.2 | .7 |
| 4.2 | Scale serving backend by throughput | M6.6, C10.4 | .7 |
| 4.2 | Tune models for training and serving in prod | M4.4, M7.2 | .6, .7 |
| 5.1 | Validate data and models | M3.4, M6.9 | .8 |
| 5.1 | Pipelines, Managed Airflow, Ray | M6.4, M7.4 | .8, .6 |
| 5.1 | Consistent preprocessing train/serve | M3.5 | .2, .4, .8 |
| 5.2 | Retraining policy | M6.7 | .8 |
| 5.2 | CI/CD/CT with Cloud Build | M6.9, M6.1 | .8 |
| 6.1 | Secure AI: exfiltration, malicious prompting, Model Armor, regex, safety filters | C9.1, C9.2, C9.4 | .9, GCP-PAA.1, GCP-PAA.10 |
| 6.1 | Responsible AI, bias monitoring | M9.1, C9.5 | .5, .9 |
| 6.1 | Explainability on Agent Platform | M9.2 | .9 |
| 6.2 | Model Monitoring continuous evaluation | M6.8 | .9 |
| 6.2 | Skew, data/concept/attribution drift | M6.8 | .9 |
| 6.2 | Monitor/test/evaluate gen AI | C8.2, C8.3 | .9, GCP-PAA.8 |

Every row ends in a hands-on exercise (the module's exercise or .11) and a self-check (the module's self-check or .10 cards).

---

## Schedule

| Module | Hours |
|---|---|
| .0 | 2 |
| .1 | 4 |
| .2 | 7 |
| .3 | 6 |
| .4 | 6 |
| .5 | 5 |
| .6 | 7 |
| .7 | 6 |
| .8 | 8 |
| .9 | 5 |
| .10 | 4 |
| .11 | 12 |
| .12 | 6 |
| **Total** | **78 h** (+ 6.75 h of GCP-PAA.1 and the GCP-PAA.8/.9/.10 slices if you skipped the Google Cloud AI path; list at the top) |

**Default (~7 h/week, 12 weeks):** W1 .0 + .1 (6) · W2 .2 (7) · W3 .3 (6) · W4 .4 (6) · W5 .5 + start .10 cards (5 + 2) · W6 .6 (7) · W7 .7 (6) · W8 .8 (8) · W9 .9 + finish .10 (5 + 2) · W10 .11 (7) · W11 .11 + start .12 (5 + 2) · W12 .12 (4) + exam = 78 h.

**Compressed (~13 h/week, 6 weeks):** W1 .0–.2 (13) · W2 .3–.4 (12) · W3 .5–.6 (12) · W4 .7–.8 (14) · W5 .9–.10 + .11 start (5 + 4 + 4) · W6 .11 finish + .12 (8 + 6) = 78 h.

Add .10 cards continuously: write 3–5 cards at the end of every module.

---

## Practice, readiness and exam day

**Practice strategy (official and free first).**
1. [Sample questions][sq]: cold in .0 (baseline), then again in .12 explaining why each wrong option is wrong. They are few and Google says they don't show the full difficulty, so don't treat a score as a forecast.
2. Skill-badge challenge labs (.2, .3) are the official hands-on check.
3. Your .10 deck is the main drill.
4. Third-party practice tests only if they are dated **after June 2026** and use Agent Platform names. Most still test TFX/Horovod/Vertex naming. Price and quality are unverified. Never use dumps (ExamTopics-style): they breach the candidate agreement and are stale against the June-2026 guide.

**Readiness signal (book when all four hold):** [ ] .11 runs end to end · [ ] two consecutive .10 quizzes ≥ 90% · [ ] all sample questions right *with reasons* · [ ] you can fill the product-choice and rename tables from memory. Book ~2 weeks ahead.

**Final week:** day 1–2 re-read the guide bullets you missed in `misses.csv`; day 3 rename table + BQML syntax + Monitoring objectives; day 4 a timed 50 cards in 100 min; day 5 serving/pipelines decision tables; day 6 light review; day 7 rest.

**Exam-day tactics:** ~2 min per question. Mark long scenarios and come back. Underline the constraint ("minimal effort", "lowest cost", "data in BigQuery", "real time"). Prefer the managed, Google-native answer that meets it. Eliminate options that name removed or legacy tools (TFX, Reduction Server, Jenkins, self-hosted MLflow) unless the question forces them. For online delivery, review the testing requirements and check your machine against Pearson's system requirements the day before: Google says a session lost to connectivity or an unmet system requirement still counts as an attempt ([delivery FAQ][deliv]).

**Traps found first-hand (research, as of 2026-10-31):**
- Pre-June-2026 courses, blogs and practice sets use Vertex AI names and dropped tools, and learning path 17 is only partly renamed. Answer options use the 2026 names from the .1 rename table (e.g. Managed Service for Apache Airflow, Managed Service for Apache Spark).
- Vertex Explainable AI is deprecated (2026-03-16, gone 2027-03-16) but still in guide §6.1.
- Feature Store Optimized online serving is deprecated (sunset 2027-02-17); Bigtable online serving is the current type ([docs][fs-online]).
- Model Monitoring v2 is Preview and v1 is GA for endpoints ([overview][mm]). Don't assume "v2 replaced v1".
- RAG and agents left PMLE for PAA. A RAG-design answer is rarely the PMLE answer.
- Older community reports say the exam is product-centric ("intermediate++"; Pipelines, endpoints, Feature Store, Monitoring dominate). These come from the pre-2026 version **[unconfirmed for June-2026]**.

---

## Combining with other paths

- **GCP-PAA (ai-engineering) first:** recommended. Every GCP-PAA module GCP-PMLE needs (GCP-PAA.1 plus the .8, .9 and .10 slices listed at the top) is then already done, and so are the GenAI rows in the matrix. Together: AI core → ML core → GCP-PAA → GCP-PMLE.
- **With AWS-MLA, AZ-AI300 or DBX-MLA/DBX-MLP:** no content overlap beyond each file's `.0` refresh. Each teaches its own services against the same M modules. Re-platform the **same** M6.10 capstone each time; by the second cloud I estimate the capstone takes ~half the hours because only the service mapping changes (my estimate, not measured).
- **With NV-GENL:** independent. GCP-PMLE.6 hardware choices complement NV-GENL's GPU depth. Nothing is repeated.
- **Order suggestion if you only want one ML cert:** pick the cloud your employer runs. PMLE has the widest model-to-ops span and a 2-year validity ([cert page][cert]).

---

## Gaps for the core

- JAX (named in guide 2.2 as a notebook framework) has no core owner. Here it is only a name to recognise.

---

## Skipped on purpose

| Item | Reason |
|---|---|
| Learning path 17: "Introduction to AI and Machine Learning on Google Cloud" (8 h) | Overlaps M2/M6 and .1. Its product tour is covered by the docs pages in .1–.2 |
| Learning path 17: "Introduction to Generative AI", "Introduction to Large Language Models", "MLOps for Generative AI" | Fully covered by C1, C8.4 |
| Learning path 17: "Create Generative AI Apps on Google Cloud" (4 h) | App building is C core + GCP-PAA |
| Learning path 17: "Responsible AI for Developers: Fairness & Bias" (4 h) | M9.1 covers it; only Google-tool labs of the other two RAI courses are kept (.9) |
| Learning path 17: "Production Machine Learning Systems" beyond the distributed-training module | M6/M7 cover it; TensorFlow-centric |
| Learning path 17: "Build a Certification Study Guide: PMLE" (1 h) | .0 + .10 do the same job |
| Kaggle "Operationalizing Generative AI on Vertex AI using MLOps" whitepaper | Pre-rename naming. Concepts are in C8.4 + M6.1 |
| Professional Data Engineer, Associate Data Practitioner | Data-engineering exams. Their ML bullets are a subset of .2/.4 |
| Pre-June-2026 PMLE material (TFX, TensorBoard, Reduction Server, Horovod, MLflow on GCP) | Removed from the guide |
| Third-party video courses for PMLE | None verified as updated for June 2026 |
| *Official Google Cloud Certified Professional Machine Learning Engineer Study Guide* (book, named on the [cert page][cert]) | Paid; whether its current edition covers the June-2026 guide and Agent Platform names is unverified. The guide + docs + .10 deck do the job |
| Exam dumps | Breach of the candidate agreement; stale |

---

## Sources

All accessed 2026-10-31. Every URL is in this file's link definitions (view the raw Markdown).

- **Exam:** [cert], [guide], [guide25], [sq], [lp17], [pmls], [deliv], [renew], [testero] (third party), [skills-plans], [free]
- **BigQuery:** [bqml], [bqml-transform], [bq-infer], [bq-xai], [bq-registry], [bq-e2e], [bq-tune], [bqdf]
- **AutoML / tuning / APIs:** [automl], [tabwf], [sft], [sft-use], [pref], [openft], [docai], [imagen], [veo]
- **Data:** [dfml], [spark], [airflow], [fs], [fs-online]
- **Notebooks / experiments / evaluation:** [nb-compare], [exp], [mlmd], [eval], [bias]
- **Training:** [train], [hpo], [dist], [tpu], [ray]
- **Serving:** [registry], [online], [batch], [psc], [dedicated], [prebuilt], [custom], [rolling], [autoscale]
- **Pipelines / CI:** [pipes], [pipes-build], [gcpc], [pipe-tmpl], [pipe-cache], [pipe-sched], [pipe-pubsub], [cb]
- **Monitoring / explainability:** [mm], [mm-setup], [xai]

[cert]: https://cloud.google.com/learn/certification/machine-learning-engineer
[guide]: https://services.google.com/fh/files/misc/professional_machine_learning_engineer_exam_guide_english_new.pdf
[guide25]: https://web.archive.org/web/20250813082528/https://services.google.com/fh/files/misc/professional_machine_learning_engineer_exam_guide_english.pdf
[sq]: https://docs.google.com/forms/d/e/1FAIpQLSeYmkCANE81qSBqLW0g2X7RoskBX9yGYQu-m1TtsjMvHabGqg/viewform
[lp17]: https://www.skills.google/paths/17
[pmls]: https://www.skills.google/course_templates/17
[deliv]: https://support.google.com/cloud-certification/answer/9907748?hl=en
[renew]: https://support.google.com/cloud-certification/answer/9907853?hl=en
[testero]: https://testero.ai/blog/pmle-exam-changes-june-2026
[skills-plans]: https://www.skills.google/payments/new
[free]: https://cloud.google.com/free
[bqml]: https://docs.cloud.google.com/bigquery/docs/bqml-introduction
[bqml-transform]: https://docs.cloud.google.com/bigquery/docs/bigqueryml-transform
[bq-infer]: https://docs.cloud.google.com/bigquery/docs/inference-overview
[bq-xai]: https://docs.cloud.google.com/bigquery/docs/xai-overview
[bq-registry]: https://docs.cloud.google.com/bigquery/docs/managing-models-vertex
[bq-e2e]: https://docs.cloud.google.com/bigquery/docs/e2e-journey
[bq-tune]: https://docs.cloud.google.com/bigquery/docs/tune-evaluate
[bqdf]: https://docs.cloud.google.com/bigquery/docs/bigquery-dataframes-introduction
[automl]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/classification-regression/overview
[tabwf]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/tabular-data/tabular-workflows/overview
[sft]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/supervised-tuning
[sft-use]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/supervised-tuning/use
[pref]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/preference-tuning
[openft]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/open-model-tuning
[docai]: https://docs.cloud.google.com/document-ai/docs/overview
[imagen]: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/image/overview
[veo]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/overview
[dfml]: https://docs.cloud.google.com/dataflow/docs/machine-learning
[spark]: https://docs.cloud.google.com/managed-spark/docs/serverless-overview
[airflow]: https://docs.cloud.google.com/composer/docs/composer-3/composer-overview
[fs]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/featurestore/latest/overview
[fs-online]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/featurestore/latest/online-serving-types
[nb-compare]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/notebooks/workbench/notebook-solution
[exp]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/experiments/intro-vertex-ai-experiments
[mlmd]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/ml-metadata/introduction
[eval]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/evaluation/introduction
[bias]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/evaluation/model-bias-metrics
[train]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/training/overview
[hpo]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/training/hyperparameter-tuning-overview
[dist]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/training/distributed-training
[tpu]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/training/training-with-tpu-vm
[ray]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/open-source/ray
[registry]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/model-registry/introduction
[online]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/get-online-predictions
[batch]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/get-batch-predictions
[psc]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/private-service-connect
[dedicated]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/using-dedicated-endpoints
[prebuilt]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/pre-built-containers
[custom]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/use-custom-container
[rolling]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/rolling-deployment
[autoscale]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/autoscaling
[pipes]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/pipelines/introduction
[pipes-build]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/pipelines/build-pipeline
[gcpc]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/pipelines/components-introduction
[pipe-tmpl]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/pipelines/create-pipeline-template
[pipe-cache]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/pipelines/configure-caching
[pipe-sched]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/pipelines/schedule-pipeline-run
[pipe-pubsub]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/pipelines/trigger-pubsub
[cb]: https://docs.cloud.google.com/build/docs/overview
[mm]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/model-monitoring/overview
[mm-setup]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/model-monitoring/set-up-model-monitoring
[xai]: https://docs.cloud.google.com/vertex-ai/docs/explainable-ai/overview
