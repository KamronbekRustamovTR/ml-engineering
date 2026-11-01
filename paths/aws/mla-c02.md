# AWS Certified Machine Learning Engineer – Associate (MLA-C02): study path

**Who this is for / prerequisites.** You have finished the ML core ([core-ml-engineering.md](../../core-ml-engineering.md)) and the AI core ([core-llm-app-engineering.md](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/core-llm-app-engineering.md)). This path teaches only the AWS delta: SageMaker AI, the AWS data services, Code\* CI/CD, and ML cost and security on AWS. MLA-C02 has 107 skills, 45 of them added in C02, and about 40 of those 45 are GenAI, agent or FM skills (Bedrock, Knowledge Bases, agents, Guardrails, evaluations) [counted from the guide's domain pages and the comparison page, 2026-10-31]. This file does not teach those. It maps them to the AI core and to the AWS-AIP modules of the AWS AI path ([aip-c01.md](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/aws/aip-c01.md)) that teach each service. **If you took the AWS AI path, you already have these. Otherwise do just these modules from it: AWS-AIP.1 (7 h), AWS-AIP.2 (5 h), AWS-AIP.3 (6 h), AWS-AIP.4 (4 h), AWS-AIP.5 (6 h), AWS-AIP.6 (6 h), AWS-AIP.7 (4 h), AWS-AIP.8 (4 h), AWS-AIP.9 (4 h), AWS-AIP.10 (5 h), AWS-AIP.11 (4 h): 55 h, in that order, before AWS-MLA.1.** Exercises are Python only (SageMaker Python SDK v3, boto3).

## Contents

- [Exam at a glance](#exam-at-a-glance)
- [Already covered (core + AWS AI path)](#already-covered-core--aws-ai-path)
- [Modules](#modules): AWS-MLA.0 – AWS-MLA.13
- [Coverage matrix](#coverage-matrix)
- [Schedule](#schedule)
- [Practice and readiness](#practice-and-readiness)
- [Add-on: AWS MLOps Demonstrated](#add-on-aws-mlops-demonstrated)
- [Combining with other paths](#combining-with-other-paths)
- [Gaps for the core](#gaps-for-the-core)
- [Skipped on purpose](#skipped-on-purpose)
- [Sources](#sources)

---

## Exam at a glance

All facts are as of 2026-10-31. [unconfirmed] means AWS has not published the fact. Check the cert page again before you book.

| Item | Value | Source |
|---|---|---|
| Exam | AWS Certified Machine Learning Engineer – Associate, **MLA-C02** (the cert page prints "ME1-C02", which is almost certainly a typo) | [cert page][cert], [guide][guide] |
| Status | **Beta.** Registration opened 2026-09-01 and delivery started 2026-09-29, English only. **GA delivery is 2027-01-14** per the September 2026 blog, although the cert page still says "TBD". No beta end date has been published | [cert page][cert], [Sep-2026 blog][blog] |
| Beta format | USD 75, 85 questions, 170 min, Pearson VUE test centre or online proctored. You can attempt a beta only once | [cert page][cert], [after-testing policy][after] |
| Standard format | 65 questions (50 scored, 15 unscored), multiple choice and multiple response, scaled score 100–1,000, **pass at 720**, compensatory scoring. Unanswered questions count as wrong. The guide gives no duration or price [unconfirmed; MLA-C01 was 130 min and USD 150] | [guide][guide], [cert page][cert] |
| Beta results | The cert page says beta results are "typically available within 5 business days of exam completion". The guide's footnote says the pass/fail designation "does not apply to the beta version" | [cert page][cert], [guide][guide] |
| Domains | D1 Data Preparation for ML and AI 28% · D2 ML Model and FM Development 24% · D3 Deployment and Orchestration of ML and AI Workflows 24% · D4 Operating, Monitoring, and Securing ML and AI Solutions 24% | [guide][guide] |
| Recommended experience | At least 1 year with SageMaker AI, Bedrock and other AWS ML services, plus 1 year in a related role (for example backend developer) | [guide][guide] |
| Out of scope (job tasks) | End-to-end solution architecture, ML strategy, wide integration, deep work in 2+ ML domains | [guide][guide] |
| Validity and renewal | 3 years. Renew by passing the latest MLA, by passing **AIP-C01**, or through Skill Builder maintenance (1 year, needs a subscription) | [recertification][recert], [cert page][cert] |
| Retakes | Wait 14 days after a fail. Every attempt costs the full fee | [after-testing policy][after] |
| Discount | Passing any AWS exam gives a 50% voucher for your next exam. If you pass AIP-C01 first, you get MLA at half price | [benefits][benefits] |
| MLA-C01 | The English version ended 2026-09-28. Japanese, Korean and Simplified Chinese stay until MLA-C02 GA. Do not study C01 | [cert page][cert], [comparison][cmp] |
| Closed SageMaker features | SageMaker **Debugger, Clarify and Model Monitor are "no longer open to new customers"** (existing customers keep them). The C02 guide names none of the three, but its skills (2.3.2, 2.3.4, 2.3.5, 1.3.4, 4.1.3) are what they did, and C01 tested them heavily, so learn them by name for the exam. Do the labs with AWS's documented replacements (AWS-MLA.4, .6). When they closed and whether C02 still asks about them [unconfirmed] | [Debugger][dbg-change], [Clarify][clarify-change], [Model Monitor][mm-change] availability-change pages |

**Verdict: take MLA-C02 at GA (2027-01-14 or later) on the default schedule.** Take the USD 75 beta only if you finish the compressed schedule while it is still open.

---

## Already covered (core + AWS AI path)

These exam skills are owned elsewhere. **Refresh action:** redo the owning module's self-check (≈1 h for all rows). For the **AWS side** column: if you took the AWS AI path, you already have every AWS-AIP module named. Otherwise do the 11 AWS-AIP modules listed at the top of this file (55 h) in [aip-c01.md](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/aws/aip-c01.md) before AWS-MLA.1. This file adds only the AWS-MLA modules listed.

| Core owner | MLA-C02 skills | AWS side |
|---|---|---|
| M2.1–M2.4, M10.1 | 2.1.3, 2.1.6, 2.2.5 (fitting), 2.2.6 | AWS-MLA.4 |
| M3.1–M3.4 | 1.2.4, 1.2.6, 1.3.1–1.3.3, 1.3.5, 1.3.7 | AWS-MLA.3; Ground Truth → AWS-AIP.9 |
| M3.3, M3.6 | 1.1.5, 1.1.6 | AWS-MLA.2 |
| M4.2–M4.4, M7.3 | 2.2.3, 2.2.4, 2.2.7, 2.3.5 | AWS-MLA.4, .5 |
| M5.2, M5.7–M5.9 | 1.2.6, 1.2.9, 2.2.9, 2.3.8 | AWS-MLA.8 |
| M6.2, M6.3, M6.8 | 2.3.2, 2.3.6, 3.3.6, 4.1.2, 4.1.3 | AWS-MLA.5, .6 |
| M6.5, M6.7, M6.9 | 3.1.1, 3.1.3, 3.2.3, 3.3.1, 3.3.4, 3.3.5, 4.1.4 | AWS-MLA.7, .9 |
| M9.1, M9.2 | 1.3.3, 1.3.4, 2.3.4 | AWS-MLA.6 |
| C1.5, C3.4, C13.1 | 2.1.1, 2.1.4 | AWS-AIP.1 (Bedrock model catalogue); AWS-MLA.11 |
| C2.1, C2.2, C3.3 | 2.2.8, 2.3.1, 3.3.7, 3.3.9 | AWS-AIP.4 (Prompt Management); AWS-AIP.11 (Bedrock evaluations) |
| C3.2, C8.1, C13.2 | 2.3.6–2.3.9 | AWS-AIP.11 (human and LLM-judge evaluation); AWS-AIP.9 (A2I) |
| C5.2, C5.4, C5.5, C6.1, C6.2 | 3.1.6, 3.2.9, 3.2.11, 4.1.5 | AWS-AIP.5 (Bedrock Agents, Strands); AWS-AIP.6 (AgentCore incl. Memory and A2A hosting) |
| C7.1–C7.5 | 1.1.7, 1.2.5, 1.2.7, 2.1.5, 2.2.9, 2.3.10, 3.1.8, 3.2.7, 3.2.8, 3.3.11 | AWS-AIP.3 (Knowledge Bases, vector stores, embeddings, ingestion triggers); AWS-AIP.11 (RAG evaluation) |
| C8.2–C8.4 | 3.3.8, 4.1.1, 4.1.5, 4.1.6, 4.2.2 | AWS-AIP.10 (CloudWatch GenAI observability, Bedrock metrics, X-Ray tracing); AWS-AIP.6 (AgentCore Observability, agent deployment); AWS-AIP.5 (agent versions and aliases); AWS-AIP.11 (Bedrock Model Evaluation) |
| C9.1–C9.5 | 1.2.8, 1.3.6, 4.3.2, 4.3.3, 4.3.7–4.3.9 | AWS-AIP.1 (IAM, KMS, API keys vs IAM); AWS-AIP.4 (Comprehend PII); AWS-AIP.7 (Guardrails, Macie) |
| C10.3–C10.5 | 2.1.7, 3.1.4, 3.2.1, 3.2.10, 4.2.7–4.2.10 | AWS-AIP.1 (on-demand, Provisioned Throughput, batch, cross-Region); AWS-AIP.10 (token and cost monitoring); AWS-MLA.8, .10 |
| C11.1–C11.3 | 1.2.9, 1.3.6, 2.1.2, 2.2.5 (forgetting), 2.2.8 | AWS-AIP.9 (Bedrock fine-tuning and distillation basics); AWS-MLA.8 |
| — (shared AWS services: basics in the AI path, ML delta here) | 1.1.6 (Glue Data Catalog, Lake Formation) · 1.2.1, 1.3.1 (Glue Data Quality, Data Wrangler, SageMaker Processing) · 3.1.3 (real-time/serverless/async endpoints) · 3.3.6 (Model Registry basics) · 3.3.2 (CodePipeline/CodeBuild) · 4.2.5 (Cost Explorer, allocation tags) | AWS-AIP.8 · AWS-AIP.4 · AWS-AIP.9 · AWS-AIP.9 · AWS-AIP.2 · AWS-AIP.10; the depth is in AWS-MLA.2, .3, .7, .5, .9, .10 |

---

## Modules

Each module lists time, prerequisites, the exam skills it covers, a skip test, AWS-only concepts, resources, an exercise and a self-check. **Lab budget:** [AWS Free Tier][free] credits cover the labs (as of 2026-10-31: USD 100 up front, up to USD 100 more, on a 6-month free plan). Set a USD 20 Budgets alert on day 1. End every lab with a teardown that deletes endpoints, monitor schedules and online stores, and stops MLflow servers.

### AWS-MLA.0 Exam at a glance and core refresh
- [ ] **3 h** · prereqs: ML core, AI core · tested by: all domains
- **Skip test:** none. Everyone does this module.
- **Do:** read the [guide][guide] and its 4 domain pages. Then read the [C01→C02 comparison][cmp]: its additions and deletions define your scope. Then read the [in-scope][inscope], [out-of-scope][outscope] and [service-mentions][mentions] pages (the exam uses short service names). Finish with the refresh above.
- **Exercise:** turn the coverage matrix into a checklist. **Done when** every skill has a box and an owner ID.
- **Self-check:** name 3 skills C02 removed (Neo/edge, model-size reduction, BYOC) and 2 services that AIP uses but MLA puts out of scope (Cognito, AppConfig).

### AWS-MLA.1 SageMaker AI platform foundations
- [ ] **5 h** · prereqs: AWS-MLA.0; AWS-AIP.1 IAM/KMS/VPC basics, or your own AWS experience · tested by: 3.2.4, 3.2.5, 4.3.2, 4.3.3, 4.3.6
- **Skip test:** (1) Which data input mode streams from S3 without copying first, and which one copies to the instance's volume? (2) What does `EnableNetworkIsolation` block, and what does it still allow? (3) A training job inside a private VPC can't read S3. Name two fixes. (4) Which IAM role does a training job assume, and who passes it? (5) Where do you test a training script locally without paying for instances?
- **Concepts:** Studio domain and user profiles; execution role and `iam:PassRole`; anatomy of a training job (container image, S3 channels, `/opt/ml/*` paths, model artifact to S3); input modes **File / FastFile / Pipe**; SageMaker Python SDK v3 (`ModelTrainer`, `ModelBuilder`) vs boto3 vs CLI; local mode; `VpcConfig` for training jobs and endpoints, S3 gateway endpoint and SageMaker API/runtime interface endpoints, network isolation, inter-container traffic encryption, KMS keys for volumes and outputs.
- **Resource:** SageMaker AI Developer Guide: [Train a model (how it works)][sm-train], [setting up training jobs to access datasets][sm-data], [give SageMaker access to resources in your VPC][sm-vpc], [internet-free mode][sm-iso] (≈2.5 h).
- **Exercise:** in Python, launch an sklearn script-mode training job twice, first in local mode and then on `ml.m5.large` with FastFile input, a VPC config and a customer-managed KMS key. **Done when** both runs produce `model.tar.gz`, and the cloud run's CloudTrail `CreateTrainingJob` event shows the VPC and KMS settings.
- **Self-check:** why does a network-isolated container fail a runtime `pip install`, and what is the fix?

### AWS-MLA.2 Data ingestion and storage on AWS
- [ ] **6 h** · prereqs: M3.3, M3.6; AWS-AIP.8 (Glue Data Catalog, Lake Formation) · tested by: 1.1.1–1.1.6, 1.1.8, 1.2.3
- **Skip test:** (1) What are the differences between Kinesis Data Streams, Data Firehose and Managed Service for Apache Flink, and which one converts to Parquet on delivery? (2) Data Streams throws `ProvisionedThroughputExceeded`. Give two remedies. (3) Glue crawler vs Glue ETL job vs EMR: when do you use each? (4) Images and audio for training: where do they go, and how do you index them?
- **Concepts:**
  - Storage: S3 storage classes and Glacier; choosing EBS, EFS or FSx.
  - Sources: extracting from RDS, DynamoDB (export to S3) and OpenSearch; SQL extraction with Athena and Redshift.
  - Batch processing: crawlers that populate the Glue Data Catalog, and Glue Spark ETL; EMR and EMR Serverless. The catalog as a lineage register and Lake Formation's fine-grained access → AWS-AIP.8.
  - Streaming: **Data Streams** (shards), **Firehose** (buffering, Lambda transform, format conversion), **Managed Flink** (stateful windows), MSK; Lambda for per-record transforms.
  - Also: DataSync for on-premises copies; capacity troubleshooting (hot shards, Lambda concurrency, Glue DPUs).
- **Resource:** the overview and key-concepts pages of [Kinesis Data Streams][kds], [Firehose][fh], [Managed Flink][flink] and [Glue][glue] (≈2 h).
- **Exercise:** in Python, push 10k synthetic events through Firehose (Lambda transform → Parquet → S3), crawl them with Glue and query them with Athena via boto3. **Done when** an Athena `SELECT count(*)` matches the events sent and the Parquet files are partitioned by date.
- **Self-check:** a near-real-time requirement with no custom consumer code points to Firehose. Sub-second processing with custom consumers points to Data Streams.

### AWS-MLA.3 Transformation, features and data quality
- [ ] **6 h** · prereqs: AWS-MLA.2, M3.2, M3.4, M3.5; AWS-AIP.4 (what Glue Data Quality, Data Wrangler and SageMaker Processing are, and when to use each) · tested by: 1.1.9, 1.2.1, 1.2.2, 1.2.4, 1.2.8, 1.3.1, 1.3.7
- **Skip test:** (1) Data Wrangler vs Glue DataBrew vs a Glue Spark job: which suits which user and which scale? (2) Name the two Feature Store stores and say which one answers a point-in-time training query. (3) How do you express "column `age` is complete and between 0 and 120" in Glue Data Quality? (4) Which Glue ML transform finds duplicates when no key matches exactly? (5) How does a feature reach the online store from a streaming source?
- **Concepts:** **Data Wrangler** (the docs say it "has been integrated into Amazon SageMaker Canvas"; the standalone version is a Studio Classic feature) flows and export targets; **DataBrew** recipes, profile jobs and no-code transforms; **Glue Data Quality** (basics → AWS-AIP.4): writing DQDL rules and using a ruleset to gate ETL ingestion; Glue **FindMatches** for fuzzy dedup; the Glue Studio **Detect PII** transform ([detect and process sensitive data][glue-pii]: detect, redact, partially redact or SHA-256 hash; the redaction concept itself → C9.4); **Feature Store**: feature groups, record identifier and event time, online vs offline (S3 + Glue catalog), `PutRecord` ingestion, time-travel queries with Athena. This is the managed counterpart of Feast (M3.5).
- **Resource:** [Feature Store][fs] and [offline store format][fs-off]; [Glue Data Quality][gdq]; [DataBrew: What is][databrew]; [Data Wrangler][dw] (overview + "Transform" + "Export") (≈2.5 h).
- **Exercise:** reproduce your M3.5 Feast point-in-time join with SageMaker Feature Store. Run [`mlops/sagemaker_core_feature_store_introduction`][smx], then adapt it to your data with a Glue Data Quality ruleset that gates ingestion. **Done when** the Athena time-travel query returns the same training frame as your Feast version, and a deliberately bad batch fails the DQ ruleset.
- **Self-check:** why is the offline store, not the online store, the right source for a training set?

### AWS-MLA.4 Training on SageMaker AI
- [ ] **8 h** · prereqs: AWS-MLA.1, M2.4, M4.3, M7.3 · tested by: 2.1.3, 2.1.6, 2.2.1, 2.2.2, 2.2.4–2.2.7, 2.3.5, 3.3.3
- **Skip test:** (1) Pick a built-in algorithm for each: churn on tabular data, IP-address anomaly, multi-step demand forecast, unlabelled anomaly scores, word embeddings. (2) What must your script do to survive spot interruption? (3) Data-parallel vs model-parallel: which one for a model that doesn't fit on one GPU? (4) Which Debugger built-in rules catch vanishing gradients and a loss that isn't decreasing, and what does AWS now tell new accounts to use instead of Debugger? (5) What do managed warm pools save?
- **Concepts:**
  - Built-in algorithm choice: XGBoost, LightGBM, Linear Learner, AutoGluon-Tabular, Factorization Machines, K-Means, PCA, Random Cut Forest, IP Insights, DeepAR, BlazingText, vision algorithms. Know each one's input formats.
  - Script mode with framework containers.
  - **Managed spot training** (the docs claim savings "up to 90%"): checkpoint to S3 and set `MaxWaitTimeInSeconds` (`max_wait`) larger than the max runtime. Spot never falls back to on-demand; savings = 1 − `BillableTimeInSeconds`/`TrainingTimeInSeconds`.
  - Distributed data-parallel library; heterogeneous clusters; warm pools.
  - **Debugger** rules and profiler by name (exam vocabulary), plus early stopping. Debugger is closed to new customers. AWS maps it to managed MLflow (metrics, runs), TensorBoard on SageMaker (gradients, weights) and CloudWatch metrics and alarms (training diagnostics, system profiling).
  - Model combination: stacking, or cascades for cost (→ C4.2).
- **Resource:** [built-in algorithms][algos] (selection table + the XGBoost and DeepAR pages); [managed spot training][spot]; [distributed training][dist] + [data-parallel library][ddp]; [built-in rules][dbg-rules] (rule names only) + [Debugger availability change][dbg-change] (capability-mapping table) + [training-job CloudWatch metrics][train-metrics]; [warm pools][warm] (≈3 h). Run the examples [`training/sm-managed_spot_training_xgboost`][smx] and `training/sm-regression_xgboost`.
- **Exercise:** train your M6.10 model twice: built-in XGBoost on spot with checkpoints, then a PyTorch script-mode job that logs `train:loss` through a metric definition (regex) and to managed MLflow, with a CloudWatch alarm on that metric (the AWS replacement for Debugger's `LossNotDecreasing` rule). **Done when** the spot job's `BillableTimeInSeconds` is below its `TrainingTimeInSeconds`, and a deliberately bad learning rate drives the loss alarm to `ALARM` while MLflow shows the diverging run next to a healthy one.
- **Self-check:** which feature fits "cheapest training, interruptions OK"? Which fits "slow start between iterations"?

### AWS-MLA.5 Tuning, experiments and model registry
- [ ] **6 h** · prereqs: AWS-MLA.4, M4.4, M6.2 · tested by: 2.2.3, 2.3.1, 3.3.6
- **Skip test:** (1) Grid vs random vs Bayesian vs Hyperband in AMT: when do you pick each? (2) What does warm start reuse? (3) Managed MLflow on SageMaker comes as tracking servers and as MLflow Apps. What is the difference? (4) How does a model version move from `PendingManualApproval` to deployment automatically? (5) How do MLflow Model Registry and SageMaker Model Registry relate?
- **Concepts:** **AMT** strategies; Hyperband only for iterative algorithms that report intermediate metrics; early stopping of tuning jobs; warm start (identical data/algorithm vs transfer learning); objective-metric regex; Autopilot/Canvas AutoML by name only; **managed MLflow** (tracking-server sizes, artifact store in your S3, IAM actions under the `sagemaker-mlflow` prefix, auto-registration to Model Registry); **Model Registry** (versions and approval status as the rollout and rollback gate → AWS-AIP.9): model package groups, the approval-status transitions, metadata and lineage, approval events in EventBridge.
- **Resource:** [AMT][amt] + [tuning strategies][amt-how]; [managed MLflow][mlflow] (all sub-sections up to "Integrate MLflow with your environment"); [Model Registry][reg] (≈2 h).
- **Exercise:** run a 12-job Bayesian AMT on your AWS-MLA.4 XGBoost job, with every trial logged to a managed MLflow tracking server. Auto-register the best model, then approve it with boto3. **Done when** MLflow shows 12 runs, the registry has version 1 with `Approved`, and an EventBridge rule captured the approval event in CloudWatch Logs.
- **Self-check:** which services answer "repeatable experiments plus an audit trail from data to model"? (Experiments in Studio Classic is legacy.)

### AWS-MLA.6 Clarify and Model Monitor (exam) and their replacements (labs)
- [ ] **6 h** · prereqs: AWS-MLA.5, M6.8, M9.1, M9.2 · tested by: 1.3.3–1.3.5, 2.3.2, 2.3.4, 4.1.2, 4.1.3
- **Skip test:** (1) Name 2 pre-training and 2 post-training Clarify bias metrics. (2) What are the 4 Model Monitor types? (3) Which endpoint feature must be enabled before any monitor can run? (4) What files does a baselining job produce? (5) Model-quality monitoring needs one thing the other monitors don't. What is it? (6) Clarify and Model Monitor are closed to new customers: what does AWS document as the replacement for each?
- **Concepts (exam vocabulary):** **Clarify** pre-training bias (class imbalance, difference in proportions of labels, KL/JS divergence), post-training bias, SHAP feature attribution (baseline choice), the explainability report; **Model Monitor**: data capture → baseline (statistics + constraints) → schedule → violations → CloudWatch metrics and alarms → retrain trigger; data quality, model quality (needs ground-truth labels merged in), bias drift, feature-attribution drift. Pre-built monitors vs `ClarifyCheck`/`QualityCheck` steps in Pipelines.
- **Concepts (what new accounts build):** both services are "no longer open to new customers". AWS's documented replacement: bias metrics computed with pandas/scikit-learn from the Clarify formula pages, SHAP directly (the engine Clarify uses, → M9.2), and Evidently drift reports (→ M6.8) run as Processing steps or EventBridge-scheduled Lambdas over endpoint data capture, logged to managed MLflow, with CloudWatch alarms and SNS alerts. Its reference code is the open-source [SageMaker AI monitoring solutions][aiops-mon].
- **Resource:** [Clarify overview][clarify], [pre-training bias metrics][clarify-pre], [Shapley values][clarify-shap]; [Model Monitor][mm], [model quality][mm-mq], [bias drift][mm-bias], [feature attribution drift][mm-fa], [scheduling][mm-sched]; [Clarify availability change][clarify-change] and [Model Monitor availability change][mm-change] (replacement sections) (≈2.5 h). If your account is an existing Clarify customer, also run [`mlops/v3-sagemaker-clarify.ipynb`][smx].
- **Exercise:** compute DPL and DI for one protected attribute with pandas (match the formulas on the pre-training metrics page), plus SHAP global importance, in a SageMaker Processing job that logs to managed MLflow. Then deploy the model with data capture and run the `predictiveml-endpoint-monitoring` solution from [the monitoring solutions repo][aiops-mon] (Evidently `DataDriftPreset`) against your baseline, and replay shifted traffic. If your account still has Model Monitor, you may use a `DefaultModelMonitor` hourly schedule instead. **Done when** the bias metrics and SHAP plot are in MLflow, a drift report names the shifted feature, and a CloudWatch alarm on the published drift metric goes to `ALARM`. Delete the endpoint, schedules and Lambdas afterwards.
- **Self-check:** feature-attribution drift can fire while data-quality drift stays green. Why (relationships changed while marginal distributions stayed the same, → M6.8)?

### AWS-MLA.7 Inference and deployment on SageMaker AI
- [ ] **8 h** · prereqs: AWS-MLA.5; endpoint basics in AWS-AIP.9 (real-time/serverless/async) · tested by: 2.3.3, 3.1.1–3.1.3, 3.2.5, 3.2.6, 3.3.1, 4.1.4, 4.2.1, 4.2.4
- **Skip test:** (1) Score 50 GB nightly with no endpoint left running: which option? (2) There are 1,000 per-customer models with sparse traffic. Which option? (3) Serverless inference rules out what (name 3)? (4) Shadow test vs production-variant A/B: what is the difference? (5) Which deployment-guardrail mode shifts traffic in equal steps, and what triggers auto-rollback? (6) Which predefined metric does a target-tracking policy usually track, and which policy type must you use to scale out from zero instances?
- **Concepts:** the decision table. Real-time, serverless and async endpoints are taught in AWS-AIP.9 (the extra limits MLA asks about are in the traps list below). This module adds **batch transform** as the fourth option; **multi-model endpoints**, **multi-container endpoints** (direct or serial invocation), **inference pipelines**; production variants with weights or `TargetVariant`; **shadow tests**; **deployment guardrails** (real-time and async endpoints only): blue/green with all-at-once, canary or linear traffic shifting, or rolling deployments in batches, all with a baking period and auto-rollback on CloudWatch alarms; autoscaling (target tracking, which AWS recommends in most cases, usually on the predefined `SageMakerVariantInvocationsPerInstance` metric; step scaling, which is required to scale out from zero instances; scheduled actions; cooldowns default to 300 s); **Inference Recommender**; instance families (CPU c/m/r, GPU g/p, Inferentia `inf`, Trainium `trn`).
- **Resource:** [endpoints][sm-ep]; [batch transform][bt]; [multi-model][mme]; [multi-container][mce]; [inference pipelines][ip]; [production variants][ab]; [shadow tests][shadow]; [deployment guardrails][dg]; [autoscaling][as]; [Inference Recommender][ir]; [inference cost optimization][ico] (≈3 h). Examples: `inference/sm-a_b_testing`, `inference/sm-shadow_variant_shadow_api`, `inference/sm-deployment_guardrails_update_inference_endpoint_with_linear_traffic_shifting` in [amazon-sagemaker-examples][smx].
- **Exercise:** deploy model v1, add v2 as a shadow variant, compare the two, promote v2 with linear traffic shifting and a latency alarm, then force a rollback by injecting a sleep. **Done when** the endpoint event history shows the rollback and a batch transform job scored the holdout set with no endpoint running.
- **Self-check:** for "intermittent traffic, cold starts acceptable, cheapest", answer serverless. For "large video frames, minutes per request", answer async.

### AWS-MLA.8 Foundation-model customization and hosting on AWS
- [ ] **5 h** · prereqs: AWS-MLA.7, C11.1–C11.4, M5.7; AWS-AIP.1 (Bedrock runtime); AWS-AIP.9 (JumpStart, LMI containers, inference-component and Bedrock-customization basics) · tested by: 1.2.9, 2.1.2, 2.2.8, 3.1.4, 3.1.5, 3.1.7, 3.2.10, 3.3.10
- **Skip test:** (1) Bedrock now lists three customization methods. Which one is not fine-tuning or distillation, and what does it need from you? (2) How is a customized Bedrock model billed for training? (3) You have weights fine-tuned outside AWS and want serverless Bedrock-style invocation. Which feature, and what must the weights be? (4) How do you give each model on a shared GPU endpoint its own accelerator and memory allocation, and let it scale to zero? (5) Which metric would you scale a GPU LLM endpoint on, rather than invocations?
- **Concepts:** Bedrock customization (fine-tuning and distillation as options → AWS-AIP.9): the full method list is **supervised fine-tuning, reinforcement fine-tuning (Lambda reward functions), distillation** (teacher → student). Training is billed per token × epoch, plus monthly storage. Skill 1.2.9 still says "continuous pre-training", but the Bedrock page lists only these three methods.
  - JSONL data prep; **Custom Model Import** (owned here; AIP-C01 does not name it): bring open-weight models of supported architectures (Hugging Face safetensors weights in S3), fine-tuned outside AWS, into Bedrock for on-demand invocation (skill 3.1.5); customized-model versioning in pipelines.
  - JumpStart fine-tuning (JumpStart deployment → AWS-AIP.9); LMI container configuration (container and model-loading basics → AWS-AIP.9).
  - **Inference components** (hosting several models and adapters on one endpoint → AWS-AIP.9): per-model resource allocation, adapter components, autoscaling copies, scale to zero.
- **Resource:** Bedrock [customize your model][bcust] + [Custom Model Import][bcmi]; SageMaker [model parallelism and large model inference][lmi]; [real-time endpoints][sm-rt] → "adapter inference components" (≈2 h). Example: `inference/sm-app_autoscaling_realtime_endpoints_inference_components` in [amazon-sagemaker-examples][smx].
- **Exercise:** write `prepare_ft.py`, which turns your C11.4 QLoRA dataset into Bedrock SFT JSONL, validates it (schema, token-length cap, dedup, PII scan using C9.4) and prints the token count × epochs cost estimate. Then deploy a small open model with an inference component and a target-tracking policy. **Done when** the validator rejects 3 planted bad records and the inference-component copy count changes under load. Do **not** launch the Bedrock training job unless you budget for it.
- **Self-check:** "lowest latency, steady high volume, custom model on Bedrock" → which capacity option (→ AWS-AIP.1 Provisioned Throughput)?

### AWS-MLA.9 Orchestration and CI/CD for ML
- [ ] **8 h** · prereqs: AWS-MLA.5–.7, M6.4, M6.9; AWS-AIP.2 (Step Functions, EventBridge, CodePipeline/CodeBuild basics); AWS-AIP.3 (KB ingestion) · tested by: 3.2.2, 3.2.3, 3.3.1–3.3.6, 3.3.8, 3.3.10, 3.3.11, 4.3.1
- **Skip test:** (1) SageMaker Pipelines vs Step Functions vs MWAA: pick one for an existing Airflow shop, one for ML lineage and caching, and one for a cross-service business workflow. (2) Which Pipelines step stops a run when a metric is below the threshold? (3) What events start retraining on a schedule and on drift? (4) What do CodeConnections and CodeArtifact do in an ML pipeline? (5) Where do you scan training and inference images for CVEs?
- **Concepts:**
  - **SageMaker Pipelines**: parameters; Processing, Training, Tuning, Model, RegisterModel, Transform, Condition, Fail, Lambda, Callback, ClarifyCheck, QualityCheck and EMR steps (ClarifyCheck and QualityCheck depend on the closed Clarify/Model Monitor services; new accounts use a Processing step, → AWS-MLA.6); caching; lineage.
  - **Projects** templates.
  - EventBridge retrain triggers (schedule, approval, monitor alarm); Step Functions (→ AWS-AIP.2) and MWAA as alternatives.
  - CodePipeline stages and CodeBuild gates → AWS-AIP.2. This module adds **CodeDeploy, CodeCommit, CodeConnections, CodeArtifact**, troubleshooting a failed pipeline stage (skill 3.3.2), and wiring CodeBuild to start a SageMaker Pipeline; CloudFormation and CDK cross-stack outputs.
  - Images: ECR builds with **Inspector** scanning.
  - Test stages: M6.9 tests, C3.3 prompt tests.
  - KB refresh as a scheduled pipeline step (ingestion jobs and their triggers → AWS-AIP.3).
- **Resource:** [Pipelines][pipe] (overview, steps, caching, parameters); [Projects][proj]; [Inspector ECR scanning][insp] (≈2.5 h). Run [`mlops/v3-pipeline-train-create-registry.ipynb` and `mlops/sm-mlflow_pipelines`][smx].
- **Exercise:** a CDK (Python) app with two stacks, (a) CodePipeline + CodeBuild, which runs pytest and builds/pushes an image scanned by Inspector, and (b) a SageMaker Pipeline (process → train → evaluate → Condition → RegisterModel), plus an EventBridge rule that retrains weekly. **Done when** a Git push runs the pipeline end to end, a failing metric stops at `Fail`, and `cdk destroy` leaves nothing billable.
- **Self-check:** which orchestrator gives ML DAGs with lineage at the least operational overhead?
- **Trap:** CodeGuru Reviewer has rejected new repository associations since 2025-11-07 ([its docs][cgr]), and the CodeGuru Security docs no longer resolve (as of 2026-10-31). Skill 4.3.1 still names CodeGuru, so know it by name. Use Inspector in practice.

### AWS-MLA.10 Operate, cost and secure ML workloads
- [ ] **5 h** · prereqs: AWS-MLA.7, AWS-MLA.9; CloudWatch/CloudTrail basics in AWS-AIP.1; dashboards, anomaly alarms, Cost Explorer and cost-allocation tags in AWS-AIP.10 · tested by: 4.1.2, 4.2.1, 4.2.3–4.2.6, 4.2.10, 4.3.1–4.3.7
- **Skip test:** (1) Which purchase option fits steady SageMaker usage for 1–3 years? (2) Which service checks continuously that every endpoint config uses a KMS key? (3) Name 3 endpoint CloudWatch metrics you would put on a dashboard. (4) A user can't create a training job in a VPC subnet. Name 2 places to look.
- **Concepts:**
  - Metrics: `Invocations`, `ModelLatency`, `OverheadLatency`, 4xx/5xx, utilization.
  - Dashboards: SageMaker endpoint metrics on a CloudWatch dashboard (building dashboards → AWS-AIP.10) or Amazon Quick; DevOps Guru.
  - **Cost** (Cost Explorer, Cost Anomaly Detection and cost-allocation tags → AWS-AIP.10): Budgets, ML Savings Plans, spot, scale-to-zero, Compute Optimizer, Trusted Advisor.
  - **Security**: SageMaker IAM condition keys, Config rules, CloudTrail, SCPs.
  - Debugging `AccessDenied`: role trust, `PassRole`, KMS key policy, bucket policy, VPC endpoint policy.
- **Resource:** [ML Savings Plans][mlsp]; the [Service Authorization Reference page for SageMaker][sar] (condition keys such as `sagemaker:VpcSubnets`, `sagemaker:VolumeKmsKey`, `sagemaker:NetworkIsolation`) (≈1.5 h).
- **Exercise:** write `guardrails.py`, which (a) creates a Budget with an alert, (b) deploys an SCP-style IAM policy that denies `CreateTrainingJob` without `VpcConfig` and a KMS key, and (c) builds a CloudWatch dashboard for your endpoint via boto3. **Done when** a non-compliant job request returns `AccessDenied`, the compliant one runs, and the dashboard shows latency p50/p99.
- **Self-check:** in a KMS-encrypted S3 → training-job failure, which three policies can deny access?

### AWS-MLA.11 AWS AI services picker
- [ ] **2 h** · prereqs: C12.3, M8.1, AWS-AIP.4 · tested by: 2.1.4, 2.1.8
- **Skip test:** match each need to a service: image moderation, product recommendations, text-to-speech, chatbot intents, translation, medical entities from clinical notes, a FHIR data store, video ingestion for analysis.
- **Concepts:** Rekognition, Personalize (managed recsys, → M8.1), Polly, Translate, Lex, Comprehend Medical, HealthLake, Kinesis Video Streams. Comprehend, Textract and Transcribe → AWS-AIP.4. If you made the AWS-AIP.13 or AIF-C01 flashcards, you already have Rekognition and Lex (and, from AIF, Polly, Translate and Personalize); only re-check them here. When a managed AI service beats a custom SageMaker model: no labelled data, standard task, fastest time to value (→ C12.3 for the LLM-vs-specialised trade-off).
- **Resource:** each service's "What is" page, overview only (≈1 h).
- **Exercise:** write `picker.py`, a flash-card quiz of 12 one-line scenarios that shuffles them and checks your answers. **Done when** two consecutive runs score 12/12, with each answer checked against the service's "What is" page.
- **Self-check:** "recommendations with no ML team and real-time personalisation": Personalize, not a SageMaker two-tower model.

### AWS-MLA.12 Capstone: re-platform the M6.10 capstone on AWS
- [ ] **12 h** · prereqs: AWS-MLA.1–.10, M6.10 · tested by: D1–D4 end to end
- **Skip test:** you already run a production ML workload on AWS with IaC, registry approval, alarm-guarded deployments and drift alerts. Then do only done-when item (3), the README mapping (≈1 h).
- **Brief:** move your M6.10 MLOps capstone from its OSS stack to AWS. The target flow:
  1. S3 with Glue Data Quality checks.
  2. Feature Store.
  3. A SageMaker Pipeline: spot training, small AMT, evaluate, a bias + SHAP Processing step (Clarify if your account still has it), Condition, RegisterModel.
  4. MLflow tracking.
  5. An approval event.
  6. A CodePipeline canary deploy with alarm rollback.
  7. Drift monitoring over data capture, using the AWS-MLA.6 replacement (or Model Monitor if your account still has it).
  8. EventBridge retraining.

  Build the IaC in CDK (Python) and the CI in CodeBuild. Optional, only if you did AWS-AIP: add a Guardrails-checked Bedrock explanation endpoint.
- **Done when:** (1) one Git push produces a registered, approved and canary-deployed model with no console clicks; (2) a drift replay raises an alarm that starts a retraining run; (3) `README.md` maps each OSS component to its AWS replacement and shows measured cost; (4) `cdk destroy` plus the teardown script leave USD 0/hour of resources.
- **Self-check:** for each step, name the exam skill it proves (use the coverage matrix).

### AWS-MLA.13 Practice and readiness
- [ ] **10 h** · prereqs: all · tested by: all
- **Skip test:** none. Everyone does this module.
- See [Practice and readiness](#practice-and-readiness). The 10 h covers the official practice material, two timed runs, error review and the final-week review.
- **Done when** every item of the readiness signal below is met.

---

## Coverage matrix

Every MLA-C02 task statement and skill (per the [guide][guide]) maps to module IDs. "AIP.n" = AWS-AIP.n in [aip-c01.md](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/aws/aip-c01.md).

| Task | Skills → modules |
|---|---|
| **1.1 Collect and store data** | 1.1.1–1.1.3 → MLA.2 · 1.1.4 → MLA.2 · 1.1.5 → MLA.2, M3.3 · 1.1.6 → MLA.2, M3.6, AIP.8 · 1.1.7 → C7.3, AIP.3 · 1.1.8 → MLA.2, C12.1 · 1.1.9 → MLA.3 |
| **1.2 Transform, feature engineering, pre-process** | 1.2.1 → MLA.3, MLA.2, AIP.4 · 1.2.2 → MLA.3 · 1.2.3 → MLA.2 · 1.2.4 → M3.2, MLA.3 · 1.2.5 → C7.3, AIP.3 · 1.2.6 → M5.2, M3.1 · 1.2.7 → C7.2, AIP.3 · 1.2.8 → C9.4, AIP.4, AIP.7, MLA.3 · 1.2.9 → C11.3, M5.7, AIP.9, MLA.8 |
| **1.3 Data quality and bias** | 1.3.1 → MLA.3, M3.4, AIP.4 · 1.3.2 → M3.1, AIP.9 · 1.3.3 → M3.1, MLA.6 · 1.3.4 → M9.1, MLA.6 · 1.3.5 → M3.1, MLA.6 · 1.3.6 → C11.3, C9.4, AIP.7 · 1.3.7 → M3.2, MLA.3 |
| **2.1 Modelling approach** | 2.1.1 → C3.4, AIP.1 · 2.1.2 → C11.1, AIP.9, MLA.8 · 2.1.3 → M2.1–M2.4, MLA.4 · 2.1.4 → C13.1, MLA.11 · 2.1.5 → C7.1 · 2.1.6 → M10.1, MLA.4 · 2.1.7 → C10.5, AIP.1 · 2.1.8 → AIP.4, MLA.11 |
| **2.2 Train, fine-tune, customize** | 2.2.1, 2.2.2 → MLA.4 · 2.2.3 → MLA.5, M4.4 · 2.2.4 → MLA.4, M7.3 · 2.2.5 → M2.3, C11.1 · 2.2.6 → M2.4, MLA.4 · 2.2.7 → M4.3 · 2.2.8 → C2.1, C11.2, MLA.8 · 2.2.9 → C7.3, C7.4, M5.8 |
| **2.3 Analyze and evaluate** | 2.3.1 → MLA.5, AIP.4, AIP.11 · 2.3.2 → M6.3, MLA.6 · 2.3.3 → MLA.7 · 2.3.4 → M9.2, MLA.6 · 2.3.5 → MLA.4, M4.2 · 2.3.6 → M2.2, M6.3, C3.2 · 2.3.7 → C8.1, AIP.11, AIP.9 · 2.3.8 → C3.2, M5.9 · 2.3.9 → C8.1, AIP.11 · 2.3.10 → C7.5, AIP.11 |
| **3.1 Deployment infrastructure** | 3.1.1 → MLA.7, M6.5 · 3.1.2 → MLA.7, MLA.9 · 3.1.3 → MLA.7, AIP.9 · 3.1.4 → AIP.1, AIP.9, MLA.8 · 3.1.5 → MLA.8 · 3.1.6 → C5.2, C6.1, C6.2, AIP.5, AIP.6 · 3.1.7 → MLA.8 · 3.1.8 → C7.4, AIP.3 |
| **3.2 Provision and configure resources** | 3.2.1 → C10.4, AIP.1 · 3.2.2 → MLA.9 · 3.2.3 → MLA.9, M6.5 · 3.2.4 → MLA.1 · 3.2.5 → MLA.1, MLA.7 · 3.2.6 → MLA.7 · 3.2.7, 3.2.8 → C7.1–C7.4, AIP.3 · 3.2.9 → C5.4, AIP.6 · 3.2.10 → MLA.8, C10.4 · 3.2.11 → AIP.6 |
| **3.3 Orchestration and CI/CD** | 3.3.1 → MLA.7, MLA.9 · 3.3.2 → AIP.2, MLA.9 · 3.3.3 → MLA.4, MLA.9 · 3.3.4 → MLA.9, M6.9 · 3.3.5 → MLA.9, M6.7 · 3.3.6 → AIP.9, MLA.5 · 3.3.7 → C2.2, AIP.4 · 3.3.8 → AIP.5, AIP.6, MLA.9 · 3.3.9 → C3.3, MLA.9 · 3.3.10 → MLA.8, MLA.9 · 3.3.11 → AIP.3, MLA.9 |
| **4.1 Monitor inference and performance** | 4.1.1 → MLA.6, AIP.10, AIP.11 · 4.1.2 → MLA.6, MLA.10 · 4.1.3 → MLA.6, M6.8 · 4.1.4 → MLA.7, M6.7 · 4.1.5 → C8.2, C8.3, AIP.6, AIP.10 · 4.1.6 → AIP.10, AIP.11 |
| **4.2 Cost and performance** | 4.2.1 → MLA.7 · 4.2.2 → AIP.1, AIP.6, AIP.10, MLA.10 · 4.2.3 → AIP.10, MLA.10 · 4.2.4 → MLA.7, MLA.10 · 4.2.5 → AIP.10, MLA.10 · 4.2.6 → MLA.10 · 4.2.7–4.2.10 → C10.5, AIP.1, AIP.10, MLA.10 |
| **4.3 Secure workloads and endpoints** | 4.3.1 → MLA.9 · 4.3.2, 4.3.3 → MLA.1, MLA.10, AIP.1 · 4.3.4 → MLA.10 · 4.3.5 → MLA.10 · 4.3.6 → MLA.1 · 4.3.7 → C9.1, C9.2, MLA.10 · 4.3.8 → AIP.1 · 4.3.9 → C9.4, AIP.7 |

(MLA.n = AWS-MLA.n.)

---

## Schedule

**90 h for this file.** If you took the AWS AI path, you already have the AWS-AIP modules this file cites. Otherwise do just these modules from [aip-c01.md](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/aws/aip-c01.md) first: AWS-AIP.1 (7 h), .2 (5 h), .3 (6 h), .4 (4 h), .5 (6 h), .6 (6 h), .7 (4 h), .8 (4 h), .9 (4 h), .10 (5 h), .11 (4 h) = **55 h**, about 8 extra weeks at 7 h/week before week 1 below (follow that file's schedule for those modules and skip its .0, .12 and .13), for **145 h** in total. The optional add-on is extra (≈4 h).

| Module | .0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 | .8 | .9 | .10 | .11 | .12 | .13 | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hours | 3 | 5 | 6 | 6 | 8 | 6 | 6 | 8 | 5 | 8 | 5 | 2 | 12 | 10 | **90** |

**Default: ~7 h/week for 13 weeks.** If you start 2026-11-02, you finish around 2027-01-31, after GA (2027-01-14).

| Week | Work | h |
|---|---|---|
| 1 | .0 (3), .1 (4) | 7 |
| 2 | .1 (1), .2 (6) | 7 |
| 3 | .3 (6), .11 (1) | 7 |
| 4 | .4 (7) | 7 |
| 5 | .4 (1), .5 (6) | 7 |
| 6 | .6 (6), .11 (1) | 7 |
| 7 | .7 (7) | 7 |
| 8 | .7 (1), .8 (5), .9 (1) | 7 |
| 9 | .9 (7) | 7 |
| 10 | .10 (5), .12 (2) | 7 |
| 11 | .12 (7) | 7 |
| 12 | .12 (3), .13 (4) | 7 |
| 13 | .13 (6), then book | 6 |

**Compressed: ~15 h/week for 6 weeks (90 h).** This is the only route to the beta, and only if the beta is still open (no end date has been published).

| Week | Work |
|---|---|
| W1 | .0, .1, .2, .3 (1 h) |
| W2 | .3 (5), .4, .11 |
| W3 | .5, .6, .7 (3) |
| W4 | .7 (5), .8, .9 (5) |
| W5 | .9 (3), .10, .12 (7) |
| W6 | .12 (5), .13 |

---

## Practice and readiness

**Practice strategy (official free material first):**
1. [ ] The free **Official Practice Question Set** and the free Exam Prep course on the [MLA-C02 prep plan][prep]. [unconfirmed whether a C02-specific set is published yet. If only the C01 set exists, ignore questions on removed topics.]
2. [ ] Every module's skip test, answered cold. Each miss sends you back to that module's resource.
3. [ ] Optional, paid: in the final 2 weeks, one month of Skill Builder (USD 29/month as of 2026-10-31, [pricing][sb]) for the Official Practice Exam, but only if a C02 version is listed. Do one timed run and one review run.
4. [ ] Third-party question banks: none for C02 were confirmed as of 2026-10-31. The C01 banks (for example Tutorials Dojo) still drill the SageMaker half (62 of C02's 107 skills carried over from C01, counted from the [comparison][cmp]). Expect C01 banks to over-weight Debugger, Clarify, Model Monitor, Neo and BYOC. One report found them "significantly easier than the real exam" ([Huszcza][huszcza]), so don't use them as a readiness signal. **Never use dumps.** They break the AWS Certification Agreement.

**Readiness signal to book:**
- Every row of the coverage matrix is ticked.
- The capstone meets its "done when".
- You score ≥ 80% on the official practice material.
- You pass every skip test cold.
- You average ≤ 2 min per question (the beta allows 170 min for 85 questions).

**Final-week review:**
- Re-read the comparison page's additions.
- Redo the self-checks for .2, .7, .9 and .11.
- Go through the traps list below.
- Skim the [out-of-scope list][outscope]. Cognito, WAF, Route 53, AppConfig and Amplify are distractors.
- Do one pass over the GenAI rows of [Already covered](#already-covered-core--aws-ai-path).

**Exam-day tactics:**
- The constraint in the last sentence ("least operational overhead", "most cost-effective", "near real-time") decides between two plausible services. Read it first.
- Multiple-response questions need every correct option selected.
- Flag a question after 2 min and come back to it.
- Answer everything, because there is no guessing penalty.
- If English is not your native language, request ESL +30 min before booking ([policy][before]).

**Traps:**
- MLA-C01 was "tricky: many answers seem correct but have one little word that totally changes their meaning" ([Huszcza][huszcza]). Hands-on practice is critical ([Modi][modi]).
- Beyond the no-GPU limit (AWS-AIP.9), serverless inference has no VPC config, no data capture and no multi-model support ([docs][sl]).
- Async inference handles payloads up to 1 GB and processing up to 1 h ([docs][async]).
- Spot training without checkpoints loses progress, and spot never falls back to on-demand ([docs][spot]).
- Debugger, Clarify and Model Monitor are closed to new customers but may still appear in answer options. Know what each did and what replaces it (AWS-MLA.4, .6).
- Hyperband only suits iterative algorithms ([docs][amt-how]).
- Model-quality monitoring needs ground-truth labels.
- GenAI traps: cross-Region inference and Provisioned Throughput → AWS-AIP.1; KB sync → AWS-AIP.3; PII ANONYMIZE vs BLOCK → AWS-AIP.7.
- No first-hand MLA-C02 beta reports existed as of 2026-10-31.

---

## Add-on: AWS MLOps Demonstrated

**Verdict: take it, because it is free.** It is a hands-on assessment in a provisioned AWS environment, with no multiple-choice questions ([AWS blog][mc-free]). It earns a digital badge [Credly issuer unconfirmed]. The MLA cert page lists it as a related credential. Microcredentials have been free since 2026-04-23 ([AWS blog][mc-free], as of 2026-10-31). Catalogue: [Skill Builder microcredentials][mc].

**Cost:** USD 0, and you need no Skill Builder subscription ([AWS blog][mc-free]).

**Format:** about 2 h 20 min. It reportedly assesses "deploying a foundation model using Amazon SageMaker JumpStart and SageMaker AI". A failed microcredential reportedly needs a 25-day wait before a retry. [unconfirmed: both details come from an AWS community article, not an official policy page]. Check the Skill Builder page for the current task list before you start.

**Extra study beyond the core:** none beyond this file. You need JumpStart basics (→ AWS-AIP.9), AWS-MLA.7 (endpoint configuration, autoscaling) and AWS-MLA.8 (inference components, GPU instance choice). Allow ≈4 h: 1.5 h to re-run the .7 and .8 exercises in Studio, and ≈2.5 h for the assessment.

**Done when** the badge is issued, or after a fail you have noted which module (.7 or .8) to redo.

**When:** after AWS-MLA.8 and before AWS-MLA.12. A pass shows that your D3 skills work outside notebooks. A fail tells you which module to revisit before the capstone.

---

## Combining with other paths

- **AWS AI path first ([aip-c01.md](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/paths/aws/aip-c01.md)):** this is the recommended order. It already covers the 55 h of AWS-AIP.1–AWS-AIP.11 this file depends on, and it gives you a 50% voucher for MLA, and a pass on AIP-C01 also renews MLA.
- **Other ML paths** ([Azure AI-300](../azure/ai-300.md), [Google PMLE](../google/professional-ml-engineer.md), [Databricks ML Associate](../databricks/ml-associate.md), [NVIDIA NCP-GENL](../nvidia/ncp-genl.md)): there is no shared content beyond the ML core. The only repeat is each file's ≤1 h core refresh.
- **Capstone reuse:** re-platform the same M6.10 capstone for each vendor. Use the README mapping table from AWS-MLA.12 as the template.

---

## Gaps for the core

- None open. MLA 1.3.4 (bias metrics for **text and image** assets) was added to M9.1 on 2026-10-31; this file teaches only the Clarify side.

---

## Skipped on purpose

| Item | Reason |
|---|---|
| MLA-C01 material; SageMaker Neo, edge, compression, BYOC | The English C01 exam ended 2026-09-28 and C02 removed these topics. Compression is taught for its engineering value in M7.2 |
| MLS-C01 | Retired 2026-03-31 |
| Maarek & Kane MLA "Hands On!" (Udemy, paid) | It was built for C01, and a C02 update is unconfirmed (as of 2026-10-31). The free docs and samples cover the same ground |
| DeepLearning.AI × AWS "Generative AI with LLMs" | Not owned by any file. Its content is covered by C11 and M5.7 |
| SageMaker Experiments (Studio Classic), Canvas no-code modelling, HyperPod/Neuron depth | Legacy, or not named in the MLA-C02 guide's skills. Managed MLflow replaces Experiments |
| Cognito, WAF, Route 53, AppConfig, Amplify | Out of scope ([list][outscope]) |
| DEA-C01 | Overlaps D1 but goes much deeper. Not time-worthy here |
| Exam dumps | They break the AWS Certification Agreement |

---

## Sources

All accessed 2026-10-31. Link targets are defined at the bottom of this file.

- **Exam facts:** [MLA cert page][cert] · [MLA-C02 exam guide][guide] (+ domain pages) · [C01→C02 comparison][cmp] · [in-scope services][inscope] · [out-of-scope services][outscope] · [service mentions][mentions] · [September 2026 certification updates blog][blog] (GA 2027-01-14) · [recertification][recert] · [after-testing policy][after] · [before-testing policy][before] · [certification benefits][benefits]
- **Prep and labs:** [MLA-C02 Skill Builder prep plan][prep] · [Skill Builder pricing][sb] · [microcredentials catalogue][mc] · [microcredentials-free blog][mc-free] · [AWS Free Tier][free] · [aws/amazon-sagemaker-examples][smx] (top-level folders `training`, `model_customization`, `evaluation`, `inference`, `mlops`; default branch `default`)
- **SageMaker AI Developer Guide:** the pages linked inline in AWS-MLA.1–.10 (all checked to resolve on 2026-10-31), including the [Debugger][dbg-change], [Clarify][clarify-change] and [Model Monitor][mm-change] availability-change pages.
- **AWS replacement code for Clarify and Model Monitor:** [aws-samples/sample-aiops-on-amazon-sagemakerai, `monitoring/`][aiops-mon]
- **Bedrock:** [model customization][bcust] · [Custom Model Import][bcmi]
- **Data services:** [Kinesis Data Streams][kds] · [Data Firehose][fh] · [Managed Flink][flink] · [Glue][glue] · [Glue Data Quality][gdq] · [Glue FindMatches][findm] · [Glue Detect PII][glue-pii] · [DataBrew][databrew]
- **Ops, security, cost:** [Inspector ECR scanning][insp] · [CodeGuru Reviewer status notice][cgr] · [ML Savings Plans][mlsp] · [Service Authorization Reference: SageMaker][sar]
- **First-hand reports (MLA-C01; none found for C02):** [Huszcza][huszcza] · [Modi][modi]
- **MLOps Demonstrated details [unconfirmed, community article]:** [builder.aws article][mcart]

[cert]: https://aws.amazon.com/certification/certified-machine-learning-engineer-associate/
[guide]: https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/machine-learning-engineer-associate-02.html
[cmp]: https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/mla-02-comparison.html
[inscope]: https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/mla-02-in-scope-services.html
[outscope]: https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/mla-02-out-of-scope-services.html
[mentions]: https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/mla-02-service-mentions.html
[blog]: https://aws.amazon.com/blogs/training-and-certification/september-2026-new-offerings/
[recert]: https://aws.amazon.com/certification/recertification/
[after]: https://aws.amazon.com/certification/policies/after-testing/
[before]: https://aws.amazon.com/certification/policies/before-testing/
[benefits]: https://aws.amazon.com/certification/benefits/
[prep]: https://skillbuilder.aws/category/exam-prep/machine-learning-engineer-associate-MLA-C02
[sb]: https://aws.amazon.com/training/digital/
[mc]: https://skillbuilder.aws/certification/microcredentials
[mc-free]: https://aws.amazon.com/blogs/training-and-certification/microcredentials-from-aws-are-now-free-heres-why-that-matters/
[free]: https://aws.amazon.com/free/
[smx]: https://github.com/aws/amazon-sagemaker-examples
[sm-train]: https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html
[sm-data]: https://docs.aws.amazon.com/sagemaker/latest/dg/model-access-training-data.html
[sm-vpc]: https://docs.aws.amazon.com/sagemaker/latest/dg/infrastructure-give-access.html
[sm-iso]: https://docs.aws.amazon.com/sagemaker/latest/dg/mkt-algo-model-internet-free.html
[algos]: https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html
[spot]: https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
[dist]: https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html
[ddp]: https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html
[dbg-change]: https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-availability-change.html
[train-metrics]: https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html
[clarify-change]: https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-availability-change.html
[mm-change]: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-availability-change.html
[aiops-mon]: https://github.com/aws-samples/sample-aiops-on-amazon-sagemakerai/tree/main/monitoring
[glue-pii]: https://docs.aws.amazon.com/glue/latest/dg/detect-PII.html
[dbg-rules]: https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html
[warm]: https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html
[amt]: https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html
[amt-how]: https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html
[mlflow]: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html
[reg]: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html
[fs]: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html
[fs-off]: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-offline.html
[dw]: https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler.html
[clarify]: https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-configure-processing-jobs.html
[clarify-pre]: https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-data-bias.html
[clarify-shap]: https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-shapley-values.html
[mm]: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html
[mm-mq]: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html
[mm-bias]: https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-bias-drift.html
[mm-fa]: https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-feature-attribution-drift.html
[mm-sched]: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-scheduling.html
[sm-ep]: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-manage.html
[sm-rt]: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html
[sl]: https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html
[async]: https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html
[bt]: https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html
[mme]: https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html
[mce]: https://docs.aws.amazon.com/sagemaker/latest/dg/multi-container-endpoints.html
[ip]: https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipelines.html
[ab]: https://docs.aws.amazon.com/sagemaker/latest/dg/model-ab-testing.html
[shadow]: https://docs.aws.amazon.com/sagemaker/latest/dg/shadow-tests.html
[dg]: https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails.html
[as]: https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html
[ir]: https://docs.aws.amazon.com/sagemaker/latest/dg/inference-recommender.html
[ico]: https://docs.aws.amazon.com/sagemaker/latest/dg/inference-cost-optimization.html
[lmi]: https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference.html
[pipe]: https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html
[proj]: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects.html
[bcust]: https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html
[bcmi]: https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html
[kds]: https://docs.aws.amazon.com/streams/latest/dev/introduction.html
[fh]: https://docs.aws.amazon.com/firehose/latest/dev/what-is-this-service.html
[flink]: https://docs.aws.amazon.com/managed-flink/latest/java/what-is.html
[glue]: https://docs.aws.amazon.com/glue/latest/dg/what-is-glue.html
[gdq]: https://docs.aws.amazon.com/glue/latest/dg/glue-data-quality.html
[databrew]: https://docs.aws.amazon.com/databrew/latest/dg/what-is.html
[insp]: https://docs.aws.amazon.com/inspector/latest/user/scanning-ecr.html
[mlsp]: https://aws.amazon.com/savingsplans/ml-pricing/
[sar]: https://docs.aws.amazon.com/service-authorization/latest/reference/list_sagemaker.html
[huszcza]: https://blog.huszcza.dev/p/aws-mla-c01-en/
[modi]: https://aditmodi.hashnode.dev/how-i-passed-the-aws-certified-machine-learning-engineer-associate-exam-mla-c01
[findm]: https://docs.aws.amazon.com/glue/latest/dg/machine-learning.html
[cgr]: https://docs.aws.amazon.com/codeguru/latest/reviewer-ug/welcome.html
[mcart]: https://builder.aws.com/content/3JIRylI9CKtDeu4UJiEqJkI2mnD/aws-microcredentials-prove-your-practical-aws-skills
