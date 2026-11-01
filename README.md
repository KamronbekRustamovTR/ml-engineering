# ML Engineering

A self-study guide to the model-building half of AI work: data and features, classical and deep learning, transformer training and fine-tuning, MLOps (tracking, pipelines, deployment, monitoring, CI/CD/CT), performance and scale, and ML system design. It is written for a senior backend engineer (Java, Spring, Maven, Oracle) who already builds LLM applications and uses Claude Code daily. It uses free or low-cost resources, and every exercise is in Python. A vendor-neutral core teaches each concept once. Optional vendor paths then add only their own platform, services and exam framing.

## How to use this repo

1. **Finish the prerequisite first:** the [ai-engineering core](https://github.com/KamronbekRustamovTR/ai-engineering) (C0–C13). This repo does not teach anything from it again. An overlap is a one-line cross-reference to a `C` ID.
2. **Do the [ML core](core-ml-engineering.md)** (M0–M10). Start with its 1 h diagnostic. Every module opens with a **skip test**. If you pass it cleanly without docs, skip the reading, but still do the exercise and self-check, because they feed the capstones.
3. **Then take any vendor path or combination of paths** from the catalog below. Each path is a delta on the core. It also needs specific modules from the matching ai-engineering path (the catalog's "From the AI path" column). If you already took that AI path, you have them.
4. **Tick checkboxes** (`- [ ]`) as you go. For each module, tick "reading" (or "skipped via skip test") and "exercise + self-check".

**Module IDs**

| ID | Meaning |
|---|---|
| `C0.1`–`C13.4` | ai-engineering core (prerequisite) |
| `M0.1`–`M10.2` | ML core in this repo |
| `AWS-MLA`, `AZ-AI300`, `GCP-PMLE`, `DBX-MLA`, `DBX-MLP`, `NV-GENL`, `NV-ADS` | vendor path prefixes, numbered `<PREFIX>.<n>` |
| `<PREFIX>.0` | exam at a glance plus the core-refresh table (≤1 h): which blueprint items you already learned, and where |
| `<PREFIX>.1` | the path's platform foundations |
| `AWS-AIP.n`, `AZ-AI103.n`, `GCP-PAA.n`, `DBX-GAI.1`, `NV-AMAD.1` | shared platform modules that live in the ai-engineering paths (exact list per path in the catalog) |

## Path catalog

Status as of 2026-10-31. Costs are exam fees in USD before tax. "Hours" means hours on top of the ML core. "From the AI path" lists the ai-engineering modules a path needs if you skipped that vendor's AI path; their hours are extra.

| Vendor | Certification (code) | Level | Status | Cost | Eligibility | Hours | From the AI path (if skipped) | File |
|---|---|---|---|---|---|---|---|---|
| AWS | Machine Learning Engineer – Associate (MLA-C02) | Associate | **Beta** (delivery since 2026-09-29). GA delivery 2027-01-14 per AWS blog; cert page says "TBD" | Beta $75. GA price **[unconfirmed]** (MLA-C01 was $150) | No prerequisite listed. Recommended: 1 yr SageMaker AI/Bedrock + 1 yr related role | 90 | AWS-AIP.1–.11 (55 h) | [paths/aws/mla-c02.md](paths/aws/mla-c02.md) |
| Microsoft Azure | Machine Learning Operations Engineer Associate (AI-300) | Associate | Active. Replaced DP-100 (retired 2026-06-01); GA date **[unconfirmed]** | Priced by region; "typically" $165 | None | 51.5 | AZ-AI103.1, .2, .4, .6 (22 h) | [paths/azure/ai-300.md](paths/azure/ai-300.md) |
| Google Cloud | Professional Machine Learning Engineer (no code published; June-2026 guide) | Professional | Active | $200 | None. Recommended: 3+ yrs industry, 1+ yr on Google Cloud | 78 | GCP-PAA.1 + parts of .8–.10 (6.75 h) | [paths/google/professional-ml-engineer.md](paths/google/professional-ml-engineer.md) |
| Databricks | Certified Machine Learning Associate (no code) | Associate | Active (March 2025 outline) | $200 | None. Recommended: 6+ months hands-on | 39 | DBX-GAI.1 (6 h) | [paths/databricks/ml-associate.md](paths/databricks/ml-associate.md) |
| Databricks | Certified Machine Learning Professional (no code) | Professional | Active (Sep 2025 outline) | $200 | None. Recommended: 1+ yr hands-on | 46 (delta on DBX-MLA) | — (via DBX-MLA) | [paths/databricks/ml-professional.md](paths/databricks/ml-professional.md) |
| NVIDIA | Generative AI LLMs (NCP-GENL) | Professional | Live (Sep 2025 study guide) | $200 | None. Recommended: 2–3 yrs LLM work | 60 | NV-AMAD.1 parts 1A–1E (7.5 h) | [paths/nvidia/ncp-genl.md](paths/nvidia/ncp-genl.md) |
| NVIDIA | Accelerated Data Science (NCA-ADS / NCP-ADS) | Associate / Professional | Active (guides Feb 2026 / Mar 2025) | $125 / $200 | None. Recommended: 1–2 / 2–3 yrs | 42 / +18 | none | [paths/nvidia/nca-ads-ncp-ads.md](paths/nvidia/nca-ads-ncp-ads.md) |

The NCA-AIIO (AI Infrastructure and Operations, $125) gets a short note in the NCP-GENL file. Its verdict is "skip, unless your job touches cluster operations".

## Combining paths

Every concept tested by two or more vendors lives in the core, and each vendor service is owned by exactly one file. A later file from the same vendor is a strict delta on the earlier one. **Combining paths repeats no material.** The only repeat is each file's ≤1 h `.0` refresh table.

| Order | Good for | Notes |
|---|---|---|
| AWS-AIP (AI repo) → AWS-MLA | Full AWS AI + ML | Recommended order. Passing AIP-C01 gives a 50% voucher for MLA and also renews MLA |
| AZ-AI103 (AI repo) → AZ-AI300 → AZ-AI500 (optional) | Azure-first | AI-300's GenAIOps half is a mapping table to the AI core and AZ-AI103 |
| GCP-PAA (AI repo) → GCP-PMLE | Google ML | Needs core M3.6 + M7.4 |
| DBX-GAI.1 → DBX-MLA → DBX-MLP | Databricks ML | Book DBX-MLP within about 2 months of DBX-MLA. Both total 85 h |
| GCP-PMLE or AWS-MLA + NV-GENL | Platform MLOps + model-optimisation depth | The clouds test managed services. NV-GENL tests the GPU layer underneath them |
| DBX-MLA/MLP → AZ-AI300 | MLflow-heavy stack | AzureML is MLflow-native |
| NV-GENL → NV-ADS | NVIDIA depth | Doing NV-GENL first saves 4 h in NV-ADS.10. NV-ADS is the lowest-priority path |

If you want only one ML cert, pick the cloud your employer runs. Every path's capstone re-platforms the same core Capstone A (M6.10), so a second cloud mostly changes the service mapping.

## Hours

| Item | Hours |
|---|---|
| ML core, essential (diagnostic + M0–M10 + one extra capstone) | **290** (about 42 weeks at 7 h/week) |
| Conditional M3.6 Spark + M7.4 Ray (before DBX-MLA, DBX-MLP, GCP-PMLE; M3.6 also before NV-ADS) | +10 |
| AWS-MLA | +90 |
| AZ-AI300 | +51.5 |
| GCP-PMLE | +78 |
| DBX-MLA / DBX-MLP | +39 / +46 |
| NV-GENL | +60 |
| NV-ADS (NCA / NCP delta) | +42 / +18 |
| Core + conditional + every path above | about 725, plus any AI-path modules you skipped (up to 55 + 22 + 6.75 + 6 + 7.5 h) |

**Exam-first option:** the core's ["Where this core leads"](core-ml-engineering.md#where-this-core-leads-vendor-ml-paths) table gives each path a minimum core subset of about 100–206 h. Do that subset first and backfill the rest before your second path.

## Tools you will install

The same stack is used across both repos. The AI core's C0.2 lab (uv, Docker, MLflow server) is reused.

- **uv + Python 3.12 + Jupyter:** environment.
- **NumPy, pandas, scikit-learn, XGBoost, LightGBM:** classical ML (M0, M2–M3).
- **PyTorch:** deep learning.
- **HF Transformers + TRL + Accelerate (+ PEFT, Datasets):** training and fine-tuning (M5).
- **Optuna:** hyperparameter search (M4.4).
- **MLflow (the C0.2 server):** tracking and registry.
- **Feast:** feature store (M3.5).
- **Great Expectations-style tests:** data validation (M3.4).
- **Prefect (as taught in MLOps Zoomcamp):** orchestration (M6.4).
- **FastAPI + Docker; KServe on kind/minikube:** serving (M6.5–M6.6).
- **ONNX + ONNX Runtime:** model optimisation (M7.2). It has Java bindings.
- **Evidently:** drift monitoring (M6.8).
- **PyTorch DDP/FSDP2, Ray, Spark (M3.6 only):** distributed compute.
- **pytest + GitHub Actions + Terraform:** CI/CD/CT (M6.9).
- **Colab/Kaggle free GPU:** rent a GPU only for M5.3 and the optional M7.3 (about $10–50).

Vendor paths add only their own SDK, which each file names in its header.

## Prerequisite

**[ai-engineering](https://github.com/KamronbekRustamovTR/ai-engineering)** covers LLM application engineering: prompting, evals, agents, MCP, RAG, LLMOps, security, inference and cost. Do its [core](https://github.com/KamronbekRustamovTR/ai-engineering/blob/main/core-llm-app-engineering.md) before this repo. Its vendor paths also hold the platform modules the paths here build on (listed per path in the catalog).

## Disclaimer

Certification details (status, price, format, domains, renewal) change often. Every exam fact in the path files is dated 2026-10-31 and linked to its source. Items marked [unconfirmed] could not be verified. Check the vendor's certification page before you book.
