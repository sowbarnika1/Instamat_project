# Instamart Customer Intelligence Capstone

## Overview

This repository contains an end-to-end, locally reproducible analytics project built with synthetic Instamart-style retail data. It covers data quality and SQL analysis (Part A), customer segmentation and lapse prediction (Part B), support-ticket triage (Part C), and policy retrieval with a simulated refund workflow (Part D).

The supplied root-level `generate_data.py` data generator and `mock_llm.py` deterministic language-model stand-in are retained as provided. The project runs offline and does not call a live LLM or payment service.

Key files:

- `src/part_a.py` through `src/part_d.py`: implementation of Parts A-D.
- `sql/queries.sql`: seven SQL analysis queries.
- `notebooks/`: notebooks for exploring and running each part.
- `outputs/`: committed segmentation, model, retention, and triage artifacts.
- `data/`: synthetic input data, cleaned data, policy text, and SQLite database.
- `run_project.py`: runs the full pipeline from the repository root.
- `requirements.txt`: Python package dependencies.

## Reproduce

Use Python with the dependencies in `requirements.txt`. From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install dependencies and run the complete pipeline:

```bash
python -m pip install -r requirements.txt
python run_project.py
```

The runner regenerates the synthetic inputs, cleans and validates the data, runs the SQL and analytics workflows, and writes the generated artifacts. Part A validation includes checks for customer and order totals, unique order IDs, status counts, completed-order amounts, and delivery-time completeness.

The committed outputs are:

- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`
- `outputs/triage_results.csv`

## Part A

Part A audits the synthetic source tables, normalizes customer fields, cleans order records, stores clean CSVs and SQLite tables, runs the seven queries in `sql/queries.sql`, and cross-checks sales using pandas.

Validated dataset totals are **1,200 customers**, **40 products**, and **12,309 unique orders**: **11,110 completed** and **1,199 cancelled**. The completed orders total **INR 2,998,668.80**. Customer data covers **6 cities**. The cleaning and validation assertions run as part of `python run_project.py`.

## Part B

Part B builds customer recency, frequency, and monetary (RFM) features, compares K-means segmentations, creates a lapse cohort, tunes logistic-regression and random-forest models, and saves the selected model and a retention shortlist.

RFM features cover **1,175 customers**. The selected segmentation uses **4 clusters**. The leakage-free lapse cohort has **1,156 customers**, of whom **443 (38.32%)** are labeled lapsed. The stratified model split contains **924 training** and **232 test** examples.

On the held-out test set, the random forest scores **0.737 accuracy**, **0.659 precision**, **0.652 recall**, and **0.655 F1** for the lapse class. Its best cross-validation F1 is **0.681**. In the class-balance experiment, oversampling changes logistic-regression test F1 from **0.648** to **0.697** and recall from **0.652** to **0.854**, while precision changes from **0.644** to **0.589**. The saved retention shortlist has **10 customers**; **1** is labeled lapsed in the observed period.

## Part C

Part C redacts detected email addresses, phone numbers, and order IDs before sending ticket text to the deterministic local mock, then validates the classification response and routes tickets by category and customer segment. It writes the resulting records to `outputs/triage_results.csv`.

The output contains **164 tickets** and uses **12,520 tokens**. It records **55 email**, **91 phone**, and **132 order-ID** redactions. There are **135 JSON classifications**, **12 keyword-fallback classifications**, **9 invalid outputs**, **4 empty tickets**, and **4 long tickets**. Routing results are **102** support-queue, **34** security-team, **17** human-review, and **11** premium-desk tickets.

## Part D

Part D indexes local policy text into overlapping TF-IDF chunks and evaluates answers against a policy question set. Its tool workflow can look up an order, retrieve policy, calculate a refund, and issue or escalate a simulated action. Any ledger is local JSONL; no money moves.

The index contains **15 chunks** from **5 policy files**. In the **14-question** saved evaluation, the expected document and expected fact were each found for **12** in-scope questions; the **2** out-of-scope questions received fallback responses. The example policy allows a refund for an eligible completed order delivered in more than **45 minutes**; this is a demonstration of the local policy and not a live payment operation.

## Limitations and Next Steps

- The dataset is synthetic; model and workflow results do not establish production performance.
- The mock LLM is deterministic and offline, so its outputs do not represent a live language model.
- The retention shortlist is illustrative: its premium designation is a post-hoc, whole-window label and should not be treated as a leakage-free evaluation.
- Ticket redaction uses regular expressions and is not a guarantee that all personal information is removed.
- Refund actions are simulated and use a local ledger; there is no payment integration, production authorization, or operational audit service.

Next steps include evaluating on representative, consented data; adding temporal and drift checks; testing redaction and routing with broader edge cases; setting model acceptance criteria with human review; and hardening the refund workflow before any production integration.
