# Instamart Customer Intelligence Capstone

This project builds a full synthetic data workflow for an Instamart-style retail business. It combines data cleaning, SQL analysis, customer segmentation, churn prediction, support ticket routing, and a policy-based refund assistant using deterministic mock LLM behavior.

## Project overview

The codebase is organized into four parts:

- Part A: Data audit, cleaning, validation, SQL queries, and exploratory data analysis
- Part B: RFM segmentation, churn/lapse modeling, and retention analysis
- Part C: Support ticket classification and prioritization for customer issues
- Part D: Policy retrieval and a tool-driven refund workflow agent

The project uses synthetic but realistic customer, product, order, and ticket data generated locally, making it a good end-to-end learning project for analytics and applied ML workflows.

## Repository structure

```text
Instamat_project/
├── data/
│   ├── raw/                  # Generated source CSV datasets
│   ├── clean/                # Cleaned data outputs
│   ├── policies/             # Policy documents for retrieval
│   └── instamart.db          # SQLite database used for SQL analysis
├── outputs/                  # Generated results, reports, and model outputs
├── notebooks/                # Optional notebook work (if present in your environment)
├── src/
│   ├── part_a.py             # Data extraction, cleaning, validation, SQL, EDA
│   ├── part_b.py             # RFM, clustering, retention modeling
│   ├── part_c.py             # Ticket triage pipeline
│   └── part_d.py             # Policy RAG and refund agent logic
├── generate_data.py           # Synthetic dataset generator
├── mock_llm.py               # Offline mock language model for classification and routing
├── run_project.py            # End-to-end project runner
├── validate.py               # Validation checks for cleaned outputs
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and setup guide
└── sql/
    └── queries.sql           # SQL analysis queries used in Part A
```

## Data and workflow

### Synthetic data generation

`generate_data.py` creates realistic retail data, including:

- Customers with city, age, gender, and signup information
- Product catalog with pricing and categories
- Orders with statuses, discount data, amounts, and delivery timing
- Support tickets representing billing, delivery, technical, and fraud cases
- Policy text documents used by the retrieval workflow

### Analytics pipeline

`run_project.py` executes the full project and prints results for:

1. Data quality audit and cleaning checks
2. SQL-based revenue and customer analysis
3. Customer segmentation and retention modeling
4. Support ticket triage routing
5. Policy retrieval and refund workflow evaluation

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python generate_data.py
```

This will create the raw data under `data/raw` and populate the policy files under `data/policies`.

## Run the project

### Full pipeline

```bash
python run_project.py
```

This runs the complete analysis stack from data cleaning through segmentation, triage, and policy-driven interactions.

### Validation

```bash
python validate.py
```

This checks that the cleaned data satisfies expected project assertions and prints a pass/fail confirmation.

## Key capabilities

- Data auditing and anomaly detection for messy real-world datasets
- SQL analysis for customer and revenue insights
- RFM feature engineering and customer segmentation
- Lapse prediction using scikit-learn pipelines
- Ticket routing with deterministic LLM-like behavior
- Retrieval-augmented policy answers for customer support scenarios
- Simulated refund decision workflow using local utility tools

## Notes

- This is a synthetic, educational project and not connected to a live production system.
- The mock LLM is intentionally deterministic and offline so the project can run without API keys.
- Outputs and generated data are stored locally under `data/` and `outputs/`.

## License

This project is intended for educational and demonstration use.
