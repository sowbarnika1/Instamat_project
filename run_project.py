
from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / "generate_data.py").exists():
    ROOT = ROOT.parent
import subprocess, sys
subprocess.run([sys.executable, str(ROOT / "generate_data.py")], check=True)
print(ROOT)
raw_dir = ROOT/ "data" / "raw"
print("Raw files:", sorted(p.name for p in raw_dir.iterdir()))


from src.part_a import (
    audit,
    clean_customers,
    clean_orders,
    load_raw,
    pandas_eda,
    run_sql,
    store,
    validate,
)
from src.part_b import (
    balance_experiment,
    build_lapse_cohort,
    build_rfm,
    cluster_segments,
    interpret_persist_retain,
    load_data,
    train_evaluate,
)
from src.part_c import FEW, run_pipeline
from src.part_d import PolicyRAG, evaluate

def main():
    print("\n=== PART A: AUDIT / CLEAN / VALIDATE ===")
    raw_customers, products, raw_orders, tickets = load_raw()
    audit(raw_customers, products, raw_orders, tickets)

    customers = clean_customers(raw_customers)
    orders = clean_orders(raw_orders, products)
    tables = store(customers, products, orders)
    validate(tables["customers"], tables["products"], tables["orders"])
    sql_results = run_sql()
    pandas_eda(
        tables["customers"],
        tables["products"],
        tables["orders"],
        sql_results,
    )

    print("\n=== PART B: SEGMENTS AND RETENTION ===")
    orders, customers = load_data()
    rfm = build_rfm(orders)
    segments, diagnostics, profile, pca = cluster_segments(rfm)
    print("K-means diagnostics:\n", diagnostics.to_string(index=False))
    print("Segment profile:\n", profile.to_string(index=False))
    print("PCA explained variance:", pca.explained_variance_ratio_)

    cohort, features = build_lapse_cohort(orders, customers)
    run = train_evaluate(cohort, features)
    balance_experiment(run)
    interpret_persist_retain(run, segments)

    print("\n=== PART C: TRIAGE ===")
    triage, summary = run_pipeline(prompt=FEW, token_budget=100000)
    print(summary)
    print(triage.head().to_string(index=False))

    print("\n=== PART D: POLICY RAG ===")
    rag = PolicyRAG()
    print("chunks:", len(rag.chunks))
    print(evaluate(rag).to_string(index=False))

    print("\nProject pipeline finished. Review outputs/ and data/instamart.db.")


if __name__ == "__main__":
    main()
