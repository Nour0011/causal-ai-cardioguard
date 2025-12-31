# ============================================================
# 04_bayesian_network_improved.py
# Enhanced Bayesian Network with Adaptive Discretization
# WORKING FIX (pgmpy 1.0.0):
#   - Uses DiscreteBayesianNetwork
#   - DOES NOT pass state_names into model.fit (avoids unexpected states)
#   - Saves bn_state_map.json for UI labels + bin edges
#   - Robust CSV saving (handles Windows file lock / permission issues)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork as BNModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"

DATA_PATH = OUT_DIR / "cardio_clean.csv"
EDGES_PATH = OUT_DIR / "edges_final_best.csv"

OUT_BN_MODEL = OUT_DIR / "bn_model.pkl"
OUT_BN_DISCRETE = OUT_DIR / "bn_discretized.csv"
OUT_BN_STATE_MAP = OUT_DIR / "bn_state_map.json"
OUT_BN_REPORT = OUT_DIR / "bn_report.txt"


# -------------------------
# Configuration
# -------------------------
BN_COLS = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "bmi", "cardio"
]

CONTINUOUS_VARS = ["age", "height", "weight", "ap_hi", "ap_lo", "bmi"]
BINARY_VARS = ["gender", "smoke", "alco", "active", "cardio"]
ORDINAL_VARS = ["cholesterol", "gluc"]


# -------------------------
# Load data & edges
# -------------------------
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    missing = set(BN_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[BN_COLS].copy()


def load_edges() -> list[tuple[str, str]]:
    if not EDGES_PATH.exists():
        raise FileNotFoundError(f"Missing: {EDGES_PATH}")

    edges_df = pd.read_csv(EDGES_PATH)
    if not {"source", "target"}.issubset(edges_df.columns):
        raise ValueError("edges_hybrid.csv must contain columns: source,target")

    edges = list(zip(edges_df["source"].astype(str), edges_df["target"].astype(str)))

    # Filter to BN columns only
    valid_edges = [(s, t) for s, t in edges if s in BN_COLS and t in BN_COLS]

    # Ensure DAG (break cycles if exist)
    G = nx.DiGraph(valid_edges)
    if not nx.is_directed_acyclic_graph(G):
        print("⚠️  Warning: Edges contain cycles. Breaking cycles...")
        while True:
            try:
                cycle = nx.find_cycle(G)
                u, v = cycle[-1][0], cycle[-1][1]
                G.remove_edge(u, v)
                valid_edges = [(a, b) for a, b in valid_edges if not (a == u and b == v)]
            except nx.NetworkXNoCycle:
                break

    return valid_edges


# -------------------------
# Smart Discretization
# -------------------------
class AdaptiveDiscretizer:
    """
    Produces integer states (0..K-1) and stores UI labels + bin edges.
    NOTE: We do NOT pass state_names to pgmpy.fit() (pgmpy 1.0.0 strictness).
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.bin_edges: dict[str, list[float]] = {}
        self.state_names: dict[str, list[str]] = {}

    def discretize_continuous(self, col: str, n_bins: int = 3) -> pd.Series:
        x = pd.to_numeric(self.data[col], errors="coerce")

        # Clip extremes (robust to outliers)
        p1, p99 = x.quantile(0.01), x.quantile(0.99)
        x_clean = x.clip(lower=p1, upper=p99)

        try:
            disc, edges = pd.qcut(
                x_clean,
                q=n_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )

            edges = np.array(edges, dtype=float)
            self.bin_edges[col] = edges.tolist()

            labels: list[str] = []
            for i in range(len(edges) - 1):
                if i == 0:
                    label = f"Low (<{edges[i+1]:.1f})"
                elif i == len(edges) - 2:
                    label = f"High (≥{edges[i]:.1f})"
                else:
                    label = f"Mid ({edges[i]:.1f}-{edges[i+1]:.1f})"
                labels.append(label)

            self.state_names[col] = labels

            # Fill NaNs to middle bin
            mid_bin = int(len(labels) // 2) if len(labels) else 0
            out = disc.fillna(mid_bin).astype(int)

            # Safety: clamp to valid range
            if len(labels) > 0:
                out = out.clip(lower=0, upper=len(labels) - 1)

            return out

        except Exception as e:
            print(f"  ⚠️  Discretization failed for {col}: {e}")
            # Fallback: median split -> 2 states
            median = float(x_clean.median())
            self.bin_edges[col] = [float(x_clean.min()), median, float(x_clean.max())]
            self.state_names[col] = [f"Low (<{median:.1f})", f"High (≥{median:.1f})"]
            out = (x_clean >= median).fillna(0).astype(int)
            return out.clip(0, 1)

    def discretize_binary(self, col: str) -> pd.Series:
        x = pd.to_numeric(self.data[col], errors="coerce").fillna(0)

        # Convert 1/2 encoding -> 0/1
        if x.isin([1, 2]).all():
            x = (x - 1).astype(int)

        x = x.clip(0, 1).astype(int)

        if col == "gender":
            self.state_names[col] = ["Female", "Male"]
        elif col == "cardio":
            self.state_names[col] = ["No CVD", "Has CVD"]
        else:
            self.state_names[col] = ["No", "Yes"]

        return x

    def discretize_ordinal(self, col: str) -> pd.Series:
        x = pd.to_numeric(self.data[col], errors="coerce").fillna(1)

        # Convert 1/2/3 -> 0/1/2
        if x.isin([1, 2, 3]).all():
            x = (x - 1).astype(int)

        x = x.clip(0, 2).astype(int)
        self.state_names[col] = ["Normal", "Above Normal", "Well Above Normal"]
        return x

    def discretize_all(self) -> pd.DataFrame:
        df_disc = pd.DataFrame(index=self.data.index)

        print("\nDiscretizing variables:")
        print("-" * 60)

        for col in BN_COLS:
            if col in CONTINUOUS_VARS:
                df_disc[col] = self.discretize_continuous(col, n_bins=3)
                print(f"  ✓ {col:15} (continuous) → {df_disc[col].nunique()} states")
            elif col in BINARY_VARS:
                df_disc[col] = self.discretize_binary(col)
                print(f"  ✓ {col:15} (binary)     → {df_disc[col].nunique()} states")
            elif col in ORDINAL_VARS:
                df_disc[col] = self.discretize_ordinal(col)
                print(f"  ✓ {col:15} (ordinal)    → {df_disc[col].nunique()} states")

        # Force integer dtype
        for c in BN_COLS:
            df_disc[c] = pd.to_numeric(df_disc[c], errors="coerce").fillna(0).astype(int)

        return df_disc


# -------------------------
# Bayesian Network Training
# -------------------------
def train_bayesian_network(edges: list[tuple[str, str]], data: pd.DataFrame) -> BNModel:
    print("\n" + "=" * 80)
    print("TRAINING BAYESIAN NETWORK")
    print("=" * 80)

    model = BNModel()
    model.add_nodes_from(BN_COLS)
    model.add_edges_from(edges)

    print(f"\n✓ Created BN structure:")
    print(f"  Nodes: {len(model.nodes())}")
    print(f"  Edges: {len(model.edges())}")

    print("\n⚙️  Learning conditional probability distributions (CPDs)...")

    # KEY FIX: do NOT pass state_names (pgmpy 1.0.0 can be strict)
    model.fit(
        data=data,
        estimator=MaximumLikelihoodEstimator,
    )

    is_valid = model.check_model()
    print(f"✓ Model validation: {'PASSED' if is_valid else 'FAILED'}")
    if not is_valid:
        raise ValueError("BN model validation failed!")

    print("\n📊 Sample CPD (cardio outcome):")
    print(model.get_cpds("cardio"))

    return model


# -------------------------
# Model Evaluation
# -------------------------
def evaluate_model(model: BNModel, data: pd.DataFrame) -> dict:
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    inference = VariableElimination(model)
    true_labels = data["cardio"].astype(int).values

    print("\n⚙️  Performing inference on test samples...")

    sample_size = min(1000, len(data))
    sample_idx = np.random.choice(len(data), sample_size, replace=False)

    predictions: list[float] = []
    for idx in sample_idx:
        evidence = {col: int(data.iloc[idx][col]) for col in BN_COLS if col != "cardio"}
        try:
            result = inference.query(variables=["cardio"], evidence=evidence, show_progress=False)
            prob_cvd = float(result.values[1]) if result.values.shape[0] > 1 else 0.0
            predictions.append(prob_cvd)
        except Exception:
            predictions.append(0.5)

    predictions = np.array(predictions, dtype=float)
    true_labels_sample = true_labels[sample_idx]

    pred_binary = (predictions >= 0.5).astype(int)
    accuracy = float((pred_binary == true_labels_sample).mean())

    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(true_labels_sample)) == 2:
            auc = float(roc_auc_score(true_labels_sample, predictions))
    except Exception:
        auc = None

    metrics = {
        "accuracy": accuracy,
        "auc": auc,
        "mean_pred_prob": float(predictions.mean()),
        "sample_size": int(sample_size),
    }

    print(f"\n📈 Performance Metrics (on {sample_size} samples):")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  AUC:      {auc:.3f}" if auc is not None else "  AUC:      N/A (needs both classes)")

    return metrics


# -------------------------
# Save artifacts
# -------------------------
def save_state_map(state_names: dict, bin_edges: dict, output_path: Path):
    """Save state mapping for UI (labels + bin edges)."""
    state_map = {}

    for col in BN_COLS:
        state_map[col] = {"type": "", "states": {}, "bin_edges": []}

        if col in CONTINUOUS_VARS:
            state_map[col]["type"] = "continuous"
            state_map[col]["bin_edges"] = bin_edges.get(col, [])
        elif col in BINARY_VARS:
            state_map[col]["type"] = "binary"
        elif col in ORDINAL_VARS:
            state_map[col]["type"] = "ordinal"

        labels = state_names.get(col, [])
        state_map[col]["states"] = {str(i): labels[i] for i in range(len(labels))}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state_map, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved state map: {output_path}")


def generate_bn_report(edges: list[tuple[str, str]], metrics: dict, output_path: Path):
    lines = []
    lines.append("=" * 80)
    lines.append("BAYESIAN NETWORK REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("MODEL STRUCTURE")
    lines.append("-" * 80)
    lines.append(f"Nodes:  {len(BN_COLS)}")
    lines.append(f"Edges:  {len(edges)}")
    lines.append("")

    lines.append("NODES")
    lines.append("-" * 80)
    for col in BN_COLS:
        if col in CONTINUOUS_VARS:
            lines.append(f"  • {col:15} (continuous, 3 bins)")
        elif col in BINARY_VARS:
            lines.append(f"  • {col:15} (binary)")
        elif col in ORDINAL_VARS:
            lines.append(f"  • {col:15} (ordinal, 3 levels)")
    lines.append("")

    if metrics:
        lines.append("PREDICTIVE PERFORMANCE")
        lines.append("-" * 80)
        lines.append(f"Accuracy: {metrics['accuracy']:.3f}")
        if metrics.get("auc") is not None:
            lines.append(f"AUC:      {metrics['auc']:.3f}")
        else:
            lines.append("AUC:      N/A (needs both classes)")
        lines.append("")

    lines.append("=" * 80)

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"✓ Generated report: {output_path}")


def safe_save_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Windows-friendly CSV write:
    - creates outputs folder
    - if target locked, writes to bn_discretized__new.csv instead of crashing
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "__new" + path.suffix)
        df.to_csv(alt, index=False)
        print(f"⚠️  Permission denied writing {path} (file may be open/locked).")
        print(f"✅ Saved instead to: {alt}")
        return alt


# -------------------------
# Main
# -------------------------
def main():
    print("\n" + "=" * 80)
    print("STEP 4: BAYESIAN NETWORK")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"\n✓ Loaded data: {df.shape}")

    edges = load_edges()
    print(f"✓ Loaded causal structure: {len(edges)} edges")

    discretizer = AdaptiveDiscretizer(df)
    df_disc = discretizer.discretize_all()

    saved_csv = safe_save_csv(df_disc, OUT_BN_DISCRETE)
    print(f"\n✓ Saved discretized data: {saved_csv}")

    model = train_bayesian_network(edges, df_disc)

    metrics = evaluate_model(model, df_disc)

    joblib.dump(model, OUT_BN_MODEL)
    print(f"\n✓ Saved BN model: {OUT_BN_MODEL}")

    save_state_map(discretizer.state_names, discretizer.bin_edges, OUT_BN_STATE_MAP)

    generate_bn_report(edges, metrics, OUT_BN_REPORT)

    print("\n" + "=" * 80)
    print("✅ Step 4 complete. Next: streamlit run src/05_streamlit_app_improved.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
