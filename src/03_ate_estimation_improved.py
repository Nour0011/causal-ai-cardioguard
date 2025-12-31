# ============================================================
# 03_ate_estimation_improved.py
# Robust Average Treatment Effect (ATE) Estimation
# FIXED:
#  - Uses edges_final_best.csv
#  - Ensures correct encodings for treatments/outcome
#  - Converts ORDINAL (cholesterol/gluc) to binary "high vs normal"
#  - Uses control_value/treatment_value for binary (prevents sign confusion)
#  - Logs failures into report (report is never empty)
#  - Handles PermissionError (writes __new file if locked)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"

DATA_PATH = OUT_DIR / "cardio_clean.csv"
EDGES_PATH = OUT_DIR / "edges_final_best.csv"

OUT_ATE_CSV = OUT_DIR / "ate_results.csv"
OUT_ATE_JSON = OUT_DIR / "ate_results.json"
OUT_ATE_REPORT = OUT_DIR / "ate_report.txt"


# -------------------------
# Safe writers (handle Windows file locks)
# -------------------------
def safe_path_if_locked(path: Path) -> Path:
    """
    If target file is locked (PermissionError), return a new available filename:
    name__new.ext, name__new2.ext, ...
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    # first try __new
    candidate = parent / f"{stem}__new{suffix}"
    if not candidate.exists():
        return candidate

    # then __new2, __new3...
    for i in range(2, 1000):
        candidate = parent / f"{stem}__new{i}{suffix}"
        if not candidate.exists():
            return candidate

    # fallback (very unlikely)
    return parent / f"{stem}__new999{suffix}"


def safe_to_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    try:
        df.to_csv(path, index=index)
        return path
    except PermissionError:
        new_path = safe_path_if_locked(path)
        print(f"⚠️  Permission denied writing {path} (file may be open/locked).")
        df.to_csv(new_path, index=index)
        print(f"✅ Saved instead to: {new_path}")
        return new_path


def safe_write_text(text: str, path: Path, encoding: str = "utf-8") -> Path:
    try:
        path.write_text(text, encoding=encoding)
        return path
    except PermissionError:
        new_path = safe_path_if_locked(path)
        print(f"⚠️  Permission denied writing {path} (file may be open/locked).")
        new_path.write_text(text, encoding=encoding)
        print(f"✅ Saved instead to: {new_path}")
        return new_path


def safe_write_json(obj: dict, path: Path, encoding: str = "utf-8") -> Path:
    try:
        with open(path, "w", encoding=encoding) as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        return path
    except PermissionError:
        new_path = safe_path_if_locked(path)
        print(f"⚠️  Permission denied writing {path} (file may be open/locked).")
        with open(new_path, "w", encoding=encoding) as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved instead to: {new_path}")
        return new_path


# -------------------------
# Load data & graph
# -------------------------
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def load_edges() -> pd.DataFrame:
    if not EDGES_PATH.exists():
        raise FileNotFoundError(f"Missing: {EDGES_PATH}")
    edges = pd.read_csv(EDGES_PATH)
    if not {"source", "target"}.issubset(edges.columns):
        raise ValueError("Edges file must have 'source' and 'target' columns")
    return edges


def edges_to_dot(edges_df: pd.DataFrame) -> str:
    lines = ["digraph {"]
    for _, row in edges_df.iterrows():
        src = str(row["source"]).strip()
        tgt = str(row["target"]).strip()
        lines.append(f'  "{src}" -> "{tgt}";')
    lines.append("}")
    return "\n".join(lines)


# -------------------------
# Treatment definitions
# -------------------------
class TreatmentDefinition:
    # 0->1 means "more of that factor"
    BINARY = {
        "smoke": "Smoking (0→1)",
        "alco": "Alcohol consumption (0→1)",
        "active": "Physical activity (0→1)",
    }

    CONTINUOUS = {
        "bmi": "BMI (+1 unit)",
        "ap_hi": "Systolic BP (+1 mmHg)",
        "ap_lo": "Diastolic BP (+1 mmHg)",
    }

    # Converted to binary:
    # 0 = normal, 1 = high (Above/Well Above). 0->1 means worse.
    ORDINAL = {
        "cholesterol": "High cholesterol (normal→high)",
        "gluc": "High glucose (normal→high)",
    }

    @classmethod
    def get_all_treatments(cls):
        return list(cls.BINARY.keys()) + list(cls.CONTINUOUS.keys()) + list(cls.ORDINAL.keys())


# -------------------------
# Preprocessing
# -------------------------
def to_binary_01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")

    # 1/2 -> 0/1
    if x.dropna().isin([1, 2]).all():
        x = x - 1

    x = x.fillna(0)
    x = x.clip(0, 1).astype(int)
    return x


def ordinal_to_high_normal(s: pd.Series) -> pd.Series:
    """
    Convert ordinal to binary high vs normal:
      normal -> 0
      above/well-above -> 1
    Supports:
      1/2/3 encoding or 0/1/2 encoding.
    """
    x = pd.to_numeric(s, errors="coerce")
    vals = set(x.dropna().unique().tolist())

    if vals.issubset({1, 2, 3}):
        x = (x >= 2).astype(int)   # 1 normal, 2/3 high
    elif vals.issubset({0, 1, 2}):
        x = (x >= 1).astype(int)   # 0 normal, 1/2 high
    else:
        x = x.fillna(0)
        x = (x >= 1).astype(int)

    return x


def prepare_ate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["cardio"] = to_binary_01(df["cardio"])

    for col in TreatmentDefinition.BINARY:
        if col in df.columns:
            df[col] = to_binary_01(df[col])

    for col in TreatmentDefinition.ORDINAL:
        if col in df.columns:
            df[col] = ordinal_to_high_normal(df[col])

    for col in TreatmentDefinition.CONTINUOUS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    needed = TreatmentDefinition.get_all_treatments() + ["cardio"]
    needed = [c for c in needed if c in df.columns]
    df = df.dropna(subset=needed)

    return df


# -------------------------
# ATE estimation
# -------------------------
def estimate_ate(
    df: pd.DataFrame,
    graph_dot: str,
    treatment: str,
    outcome: str = "cardio",
    method: str = "backdoor.linear_regression",
    is_binary: bool = False,
) -> dict:
    try:
        from dowhy import CausalModel
    except ImportError:
        return {"treatment": treatment, "outcome": outcome, "error": "DoWhy not installed. Run: pip install dowhy"}

    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        graph=graph_dot,
        proceed_when_unidentifiable=True,
    )

    identified_estimand = model.identify_effect()

    res = {
        "treatment": treatment,
        "outcome": outcome,
        "method": method,
        "ate": None,
        "backdoor_set": [],
        "error": None,
    }

    try:
        bset = identified_estimand.get_backdoor_variables()
        res["backdoor_set"] = list(bset) if bset else []
    except:
        res["backdoor_set"] = []

    try:
        if is_binary:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method,
                method_params={"control_value": 0, "treatment_value": 1},
            )
        else:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method,
            )

        res["ate"] = float(estimate.value)

    except Exception as e:
        res["error"] = str(e)

    return res


# -------------------------
# Interpretation
# -------------------------
def interpret_ate(treatment: str, ate: float, ttype: str) -> dict:
    magnitude = "small" if abs(ate) < 0.05 else ("moderate" if abs(ate) < 0.15 else "large")
    direction = "increases" if ate > 0 else "decreases"

    if ttype == "binary":
        clinical = f"{treatment} (0→1) {direction} CVD risk by ~{abs(ate)*100:.1f} pp."
    elif ttype == "continuous":
        clinical = f"{treatment} (+1) {direction} CVD risk by ~{abs(ate)*100:.1f} pp."
    else:
        clinical = f"{treatment} (normal→high) {direction} CVD risk by ~{abs(ate)*100:.1f} pp."

    return {"magnitude": magnitude, "direction": direction, "clinical_relevance": clinical}


# -------------------------
# Report
# -------------------------
def generate_ate_report(results_ok: list, results_fail: list) -> Path:
    lines = []
    lines.append("=" * 80)
    lines.append("AVERAGE TREATMENT EFFECT (ATE) REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("NOTE")
    lines.append("-" * 80)
    lines.append("ATE is estimated from observational data, so confounding can flip signs.")
    lines.append("For clinical UI, apply guardrails (e.g., smoking/alcohol cannot be 'protective').")
    lines.append("")

    lines.append("RESULTS (SUCCESSFUL ESTIMATES)")
    lines.append("-" * 80)
    if not results_ok:
        lines.append("No successful ATE estimates were produced.")
    else:
        for i, r in enumerate(results_ok, 1):
            lines.append(f"\n{i}. {r['treatment'].upper()} ({r['type']})")
            lines.append(f"   Description:  {r['description']}")
            lines.append(f"   ATE:          {r['ate']:+.6f}")
            lines.append(f"   Effect:       {r['magnitude']} {r['direction']}")
            lines.append(f"   Clinical:     {r['clinical_relevance']}")
            lines.append(f"   Adjusted for: {r['backdoor_set']}")

    lines.append("\nFAILED ESTIMATIONS (DEBUG)")
    lines.append("-" * 80)
    if not results_fail:
        lines.append("No failures recorded.")
    else:
        for r in results_fail:
            lines.append(f"• {r['treatment']}: {r.get('error')}")

    return safe_write_text("\n".join(lines), OUT_ATE_REPORT)


# -------------------------
# Main
# -------------------------
def main():
    print("\n" + "=" * 80)
    print("STEP 3: AVERAGE TREATMENT EFFECT (ATE) ESTIMATION")
    print("=" * 80)

    df_raw = load_data()
    print(f"\n✓ Loaded data: {df_raw.shape}")

    edges_df = load_edges()
    graph_dot = edges_to_dot(edges_df)
    print(f"✓ Loaded causal graph: {len(edges_df)} edges ({EDGES_PATH.name})")

    df = prepare_ate_dataframe(df_raw)
    print(f"✓ Prepared ATE dataframe (after encoding + dropna): {df.shape}")

    treatments = TreatmentDefinition.get_all_treatments()

    all_ok, all_fail = [], []
    ate_summary = {}

    for treatment in treatments:
        if treatment in TreatmentDefinition.BINARY:
            ttype = "binary"
            desc = TreatmentDefinition.BINARY[treatment]
            is_binary = True
        elif treatment in TreatmentDefinition.CONTINUOUS:
            ttype = "continuous"
            desc = TreatmentDefinition.CONTINUOUS[treatment]
            is_binary = False
        else:
            ttype = "ordinal"
            desc = TreatmentDefinition.ORDINAL[treatment]
            is_binary = True  # because we converted it into 0/1

        res = estimate_ate(
            df=df,
            graph_dot=graph_dot,
            treatment=treatment,
            outcome="cardio",
            method="backdoor.linear_regression",
            is_binary=is_binary,
        )

        if res.get("error") or res.get("ate") is None:
            all_fail.append(res)
            continue

        ate = float(res["ate"])
        interp = interpret_ate(treatment, ate, ttype)

        row = {
            "treatment": treatment,
            "description": desc,
            "type": ttype,
            "ate": ate,
            "magnitude": interp["magnitude"],
            "direction": interp["direction"],
            "clinical_relevance": interp["clinical_relevance"],
            "backdoor_set": ", ".join(res.get("backdoor_set", [])),
        }

        all_ok.append(row)
        ate_summary[treatment] = ate

    # Save outputs safely (even if locked)
    df_out = pd.DataFrame(all_ok)
    if not df_out.empty:
        df_out = df_out.sort_values("ate", key=np.abs, ascending=False)

    csv_path = safe_to_csv(df_out, OUT_ATE_CSV, index=False)
    json_path = safe_write_json(ate_summary, OUT_ATE_JSON)
    report_path = generate_ate_report(all_ok, all_fail)

    print(f"\n✓ Saved ATE results: {csv_path}")
    print(f"✓ Saved ATE summary: {json_path}")
    print(f"✓ Saved ATE report:  {report_path}")
    print("\n✅ Step 3 complete. Next: python src/04_bayesian_network_improved.py")


if __name__ == "__main__":
    main()
