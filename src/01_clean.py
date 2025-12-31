# ============================================================
# 01_clean.py
# Enhanced data cleaning & feature engineering for causal modeling
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cardio_train_age_fixed.csv"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_CLEAN = OUT_DIR / "cardio_clean.csv"
OUT_REPORT = OUT_DIR / "cleaning_report.txt"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
class ValidationRules:
    """Clinically validated ranges"""
    AGE_MIN, AGE_MAX = 18, 100
    HEIGHT_MIN, HEIGHT_MAX = 120, 220  # cm
    WEIGHT_MIN, WEIGHT_MAX = 30, 250   # kg
    BMI_MIN, BMI_MAX = 10, 60
    SBP_MIN, SBP_MAX = 60, 250         # systolic BP (mmHg)
    DBP_MIN, DBP_MAX = 40, 150         # diastolic BP (mmHg)


# ------------------------------------------------------------
# Load & validate
# ------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    """Load raw data with validation"""
    if not path.exists():
        raise FileNotFoundError(f"❌ CSV not found: {path}")
    
    df = pd.read_csv(path)
    
    required_cols = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", 
                     "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")
    
    return df


# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features"""
    df = df.copy()
    
    # BMI calculation
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    
    # Pulse pressure (clinical indicator)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    
    # MAP (Mean Arterial Pressure)
    df["map"] = df["ap_lo"] + (df["pulse_pressure"] / 3)
    
    return df


# ------------------------------------------------------------
# Data Validation
# ------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Apply clinical validation rules and return clean data + report"""
    
    report = {
        "total_rows": len(df),
        "removed": {}
    }
    
    # Age validation
    age_valid = (df["age"] >= ValidationRules.AGE_MIN) & (df["age"] <= ValidationRules.AGE_MAX)
    report["removed"]["invalid_age"] = int((~age_valid).sum())
    
    # Height validation
    height_valid = (df["height"] >= ValidationRules.HEIGHT_MIN) & (df["height"] <= ValidationRules.HEIGHT_MAX)
    report["removed"]["invalid_height"] = int((~height_valid).sum())
    
    # Weight validation
    weight_valid = (df["weight"] >= ValidationRules.WEIGHT_MIN) & (df["weight"] <= ValidationRules.WEIGHT_MAX)
    report["removed"]["invalid_weight"] = int((~weight_valid).sum())
    
    # Blood pressure validation
    sbp_valid = (df["ap_hi"] >= ValidationRules.SBP_MIN) & (df["ap_hi"] <= ValidationRules.SBP_MAX)
    dbp_valid = (df["ap_lo"] >= ValidationRules.DBP_MIN) & (df["ap_lo"] <= ValidationRules.DBP_MAX)
    bp_logical = df["ap_hi"] > df["ap_lo"]
    bp_valid = sbp_valid & dbp_valid & bp_logical
    report["removed"]["invalid_bp"] = int((~bp_valid).sum())
    
    # BMI validation
    bmi_valid = (df["bmi"] >= ValidationRules.BMI_MIN) & (df["bmi"] <= ValidationRules.BMI_MAX)
    report["removed"]["invalid_bmi"] = int((~bmi_valid).sum())
    
    # Binary variables validation (0 or 1)
    binary_vars = ["gender", "smoke", "alco", "active", "cardio"]
    binary_valid = pd.Series(True, index=df.index)
    for var in binary_vars:
        var_valid = df[var].isin([0, 1, 2])  # Some datasets use 1/2 encoding
        binary_valid &= var_valid
    report["removed"]["invalid_binary"] = int((~binary_valid).sum())
    
    # Ordinal variables validation (1, 2, or 3)
    ordinal_vars = ["cholesterol", "gluc"]
    ordinal_valid = pd.Series(True, index=df.index)
    for var in ordinal_vars:
        var_valid = df[var].isin([1, 2, 3])
        ordinal_valid &= var_valid
    report["removed"]["invalid_ordinal"] = int((~ordinal_valid).sum())
    
    # Combined validation
    all_valid = (age_valid & height_valid & weight_valid & bp_valid & 
                 bmi_valid & binary_valid & ordinal_valid)
    
    report["removed"]["total"] = int((~all_valid).sum())
    report["kept_rows"] = int(all_valid.sum())
    report["removal_rate"] = f"{(report['removed']['total'] / report['total_rows'] * 100):.2f}%"
    
    df_clean = df[all_valid].copy()
    
    return df_clean, report


# ------------------------------------------------------------
# Data Normalization
# ------------------------------------------------------------
def normalize_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize variable encodings for consistency"""
    df = df.copy()
    
    # Gender: convert 1/2 to 0/1 (0=Female, 1=Male)
    unique_gender = df["gender"].unique()
    if set(unique_gender).issubset({1, 2}):
        df["gender"] = (df["gender"] - 1).astype(int)
    elif set(unique_gender).issubset({0, 1}):
        pass  # Already normalized
    else:
        # Force to 0/1 by clipping
        df["gender"] = df["gender"].clip(lower=0, upper=1).astype(int)
    
    # Binary variables: ensure 0/1 encoding
    binary_vars = ["smoke", "alco", "active", "cardio"]
    for var in binary_vars:
        unique_vals = df[var].unique()
        if set(unique_vals).issubset({1, 2}):
            df[var] = (df[var] - 1).astype(int)
        elif set(unique_vals).issubset({0, 1}):
            pass  # Already normalized
        else:
            # Force to 0/1 by clipping
            df[var] = df[var].clip(lower=0, upper=1).astype(int)
    
    return df


# ------------------------------------------------------------
# Outlier Detection
# ------------------------------------------------------------
def detect_outliers(df: pd.DataFrame, cols: list, threshold: float = 3.0) -> pd.DataFrame:
    """Flag outliers using z-score method"""
    df = df.copy()
    
    for col in cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df[f"{col}_outlier"] = z_scores > threshold
    
    return df


# ------------------------------------------------------------
# Reporting
# ------------------------------------------------------------
def generate_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame, 
                   validation_report: dict, output_path: Path):
    """Generate comprehensive cleaning report"""
    
    lines = []
    lines.append("=" * 80)
    lines.append("DATA CLEANING REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append(f"Total rows (raw):     {validation_report['total_rows']:,}")
    lines.append(f"Rows kept:            {validation_report['kept_rows']:,}")
    lines.append(f"Rows removed:         {validation_report['removed']['total']:,}")
    lines.append(f"Removal rate:         {validation_report['removal_rate']}")
    lines.append("")
    
    lines.append("REMOVAL BREAKDOWN")
    lines.append("-" * 80)
    for reason, count in validation_report['removed'].items():
        if reason != 'total':
            lines.append(f"{reason:.<30} {count:>8,} rows")
    lines.append("")
    
    lines.append("VARIABLE STATISTICS (CLEANED DATA)")
    lines.append("-" * 80)
    
    continuous_vars = ["age", "height", "weight", "bmi", "ap_hi", "ap_lo"]
    for var in continuous_vars:
        if var in df_clean.columns:
            lines.append(f"\n{var.upper()}")
            lines.append(f"  Mean:   {df_clean[var].mean():.2f}")
            lines.append(f"  Median: {df_clean[var].median():.2f}")
            lines.append(f"  Std:    {df_clean[var].std():.2f}")
            lines.append(f"  Range:  [{df_clean[var].min():.1f}, {df_clean[var].max():.1f}]")
    
    lines.append("")
    lines.append("OUTCOME DISTRIBUTION")
    lines.append("-" * 80)
    cardio_dist = df_clean["cardio"].value_counts(normalize=True)
    cardio_counts = df_clean["cardio"].value_counts()
    
    # Handle both 0 and 1 safely
    if 0 in cardio_dist.index:
        lines.append(f"No CVD (0):  {cardio_dist[0]:.1%} ({cardio_counts[0]:,} patients)")
    else:
        lines.append(f"No CVD (0):  0.0% (0 patients)")
    
    if 1 in cardio_dist.index:
        lines.append(f"Has CVD (1): {cardio_dist[1]:.1%} ({cardio_counts[1]:,} patients)")
    else:
        lines.append(f"Has CVD (1): 0.0% (0 patients)")
    
    lines.append("")
    
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    
    return report_text


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("\n" + "=" * 80)
    print("STEP 1: DATA CLEANING & FEATURE ENGINEERING")
    print("=" * 80 + "\n")
    
    # Load
    df_raw = load_raw(DATA_PATH)
    print(f"✓ Loaded raw data: {df_raw.shape}")
    
    # Remove ID if present
    if "id" in df_raw.columns:
        df_raw = df_raw.drop(columns=["id"])
    
    # Engineer features
    df_raw = engineer_features(df_raw)
    print(f"✓ Engineered features: BMI, pulse pressure, MAP")
    
    # Validate
    df_clean, validation_report = validate_data(df_raw)
    print(f"✓ Validated data: {df_clean.shape} (removed {validation_report['removed']['total']:,} rows)")
    
    # Normalize
    print(f"✓ Before normalization - cardio values: {sorted(df_clean['cardio'].unique())}")
    df_clean = normalize_encodings(df_clean)
    print(f"✓ After normalization - cardio values: {sorted(df_clean['cardio'].unique())}")
    print(f"✓ Normalized encodings (gender: 0/1, binary vars: 0/1)")
    print(f"✓ Cardio distribution: {df_clean['cardio'].value_counts().to_dict()}")
    
    # Save
    df_clean.to_csv(OUT_CLEAN, index=False)
    print(f"✓ Saved clean data: {OUT_CLEAN}")
    
    # Report
    report_text = generate_report(df_raw, df_clean, validation_report, OUT_REPORT)
    print(f"✓ Generated report: {OUT_REPORT}")
    print("\n" + report_text)
    
    print("\n" + "=" * 80)
    print("✅ Step 1 complete. Next: python src/02_discovery.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
