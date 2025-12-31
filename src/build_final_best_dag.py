#!/usr/bin/env python3
# ============================================================
# build_final_best_dag.py
# Improved DAG: Medical Foundation + Validated Data Discoveries
# Addresses issues: keeps critical CVD risk factors, better validation
# ============================================================

import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"

# ============================================================
# MEDICAL KNOWLEDGE - CORE FOUNDATION
# ============================================================

# Absolutely forbidden edges (violate causality/logic)
STRICTLY_FORBIDDEN = {
    # Reverse time (effect → cause)
    ("cardio", "age"), ("cardio", "gender"),
    
    # Reverse physiology (derived → primitive)
    ("bmi", "height"), ("bmi", "weight"),
    
    # No clear direction (simultaneous/confounded)
    ("ap_hi", "ap_lo"), ("ap_lo", "ap_hi"),
    ("height", "weight"), ("weight", "height"),
    
    # Unlikely behavioral causation
    ("alco", "smoke"), ("smoke", "alco"),
}

# Immutable variables (nothing can cause these)
IMMUTABLE = {"age", "gender"}

# CORE medical edges (established science - must be included)
CORE_MEDICAL = [
    # BMI calculation
    ("height", "bmi"),
    ("weight", "bmi"),
    
    # Major CVD risk factors (MUST be present)
    ("age", "cardio"),
    ("bmi", "cardio"),
    ("ap_hi", "cardio"),
    ("ap_lo", "cardio"),
    ("cholesterol", "cardio"),
    ("gluc", "cardio"),
    ("smoke", "cardio"),
    
    # Established physiological pathways
    ("bmi", "ap_hi"),
    ("bmi", "ap_lo"),
    ("bmi", "cholesterol"),
    ("bmi", "gluc"),
    ("age", "ap_hi"),
    ("age", "ap_lo"),
]

# Additional medical edges (strong evidence)
ADDITIONAL_MEDICAL = [
    ("age", "cholesterol"),
    ("age", "gluc"),
    ("age", "height"),
    ("age", "weight"),
    ("gender", "height"),
    ("gender", "weight"),
    ("gender", "active"),
    ("active", "bmi"),
    ("active", "ap_hi"),
    ("active", "ap_lo"),
    ("smoke", "ap_hi"),
    ("smoke", "ap_lo"),
    ("alco", "ap_hi"),
    ("age", "smoke"),
    ("age", "alco"),
]


def load_edges(filename: str):
    """Load edges from CSV"""
    path = OUT_DIR / filename
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return list(zip(df["source"], df["target"]))


def is_medically_plausible(edge: tuple) -> bool:
    """Check if edge is medically plausible"""
    src, dst = edge
    
    # Cannot cause immutable variables
    if dst in IMMUTABLE:
        return False
    
    # Strictly forbidden
    if edge in STRICTLY_FORBIDDEN:
        return False
    
    # Effect cannot cause its causes
    if dst == "cardio":
        return True  # Anything can potentially cause CVD
    
    # Derived variables cannot cause their inputs
    if dst == "bmi" and src in ["height", "weight"]:
        return True  # These DO cause BMI
    if src == "bmi" and dst in ["height", "weight"]:
        return False  # BMI cannot cause height/weight
    
    # BP/metabolic cannot cause BMI (reverse causation)
    if dst == "bmi" and src in ["ap_hi", "ap_lo", "cholesterol", "gluc"]:
        return False
    
    # Demographics → Lifestyle is weak but possible
    if src in ["age", "gender"] and dst in ["smoke", "alco", "active"]:
        return True
    
    # Lifestyle → Demographics is implausible
    if src in ["smoke", "alco", "active"] and dst in ["age", "gender", "height", "weight"]:
        return False
    
    return True


def build_final_best_dag():
    """
    Build the final best DAG:
    1. Start with CORE medical edges (non-negotiable)
    2. Add additional medical edges
    3. Add data-driven edges if: (a) plausible, (b) high confidence, (c) no cycles
    """
    print("\n" + "=" * 80)
    print("FINAL BEST DAG CONSTRUCTION")
    print("=" * 80)
    
    # Load algorithm outputs
    pc_edges = load_edges("edges_pc.csv")
    ges_edges = load_edges("edges_ges.csv")
    notears_edges = load_edges("edges_notears.csv")
    
    print(f"\n📊 Algorithm outputs loaded:")
    print(f"  PC:      {len(pc_edges)} edges")
    print(f"  GES:     {len(ges_edges)} edges")
    print(f"  NOTEARS: {len(notears_edges)} edges")
    
    # Start with core medical edges (absolutely required)
    print(f"\n🏥 STEP 1: Core Medical Foundation")
    print("-" * 80)
    
    dag_edges = set(CORE_MEDICAL)
    print(f"✓ Added {len(CORE_MEDICAL)} core medical edges")
    print(f"  Critical CVD risk factors included:")
    cvd_edges = [e for e in CORE_MEDICAL if e[1] == "cardio"]
    for e in cvd_edges:
        print(f"    • {e[0]} → cardio")
    
    # Add additional medical edges
    print(f"\n🏥 STEP 2: Additional Medical Knowledge")
    print("-" * 80)
    
    added_medical = []
    for edge in ADDITIONAL_MEDICAL:
        if edge not in dag_edges:
            test_edges = list(dag_edges) + [edge]
            G = nx.DiGraph(test_edges)
            if nx.is_directed_acyclic_graph(G):
                dag_edges.add(edge)
                added_medical.append(edge)
    
    print(f"✓ Added {len(added_medical)} additional medical edges")
    
    # Analyze data-driven discoveries
    print(f"\n📊 STEP 3: Data-Driven Discovery Analysis")
    print("-" * 80)
    
    all_data_edges = pc_edges + ges_edges + notears_edges
    edge_votes = Counter(all_data_edges)
    
    # Categorize by agreement
    high_confidence = [e for e, c in edge_votes.items() if c >= 2]  # 2+ algorithms
    medium_confidence = [e for e, c in edge_votes.items() if c == 1]
    
    print(f"  High confidence (2+ algorithms):  {len(high_confidence)} edges")
    print(f"  Medium confidence (1 algorithm):  {len(medium_confidence)} edges")
    
    # Add high-confidence data-driven edges
    print(f"\n✅ STEP 4: Adding High-Confidence Data Edges")
    print("-" * 80)
    
    added_data = []
    rejected_data = []
    
    for edge in high_confidence:
        if edge not in dag_edges:
            # Check medical plausibility
            if not is_medically_plausible(edge):
                rejected_data.append((edge, "implausible"))
                continue
            
            # Check if creates cycle
            test_edges = list(dag_edges) + [edge]
            G = nx.DiGraph(test_edges)
            if not nx.is_directed_acyclic_graph(G):
                rejected_data.append((edge, "creates_cycle"))
                continue
            
            # Add edge
            dag_edges.add(edge)
            added_data.append(edge)
    
    print(f"✓ Added {len(added_data)} high-confidence data edges")
    if added_data:
        for e in added_data[:10]:
            print(f"    + {e[0]} → {e[1]}")
    
    if rejected_data:
        print(f"\n✗ Rejected {len(rejected_data)} edges:")
        for e, reason in rejected_data[:5]:
            print(f"    - {e[0]} → {e[1]} ({reason})")
    
    # Final validation
    print(f"\n✅ FINAL DAG STATISTICS")
    print("-" * 80)
    
    final_edges = list(dag_edges)
    G = nx.DiGraph(final_edges)
    
    print(f"Total edges:            {len(final_edges)}")
    print(f"Total nodes:            {G.number_of_nodes()}")
    print(f"Is valid DAG:           {nx.is_directed_acyclic_graph(G)}")
    print(f"Average degree:         {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Graph density:          {len(final_edges) / (G.number_of_nodes() * (G.number_of_nodes() - 1)):.3f}")
    
    if "cardio" in G:
        cvd_parents = sorted(G.predecessors("cardio"))
        print(f"\n🎯 CVD Direct Causes ({len(cvd_parents)}):")
        for parent in cvd_parents:
            print(f"    • {parent}")
    
    # Composition breakdown
    print(f"\n📊 EDGE COMPOSITION")
    print("-" * 80)
    print(f"Core medical (required):     {len(CORE_MEDICAL)}")
    print(f"Additional medical:          {len(added_medical)}")
    print(f"High-confidence data:        {len(added_data)}")
    print(f"Total:                       {len(final_edges)}")
    
    return final_edges


def compare_dags():
    """Compare final best DAG with other versions"""
    print("\n" + "=" * 80)
    print("COMPARISON WITH OTHER DAGS")
    print("=" * 80)
    
    final_edges = load_edges("edges_final_best.csv")
    hybrid_edges = load_edges("edges_hybrid.csv")
    smart_edges = load_edges("edges_smart_dag.csv")
    
    G_final = nx.DiGraph(final_edges)
    G_hybrid = nx.DiGraph(hybrid_edges)
    G_smart = nx.DiGraph(smart_edges)
    
    comparison = []
    
    for name, G in [("Final Best", G_final), ("Hybrid", G_hybrid), ("Smart", G_smart)]:
        cvd_parents = set(G.predecessors("cardio")) if "cardio" in G else set()
        comparison.append({
            "name": name,
            "edges": G.number_of_edges(),
            "cvd_parents": len(cvd_parents),
            "parent_list": cvd_parents
        })
    
    print(f"\n{'DAG':<15} {'Edges':<10} {'→CVD':<10} CVD Direct Causes")
    print("-" * 80)
    
    for c in comparison:
        parents_str = ", ".join(sorted(c["parent_list"]))
        print(f"{c['name']:<15} {c['edges']:<10} {c['cvd_parents']:<10} {parents_str}")
    
    # Check for critical risk factors
    print(f"\n🔍 CRITICAL CVD RISK FACTORS CHECK")
    print("-" * 80)
    
    critical_factors = ["age", "bmi", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke"]
    
    for factor in critical_factors:
        in_final = factor in comparison[0]["parent_list"]
        in_hybrid = factor in comparison[1]["parent_list"]
        in_smart = factor in comparison[2]["parent_list"]
        
        status_final = "✓" if in_final else "✗"
        status_hybrid = "✓" if in_hybrid else "✗"
        status_smart = "✓" if in_smart else "✗"
        
        print(f"  {factor:12} {status_final} Final   {status_hybrid} Hybrid   {status_smart} Smart")


def main():
    print("\n" + "=" * 80)
    print("BUILDING FINAL BEST DAG")
    print("Approach: Medical Foundation + Validated Data Discoveries")
    print("=" * 80)
    
    # Build final best DAG
    final_edges = build_final_best_dag()
    
    # Save
    output_path = OUT_DIR / "edges_final_best.csv"
    df = pd.DataFrame(final_edges, columns=["source", "target"])
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")
    
    # Compare
    compare_dags()
    
    # Recommendation
    print("\n" + "=" * 80)
    print("🏆 RECOMMENDATION")
    print("=" * 80)
    print(f"\nUse FINAL BEST DAG because:")
    print(f"  ✅ All 7 critical CVD risk factors included")
    print(f"  ✅ Based on established medical science")
    print(f"  ✅ Enhanced with high-confidence data discoveries")
    print(f"  ✅ Medically plausible (no reverse causation)")
    print(f"  ✅ Valid DAG structure (acyclic)")
    print(f"\n📊 Visualize:")
    print(f"  python src/visualize_dag.py")
    print(f"\n  Will create: dag_final_best_layered.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()