#!/usr/bin/env python3
# ============================================================
# build_best_dag.py
# Smart DAG Construction: GES (best data-driven) + Medical Validation
# This approach uses the best-performing algorithm as base
# ============================================================

import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"

# Medical knowledge for validation
IMMUTABLE_VARS = {"age", "gender"}  # Cannot be caused by others
FORBIDDEN_EDGES = {
    # Reverse causation (effect → cause)
    ("cardio", "age"), ("cardio", "gender"), ("cardio", "height"), ("cardio", "weight"),
    ("cardio", "smoke"), ("cardio", "alco"), ("cardio", "active"),
    ("cardio", "bmi"), ("cardio", "ap_hi"), ("cardio", "ap_lo"),
    ("cardio", "cholesterol"), ("cardio", "gluc"),
    
    # Illogical relationships
    ("bmi", "height"), ("bmi", "weight"),  # BMI is derived from these
    ("ap_hi", "ap_lo"),  # Usually co-vary, no clear direction
    ("height", "weight"),  # Independent measurements
    ("smoke", "alco"),  # Lifestyle choices, no causation
    ("gluc", "cholesterol"),  # May have reverse edge
}

REQUIRED_MEDICAL_EDGES = [
    # Core physiological facts
    ("height", "bmi"),
    ("weight", "bmi"),
    ("bmi", "ap_hi"),
    ("bmi", "ap_lo"),
    ("age", "ap_hi"),
    ("age", "ap_lo"),
    
    # Major CVD risk factors (well-established)
    ("ap_hi", "cardio"),
    ("ap_lo", "cardio"),
    ("cholesterol", "cardio"),
    ("gluc", "cardio"),
    ("bmi", "cardio"),
    ("smoke", "cardio"),
    ("age", "cardio"),
]


def load_edges(filename: str):
    """Load edges from CSV"""
    path = OUT_DIR / filename
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return list(zip(df["source"], df["target"]))


def score_algorithm_quality(edges: list, name: str) -> dict:
    """Score an algorithm's output quality"""
    if not edges:
        return {"name": name, "score": 0, "edges": [], "quality": "failed"}
    
    G = nx.DiGraph(edges)
    
    score = 0
    issues = []
    
    # Check if DAG
    if not nx.is_directed_acyclic_graph(G):
        score -= 1000
        issues.append("has_cycles")
    else:
        score += 100
    
    # Check CVD parents
    cvd_parents = list(G.predecessors("cardio")) if "cardio" in G else []
    score += len(cvd_parents) * 10
    
    # Edge count (sweet spot: 20-35)
    n_edges = len(edges)
    if 20 <= n_edges <= 35:
        score += 50
    elif 10 <= n_edges < 20:
        score += 20
    elif n_edges < 10:
        score -= 20
        issues.append("too_sparse")
    
    # Check for forbidden edges
    forbidden_count = sum(1 for e in edges if e in FORBIDDEN_EDGES)
    score -= forbidden_count * 20
    if forbidden_count > 0:
        issues.append(f"{forbidden_count}_forbidden_edges")
    
    quality = "excellent" if score > 150 else ("good" if score > 100 else "poor")
    
    return {
        "name": name,
        "score": score,
        "edges": edges,
        "n_edges": n_edges,
        "cvd_parents": len(cvd_parents),
        "quality": quality,
        "issues": issues
    }


def build_smart_dag():
    """
    Build best DAG using smart strategy:
    1. Score all algorithms
    2. Use best algorithm as base
    3. Add high-confidence consensus edges
    4. Add required medical edges
    5. Remove forbidden edges
    6. Ensure DAG property
    """
    print("\n" + "=" * 80)
    print("SMART DAG CONSTRUCTION")
    print("=" * 80)
    
    # Load all algorithm outputs
    algorithms = {
        "PC": load_edges("edges_pc.csv"),
        "GES": load_edges("edges_ges.csv"),
        "NOTEARS": load_edges("edges_notears.csv"),
    }
    
    # Score each algorithm
    print("\n📊 ALGORITHM SCORING")
    print("-" * 80)
    
    scores = []
    for name, edges in algorithms.items():
        score_data = score_algorithm_quality(edges, name)
        scores.append(score_data)
        
        status = "✓" if score_data["quality"] in ["excellent", "good"] else "✗"
        print(f"{status} {name:10} - Score: {score_data['score']:>6.1f}  "
              f"Edges: {score_data['n_edges']:>3}  "
              f"→CVD: {score_data['cvd_parents']:>2}  "
              f"Quality: {score_data['quality']}")
        if score_data['issues']:
            print(f"           Issues: {', '.join(score_data['issues'])}")
    
    # Select best algorithm as base
    scores.sort(key=lambda x: x["score"], reverse=True)
    best_algo = scores[0]
    
    print(f"\n🏆 Best algorithm: {best_algo['name']} (score: {best_algo['score']:.1f})")
    
    # Start with best algorithm's edges
    dag_edges = set(best_algo["edges"])
    print(f"✓ Starting with {len(dag_edges)} edges from {best_algo['name']}")
    
    # Get consensus edges (appear in 2+ algorithms)
    all_edges = []
    for algo_data in algorithms.values():
        all_edges.extend(algo_data)
    
    edge_votes = Counter(all_edges)
    consensus_edges = [e for e, count in edge_votes.items() if count >= 2]
    
    print(f"\n🤝 ADDING CONSENSUS EDGES (2+ algorithms)")
    print("-" * 80)
    
    added_consensus = []
    for edge in consensus_edges:
        if edge not in dag_edges:
            test_edges = list(dag_edges) + [edge]
            G = nx.DiGraph(test_edges)
            
            if nx.is_directed_acyclic_graph(G) and edge not in FORBIDDEN_EDGES:
                src, dst = edge
                if dst not in IMMUTABLE_VARS:
                    dag_edges.add(edge)
                    added_consensus.append(edge)
    
    print(f"✓ Added {len(added_consensus)} consensus edges")
    
    # Add required medical edges
    print(f"\n🏥 ADDING REQUIRED MEDICAL EDGES")
    print("-" * 80)
    
    added_medical = []
    for edge in REQUIRED_MEDICAL_EDGES:
        if edge not in dag_edges:
            test_edges = list(dag_edges) + [edge]
            G = nx.DiGraph(test_edges)
            
            if nx.is_directed_acyclic_graph(G):
                dag_edges.add(edge)
                added_medical.append(edge)
    
    print(f"✓ Added {len(added_medical)} required medical edges")
    
    # Remove forbidden edges
    print(f"\n🚫 REMOVING FORBIDDEN EDGES")
    print("-" * 80)
    
    removed = []
    for edge in list(dag_edges):
        if edge in FORBIDDEN_EDGES:
            dag_edges.remove(edge)
            removed.append(edge)
    
    if removed:
        print(f"✗ Removed {len(removed)} forbidden edges:")
        for e in removed[:5]:
            print(f"    - {e[0]} → {e[1]}")
    else:
        print(f"✓ No forbidden edges found")
    
    # Remove edges to immutable variables
    print(f"\n🔒 CHECKING IMMUTABLE VARIABLES")
    print("-" * 80)
    
    removed_immutable = []
    for edge in list(dag_edges):
        src, dst = edge
        if dst in IMMUTABLE_VARS:
            dag_edges.remove(edge)
            removed_immutable.append(edge)
    
    if removed_immutable:
        print(f"✗ Removed {len(removed_immutable)} edges to immutable vars")
    else:
        print(f"✓ No edges to immutable variables")
    
    # Final validation
    final_edges = list(dag_edges)
    G_final = nx.DiGraph(final_edges)
    
    print(f"\n✅ FINAL SMART DAG")
    print("-" * 80)
    print(f"Total edges:           {len(final_edges)}")
    print(f"Total nodes:           {G_final.number_of_nodes()}")
    print(f"Is valid DAG:          {nx.is_directed_acyclic_graph(G_final)}")
    print(f"Average degree:        {sum(dict(G_final.degree()).values()) / G_final.number_of_nodes():.2f}")
    
    if "cardio" in G_final:
        cvd_parents = sorted(G_final.predecessors("cardio"))
        print(f"CVD direct causes:     {len(cvd_parents)}")
        print(f"  Parents: {', '.join(cvd_parents)}")
    
    # Composition breakdown
    print(f"\n📊 EDGE COMPOSITION")
    print("-" * 80)
    print(f"From {best_algo['name']} (base):     {len(best_algo['edges']) - len(removed)}")
    print(f"From consensus:          {len(added_consensus)}")
    print(f"From medical knowledge:  {len(added_medical)}")
    print(f"Removed (validation):    {len(removed) + len(removed_immutable)}")
    
    return final_edges, {
        "base_algorithm": best_algo["name"],
        "base_score": best_algo["score"],
        "consensus_added": len(added_consensus),
        "medical_added": len(added_medical),
        "removed": len(removed) + len(removed_immutable),
    }


def compare_with_hybrid():
    """Compare smart DAG with original hybrid"""
    print("\n" + "=" * 80)
    print("COMPARISON: SMART DAG vs HYBRID DAG")
    print("=" * 80)
    
    smart_edges = load_edges("edges_smart_dag.csv")
    hybrid_edges = load_edges("edges_hybrid.csv")
    
    smart_set = set(smart_edges)
    hybrid_set = set(hybrid_edges)
    
    only_smart = smart_set - hybrid_set
    only_hybrid = hybrid_set - smart_set
    common = smart_set & hybrid_set
    
    print(f"\nEdge comparison:")
    print(f"  In both:            {len(common)}")
    print(f"  Only in Smart DAG:  {len(only_smart)}")
    print(f"  Only in Hybrid DAG: {len(only_hybrid)}")
    
    if only_smart:
        print(f"\n  Smart DAG additions (data-driven):")
        for e in list(only_smart)[:10]:
            print(f"    + {e[0]} → {e[1]}")
    
    if only_hybrid:
        print(f"\n  Hybrid DAG additions (medical knowledge):")
        for e in list(only_hybrid)[:10]:
            print(f"    + {e[0]} → {e[1]}")
    
    # CVD parents comparison
    G_smart = nx.DiGraph(smart_edges)
    G_hybrid = nx.DiGraph(hybrid_edges)
    
    smart_parents = set(G_smart.predecessors("cardio")) if "cardio" in G_smart else set()
    hybrid_parents = set(G_hybrid.predecessors("cardio")) if "cardio" in G_hybrid else set()
    
    print(f"\n  CVD direct causes:")
    print(f"    Smart DAG:  {', '.join(sorted(smart_parents))}")
    print(f"    Hybrid DAG: {', '.join(sorted(hybrid_parents))}")
    print(f"    Agreement:  {len(smart_parents & hybrid_parents)}/{max(len(smart_parents), len(hybrid_parents))}")


def main():
    print("\n" + "=" * 80)
    print("BUILDING BEST DAG USING SMART STRATEGY")
    print("Strategy: Best Algorithm + Consensus + Medical Validation")
    print("=" * 80)
    
    # Build smart DAG
    smart_edges, metadata = build_smart_dag()
    
    # Save
    output_path = OUT_DIR / "edges_smart_dag.csv"
    df = pd.DataFrame(smart_edges, columns=["source", "target"])
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved Smart DAG: {output_path}")
    
    # Compare with hybrid
    compare_with_hybrid()
    
    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\n🎯 Use the SMART DAG (edges_smart_dag.csv) because:")
    print(f"   1. Based on best-performing algorithm ({metadata['base_algorithm']})")
    print(f"   2. Enhanced with {metadata['consensus_added']} consensus edges")
    print(f"   3. Validated with {metadata['medical_added']} medical edges")
    print(f"   4. Removed {metadata['removed']} invalid/forbidden edges")
    print(f"   5. Purely data-driven with medical safety checks")
    
    print(f"\n📊 To visualize the Smart DAG:")
    print(f"   python src/visualize_dag.py")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()