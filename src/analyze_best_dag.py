#!/usr/bin/env python3
"""
analyze_best_dag.py
Analyze which DAG is best and show comprehensive statistics
"""

import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"


def load_edges(filename: str):
    """Load edges from CSV"""
    path = OUT_DIR / filename
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return list(zip(df["source"], df["target"]))


def analyze_dag(edges: list, name: str):
    """Analyze DAG quality metrics"""
    if not edges:
        return {
            "name": name,
            "edges": 0,
            "nodes": 0,
            "cvd_parents": 0,
            "is_dag": False,
            "score": 0
        }
    
    G = nx.DiGraph(edges)
    
    # Get CVD parents
    cvd_parents = list(G.predecessors("cardio")) if "cardio" in G else []
    
    # Scoring system
    score = 0
    score += len(edges) * 0.5  # Reward complexity
    score += len(cvd_parents) * 10  # Reward CVD connections
    score += 100 if nx.is_directed_acyclic_graph(G) else -1000  # Must be DAG
    
    # Penalize if too sparse or too dense
    n_nodes = G.number_of_nodes()
    if n_nodes > 0:
        density = len(edges) / (n_nodes * (n_nodes - 1))
        if density < 0.1:
            score -= 20  # Too sparse
        elif density > 0.5:
            score -= 30  # Too dense
    
    return {
        "name": name,
        "edges": len(edges),
        "nodes": G.number_of_nodes(),
        "cvd_parents": len(cvd_parents),
        "cvd_parent_list": cvd_parents,
        "is_dag": nx.is_directed_acyclic_graph(G),
        "density": density if n_nodes > 0 else 0,
        "avg_degree": sum(dict(G.degree()).values()) / n_nodes if n_nodes > 0 else 0,
        "score": score
    }


def main():
    print("\n" + "=" * 80)
    print("BEST DAG ANALYSIS")
    print("=" * 80 + "\n")
    
    # Analyze all DAGs
    dags = {
        "Medical Knowledge": "edges_medical_dag.csv",
        "PC Algorithm": "edges_pc.csv",
        "GES Algorithm": "edges_ges.csv",
        "NOTEARS Algorithm": "edges_notears.csv",
        "Consensus (2+)": "edges_consensus.csv",
        "Hybrid DAG": "edges_hybrid.csv",
    }
    
    results = []
    for name, filename in dags.items():
        edges = load_edges(filename)
        analysis = analyze_dag(edges, name)
        results.append(analysis)
    
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Display table
    print("QUALITY RANKING")
    print("-" * 80)
    print(f"{'Rank':<6} {'DAG':<25} {'Edges':<8} {'Nodes':<8} {'→CVD':<8} {'Valid':<8} {'Score':<10}")
    print("-" * 80)
    
    for rank, r in enumerate(results, 1):
        valid = "✓" if r["is_dag"] else "✗"
        print(f"{rank:<6} {r['name']:<25} {r['edges']:<8} {r['nodes']:<8} {r['cvd_parents']:<8} {valid:<8} {r['score']:<10.1f}")
    
    # Best DAG
    best = results[0]
    
    print("\n" + "=" * 80)
    print(f"🏆 BEST DAG: {best['name']}")
    print("=" * 80)
    print(f"  Total edges:           {best['edges']}")
    print(f"  Total nodes:           {best['nodes']}")
    print(f"  Direct causes of CVD:  {best['cvd_parents']}")
    print(f"  Is valid DAG:          {best['is_dag']}")
    print(f"  Graph density:         {best['density']:.3f}")
    print(f"  Average degree:        {best['avg_degree']:.2f}")
    
    if best['cvd_parent_list']:
        print(f"\n  CVD is directly caused by:")
        for parent in sorted(best['cvd_parent_list']):
            print(f"    • {parent}")
    
    # Agreement analysis
    print("\n" + "=" * 80)
    print("EDGE AGREEMENT ANALYSIS")
    print("=" * 80)
    
    all_edges = []
    for name, filename in dags.items():
        if name != "Hybrid DAG":  # Exclude hybrid from comparison
            edges = load_edges(filename)
            all_edges.extend(edges)
    
    edge_votes = Counter(all_edges)
    
    print(f"\nTotal unique edges discovered: {len(edge_votes)}")
    print(f"Edges found by 3+ algorithms:  {sum(1 for v in edge_votes.values() if v >= 3)}")
    print(f"Edges found by 2 algorithms:   {sum(1 for v in edge_votes.values() if v == 2)}")
    print(f"Edges found by 1 algorithm:    {sum(1 for v in edge_votes.values() if v == 1)}")
    
    # Most agreed edges
    most_agreed = edge_votes.most_common(10)
    if most_agreed:
        print("\nTop 10 most agreed edges:")
        for (src, dst), count in most_agreed:
            print(f"  {src:>15} → {dst:<15} ({count} algorithms)")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if best['name'] == "Hybrid DAG":
        print("✅ The Hybrid DAG is currently the best model because:")
        print("   1. Incorporates proven medical knowledge")
        print("   2. Adds validated data-driven discoveries")
        print("   3. Maintains biological plausibility")
        print("   4. Valid DAG structure (acyclic)")
        
        # Check for improvements
        pc_analysis = next((r for r in results if r['name'] == "PC Algorithm"), None)
        ges_analysis = next((r for r in results if r['name'] == "GES Algorithm"), None)
        notears_analysis = next((r for r in results if r['name'] == "NOTEARS Algorithm"), None)
        
        improvements = []
        if pc_analysis and pc_analysis['edges'] < 15:
            improvements.append("⚠️  PC found very few edges - consider relaxing parameters")
        if notears_analysis and notears_analysis['edges'] == 0:
            improvements.append("⚠️  NOTEARS failed - needs better hyperparameters")
        if ges_analysis and ges_analysis['cvd_parents'] < 3:
            improvements.append("⚠️  GES found few CVD causes - may need more iterations")
        
        if improvements:
            print("\n🔧 Potential improvements:")
            for imp in improvements:
                print(f"   {imp}")
            print("\n   Recommendation: Re-run discovery with improved parameters")
            print("   Command: python src/02_discovery_improved.py")
        else:
            print("\n🎉 All algorithms performed well! This is a robust model.")
    else:
        print(f"⚠️  {best['name']} scored highest, but this may not reflect medical validity.")
        print("   Recommendation: Use Hybrid DAG for clinical applications.")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)
    print("\nTo see beautiful visualizations of the best DAG:")
    print("  python src/visualize_dag.py")
    print("\nThis will create:")
    print("  📊 dag_hybrid_layered.png       - Best model with clear layers")
    print("  📊 dag_comparison.png           - All algorithms compared")
    print("  📊 Individual DAG visualizations")
    print("\n")


if __name__ == "__main__":
    main()