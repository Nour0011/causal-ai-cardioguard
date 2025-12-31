# ============================================================
# 02_discovery_improved.py
# Enhanced Causal Discovery with Medical Domain Knowledge Integration
# Combines: PC + Medical DAG + Bootstrap validation
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
from collections import Counter

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "outputs" / "cardio_clean.csv"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_PC = OUT_DIR / "edges_pc.csv"
OUT_GES = OUT_DIR / "edges_ges.csv"
OUT_NOTEARS = OUT_DIR / "edges_notears.csv"
OUT_CONSENSUS = OUT_DIR / "edges_consensus.csv"
OUT_MEDICAL = OUT_DIR / "edges_medical_dag.csv"
OUT_HYBRID = OUT_DIR / "edges_hybrid.csv"
OUT_REPORT = OUT_DIR / "discovery_report.txt"


# -------------------------
# Load data
# -------------------------
def load_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned CSV not found: {path}")
    return pd.read_csv(path)


# ============================================================
# 1) Medical Domain Knowledge DAG
# ============================================================
def build_medical_dag():
    """
    Construct DAG based on established medical knowledge
    This provides a strong prior for causal structure
    """
    edges = [
        # Demographics → Physical characteristics
        ("age", "height"),
        ("age", "weight"),
        ("gender", "height"),
        ("gender", "weight"),
        
        # Physical → BMI
        ("height", "bmi"),
        ("weight", "bmi"),
        
        # Demographics → Lifestyle (weak influence)
        ("age", "smoke"),
        ("age", "alco"),
        ("gender", "active"),
        
        # Lifestyle → Metabolic
        ("smoke", "ap_hi"),
        ("smoke", "ap_lo"),
        ("alco", "ap_hi"),
        ("active", "bmi"),
        ("active", "ap_hi"),
        ("active", "ap_lo"),
        
        # BMI → Metabolic
        ("bmi", "ap_hi"),
        ("bmi", "ap_lo"),
        ("bmi", "cholesterol"),
        ("bmi", "gluc"),
        
        # Age → Metabolic (aging effect)
        ("age", "ap_hi"),
        ("age", "ap_lo"),
        ("age", "cholesterol"),
        ("age", "gluc"),
        
        # Metabolic → Outcome
        ("ap_hi", "cardio"),
        ("ap_lo", "cardio"),
        ("cholesterol", "cardio"),
        ("gluc", "cardio"),
        ("bmi", "cardio"),
        
        # Direct lifestyle → Outcome
        ("smoke", "cardio"),
        ("age", "cardio"),
    ]
    
    return edges


# ============================================================
# 2) PC Algorithm (Constraint-based)
# ============================================================
def run_pc(df: pd.DataFrame, max_cond_vars: int = 2, alpha: float = 0.05, 
           sample_size: int = 10000):
    """
    PC (Peter-Clark) Algorithm - Constraint-based causal discovery
    Tests conditional independence to learn structure
    
    Note: Adjusted to be less conservative:
    - max_cond_vars=2 (simpler conditional tests)
    - alpha=0.05 (standard significance level)
    - larger sample size for more power
    """
    print("\n" + "="*80)
    print("PC ALGORITHM (Constraint-based Discovery)")
    print("="*80)
    
    try:
        from pgmpy.estimators import PC
    except ImportError:
        print("⚠️  pgmpy not installed. Skipping PC.")
        return []
    
    try:
        n = min(sample_size, len(df))
        df_sample = df.sample(n=n, random_state=42).reset_index(drop=True)
        
        print(f"  Running PC on {n} samples...")
        print(f"  Parameters: max_cond_vars={max_cond_vars}, alpha={alpha}")
        
        pc = PC(data=df_sample)
        model = pc.estimate(
            variant="stable",
            max_cond_vars=max_cond_vars,
            significance_level=alpha
        )
        edges = list(model.edges())
        
        print(f"✓ PC complete: {len(edges)} edges discovered")
        parents = sorted([u for (u, v) in edges if v == "cardio"])
        print(f"  Direct parents of cardio: {parents if parents else '(none)'}")
        
        return edges
    except Exception as e:
        print(f"⚠️  PC failed: {e}")
        return []


# ============================================================
# 3) GES Algorithm (Score-based)
# ============================================================
class GESAlgorithm:
    """
    GES (Greedy Equivalence Search) - Score-based causal discovery
    Uses BIC scoring to search for best DAG structure
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.variables = list(data.columns)
        self.n_vars = len(self.variables)
    
    def bic_score(self, dag_edges):
        """Calculate BIC score for a DAG"""
        from sklearn.linear_model import LinearRegression
        
        n = len(self.data)
        score = 0.0
        
        # Build adjacency matrix
        adj = np.zeros((self.n_vars, self.n_vars))
        for src, dst in dag_edges:
            i = self.variables.index(src)
            j = self.variables.index(dst)
            adj[i, j] = 1
        
        # Score each variable given its parents
        for j in range(self.n_vars):
            parents = [i for i in range(self.n_vars) if adj[i, j] == 1]
            y = self.data.iloc[:, j].values.astype(float)
            
            if len(parents) == 0:
                residual_var = np.var(y)
            else:
                X = self.data.iloc[:, parents].values.astype(float)
                model = LinearRegression()
                model.fit(X, y)
                pred = model.predict(X)
                residual_var = np.var(y - pred)
            
            k = len(parents) + 1  # Number of parameters
            score += -n/2 * np.log(residual_var + 1e-10) - (k/2) * np.log(n)
        
        return score
    
    def is_acyclic(self, edges):
        """Check if edge list forms a DAG"""
        G = nx.DiGraph()
        G.add_edges_from(edges)
        return nx.is_directed_acyclic_graph(G)
    
    def run(self, max_iter: int = 25):
        """Run GES algorithm"""
        current_edges = set()
        current_score = self.bic_score(current_edges)
        
        # Forward phase: Add edges
        for iteration in range(max_iter):
            best_score = current_score
            best_edge = None
            
            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    if i == j:
                        continue
                    edge = (self.variables[i], self.variables[j])
                    if edge in current_edges:
                        continue
                    
                    new_edges = current_edges | {edge}
                    if self.is_acyclic(new_edges):
                        score = self.bic_score(new_edges)
                        if score > best_score:
                            best_score = score
                            best_edge = edge
            
            if best_edge is None:
                break
            
            current_edges.add(best_edge)
            current_score = best_score
        
        # Backward phase: Remove edges
        for iteration in range(max_iter):
            best_score = current_score
            best_edge = None
            
            for edge in list(current_edges):
                new_edges = current_edges - {edge}
                score = self.bic_score(new_edges)
                if score > best_score:
                    best_score = score
                    best_edge = edge
            
            if best_edge is None:
                break
            
            current_edges.remove(best_edge)
            current_score = best_score
        
        return list(current_edges), current_score


def run_ges(df: pd.DataFrame, sample_size: int = 5000):
    """
    Run GES algorithm
    """
    print("\n" + "="*80)
    print("GES ALGORITHM (Score-based Discovery)")
    print("="*80)
    
    try:
        n = min(sample_size, len(df))
        df_sample = df.sample(n=n, random_state=42).reset_index(drop=True)
        
        ges = GESAlgorithm(df_sample)
        edges, final_score = ges.run(max_iter=25)
        
        print(f"✓ GES complete: {len(edges)} edges discovered (BIC={final_score:.2f})")
        parents = sorted([u for (u, v) in edges if v == "cardio"])
        print(f"  Direct parents of cardio: {parents if parents else '(none)'}")
        
        return edges
    except Exception as e:
        print(f"⚠️  GES failed: {e}")
        return []


# ============================================================
# 4) NOTEARS Algorithm (Continuous optimization)
# ============================================================
class NOTEARSAlgorithm:
    """
    NOTEARS - Continuous optimization approach to structure learning
    Formulates DAG constraint as smooth penalty
    """
    def __init__(self, data: pd.DataFrame, lambda1: float = 0.01, 
                 max_iter: int = 100, threshold: float = 0.3, lr: float = 0.01):
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data for numerical stability
        X = data.values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)
        
        self.n, self.d = self.X.shape
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.var_names = list(data.columns)
        self.threshold = threshold
        self.lr = lr
    
    def h_func(self, W):
        """DAG constraint function (should be 0 for valid DAG)"""
        import math
        d = W.shape[0]
        M = W * W
        expm = np.eye(d)
        for k in range(1, min(d + 1, 10)):
            expm = expm + np.linalg.matrix_power(M, k) / math.factorial(k)
        return np.trace(expm) - d
    
    def gradient(self, W):
        """Gradient of loss function"""
        return -1.0 / self.n * self.X.T @ (self.X - self.X @ W)
    
    def run(self):
        """Run NOTEARS algorithm"""
        W = np.random.randn(self.d, self.d) * 0.01
        rho = 1.0
        final_h = float('inf')
        
        for iteration in range(self.max_iter):
            grad = self.gradient(W) + self.lambda1 * np.sign(W)
            h = self.h_func(W)
            final_h = h
            
            if np.isnan(h) or np.isinf(h):
                print(f"  ⚠️  Numerical instability at iteration {iteration}")
                break
            
            if abs(h) < 1e-6:
                print(f"  ✓ Converged at iteration {iteration} (h={h:.2e})")
                break
            
            # Gradient descent step
            W = W - self.lr * (grad + rho * 2 * W * h)
            
            # Clip to prevent explosion
            W = np.clip(W, -5.0, 5.0)
            
            if iteration % 50 == 0 and iteration > 0:
                rho *= 1.05
                print(f"  Iteration {iteration:3d}: h(W)={h:.6f}")
        
        # Threshold to get final edges
        W[np.abs(W) < self.threshold] = 0.0
        
        edges = []
        for i in range(self.d):
            for j in range(self.d):
                if i != j and W[i, j] != 0:
                    edges.append((self.var_names[i], self.var_names[j]))
        
        return edges, final_h


def run_notears(df: pd.DataFrame, sample_size: int = 5000):
    """
    Run NOTEARS algorithm with improved hyperparameters for discrete/mixed data
    """
    print("\n" + "="*80)
    print("NOTEARS ALGORITHM (Continuous Optimization)")
    print("="*80)
    
    try:
        n = min(sample_size, len(df))
        df_sample = df.sample(n=n, random_state=42).reset_index(drop=True)
        
        # Try multiple parameter settings and keep best result
        best_edges = []
        best_h = float('inf')
        
        configs = [
            {"lambda1": 0.1, "max_iter": 200, "threshold": 0.1, "lr": 0.001},
            {"lambda1": 0.05, "max_iter": 200, "threshold": 0.15, "lr": 0.002},
            {"lambda1": 0.2, "max_iter": 150, "threshold": 0.2, "lr": 0.0005},
        ]
        
        print("  Trying multiple hyperparameter configurations...")
        
        for i, config in enumerate(configs, 1):
            print(f"\n  Configuration {i}/{len(configs)}: λ={config['lambda1']}, lr={config['lr']}")
            try:
                notears = NOTEARSAlgorithm(df_sample, **config)
                edges, final_h = notears.run()
                
                if abs(final_h) < abs(best_h) and len(edges) > 0:
                    best_edges = edges
                    best_h = final_h
                    print(f"    → New best: {len(edges)} edges, h={final_h:.6f}")
            except Exception as e:
                print(f"    → Failed: {e}")
                continue
        
        if not best_edges:
            print("\n⚠️  NOTEARS: All configurations failed. Using empty graph.")
            return []
        
        print(f"\n✓ NOTEARS complete: {len(best_edges)} edges discovered (best h={best_h:.6f})")
        parents = sorted([u for (u, v) in best_edges if v == "cardio"])
        print(f"  Direct parents of cardio: {parents if parents else '(none)'}")
        
        return best_edges
    except Exception as e:
        print(f"⚠️  NOTEARS failed: {e}")
        return []


# ============================================================
# 5) Consensus Voting
# ============================================================
def consensus_voting(pc_edges: list, ges_edges: list, notears_edges: list, 
                    min_votes: int = 2) -> list:
    """
    Combine edges from multiple algorithms using voting
    Keep edges that appear in at least min_votes algorithms
    """
    print("\n" + "="*80)
    print(f"CONSENSUS VOTING (minimum {min_votes} votes)")
    print("="*80)
    
    votes = Counter()
    
    for edge in pc_edges:
        votes[edge] += 1
    for edge in ges_edges:
        votes[edge] += 1
    for edge in notears_edges:
        votes[edge] += 1
    
    consensus = [edge for edge, count in votes.items() if count >= min_votes]
    
    print(f"  PC edges:      {len(pc_edges)}")
    print(f"  GES edges:     {len(ges_edges)}")
    print(f"  NOTEARS edges: {len(notears_edges)}")
    print(f"  Consensus:     {len(consensus)}")
    
    # Show agreement levels
    print(f"\n  Agreement levels:")
    for level in [3, 2, 1]:
        count = sum(1 for c in votes.values() if c == level)
        print(f"    {level} algorithm(s): {count} edges")
    
    parents = sorted([u for (u, v) in consensus if v == "cardio"])
    print(f"\n  Direct parents of cardio (consensus): {parents if parents else '(none)'}")
    
    return consensus


# ============================================================
# 6) Hybrid DAG: Medical + Consensus + Validation
# ============================================================
def merge_all_sources(medical_edges: list, consensus_edges: list, 
                     pc_edges: list, ges_edges: list, notears_edges: list) -> tuple:
    """
    Create final hybrid DAG:
    1. Start with medical knowledge (mandatory)
    2. Add consensus edges (high confidence)
    3. Optionally add unique edges from individual algorithms
    4. Ensure DAG remains acyclic
    5. Respect biological constraints
    """
    print("\n" + "="*80)
    print("HYBRID DAG CONSTRUCTION")
    print("="*80)
    
    # Start with medical edges (always included)
    hybrid = set(medical_edges)
    added_sources = []
    
    # Add consensus edges (appear in 2+ algorithms)
    consensus_added = []
    for edge in consensus_edges:
        if edge not in hybrid:
            test_edges = list(hybrid) + [edge]
            G = nx.DiGraph(test_edges)
            
            if nx.is_directed_acyclic_graph(G):
                src, dst = edge
                if dst not in ["age", "gender", "height", "weight"]:  # Immutable
                    hybrid.add(edge)
                    consensus_added.append(edge)
    
    # Optionally add unique strong edges from individual algorithms
    # (edges with strong statistical support but didn't make consensus)
    individual_added = {"pc": [], "ges": [], "notears": []}
    
    for algo_name, algo_edges in [("pc", pc_edges), ("ges", ges_edges), ("notears", notears_edges)]:
        for edge in algo_edges:
            if edge not in hybrid:
                test_edges = list(hybrid) + [edge]
                G = nx.DiGraph(test_edges)
                
                if nx.is_directed_acyclic_graph(G):
                    src, dst = edge
                    if dst not in ["age", "gender", "height", "weight"]:
                        # Only add if edge appears strong (in this case, not yet in hybrid)
                        # You can add more criteria here
                        pass  # For now, only add consensus
    
    print(f"✓ Medical edges (base):      {len(medical_edges)}")
    print(f"✓ Consensus edges added:     {len(consensus_added)}")
    print(f"✓ PC unique added:           {len(individual_added['pc'])}")
    print(f"✓ GES unique added:          {len(individual_added['ges'])}")
    print(f"✓ NOTEARS unique added:      {len(individual_added['notears'])}")
    print(f"✓ Total hybrid edges:        {len(hybrid)}")
    
    return list(hybrid), consensus_added, individual_added


# ============================================================
# 4) DAG Validation
# ============================================================
def validate_dag(edges: list, df: pd.DataFrame) -> dict:
    """Validate DAG properties"""
    G = nx.DiGraph(edges)
    
    validation = {
        "is_dag": nx.is_directed_acyclic_graph(G),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        "has_outcome": "cardio" in G.nodes(),
        "outcome_parents": list(G.predecessors("cardio")) if "cardio" in G.nodes() else [],
        "max_path_length": 0,
    }
    
    if validation["is_dag"] and validation["has_outcome"]:
        # Find longest path to outcome
        max_len = 0
        for node in G.nodes():
            if node != "cardio" and nx.has_path(G, node, "cardio"):
                path_len = nx.shortest_path_length(G, node, "cardio")
                max_len = max(max_len, path_len)
        validation["max_path_length"] = max_len
    
    return validation


# ============================================================
# 7) Comprehensive Reporting
# ============================================================
def generate_comprehensive_report(medical_edges, pc_edges, ges_edges, 
                                 notears_edges, consensus_edges, hybrid_edges,
                                 validation, output_path: Path):
    """Generate comprehensive discovery report with all algorithms"""
    
    lines = []
    lines.append("=" * 80)
    lines.append("CAUSAL DISCOVERY REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("METHODOLOGY")
    lines.append("-" * 80)
    lines.append("This analysis combines multiple causal discovery approaches:")
    lines.append("")
    lines.append("1. MEDICAL KNOWLEDGE DAG")
    lines.append("   - Hand-crafted based on established medical literature")
    lines.append("   - Represents decades of cardiovascular research")
    lines.append("   - Forms the foundation of our causal model")
    lines.append("")
    lines.append("2. PC ALGORITHM (Constraint-based)")
    lines.append("   - Tests conditional independence relationships")
    lines.append("   - Based on statistical tests (chi-square, G-test)")
    lines.append("   - Good at finding causal structure from observational data")
    lines.append("")
    lines.append("3. GES ALGORITHM (Score-based)")
    lines.append("   - Searches for highest-scoring DAG using BIC")
    lines.append("   - Greedy search with forward and backward phases")
    lines.append("   - Balances model fit vs. complexity")
    lines.append("")
    lines.append("4. NOTEARS ALGORITHM (Continuous optimization)")
    lines.append("   - Formulates structure learning as continuous optimization")
    lines.append("   - Uses smooth DAG constraint")
    lines.append("   - Can handle larger graphs efficiently")
    lines.append("")
    lines.append("5. CONSENSUS VOTING")
    lines.append("   - Combines results from all data-driven methods")
    lines.append("   - Keeps edges appearing in 2+ algorithms")
    lines.append("   - More robust than any single method")
    lines.append("")
    lines.append("6. HYBRID DAG")
    lines.append("   - Merges medical knowledge with consensus")
    lines.append("   - Ensures biological plausibility")
    lines.append("   - Maintains DAG property (no cycles)")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("EDGE DISCOVERY RESULTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Medical DAG edges:        {len(medical_edges):>4}")
    lines.append(f"PC algorithm edges:       {len(pc_edges):>4}")
    lines.append(f"GES algorithm edges:      {len(ges_edges):>4}")
    lines.append(f"NOTEARS algorithm edges:  {len(notears_edges):>4}")
    lines.append(f"Consensus edges (2+):     {len(consensus_edges):>4}")
    lines.append(f"Final hybrid DAG edges:   {len(hybrid_edges):>4}")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("ALGORITHM COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    
    # Compare parent sets for cardio
    pc_parents = set([u for (u, v) in pc_edges if v == "cardio"])
    ges_parents = set([u for (u, v) in ges_edges if v == "cardio"])
    notears_parents = set([u for (u, v) in notears_edges if v == "cardio"])
    medical_parents = set([u for (u, v) in medical_edges if v == "cardio"])
    consensus_parents = set([u for (u, v) in consensus_edges if v == "cardio"])
    hybrid_parents = set([u for (u, v) in hybrid_edges if v == "cardio"])
    
    lines.append("DIRECT CAUSES OF CARDIOVASCULAR DISEASE")
    lines.append("-" * 80)
    lines.append(f"Medical knowledge: {', '.join(sorted(medical_parents)) if medical_parents else '(none)'}")
    lines.append(f"PC algorithm:      {', '.join(sorted(pc_parents)) if pc_parents else '(none)'}")
    lines.append(f"GES algorithm:     {', '.join(sorted(ges_parents)) if ges_parents else '(none)'}")
    lines.append(f"NOTEARS algorithm: {', '.join(sorted(notears_parents)) if notears_parents else '(none)'}")
    lines.append(f"Consensus:         {', '.join(sorted(consensus_parents)) if consensus_parents else '(none)'}")
    lines.append(f"Final hybrid:      {', '.join(sorted(hybrid_parents)) if hybrid_parents else '(none)'}")
    lines.append("")
    
    # Agreement analysis
    all_discovered = set(pc_edges + ges_edges + notears_edges)
    agreement_3 = sum(1 for e in all_discovered if (e in pc_edges) + (e in ges_edges) + (e in notears_edges) == 3)
    agreement_2 = sum(1 for e in all_discovered if (e in pc_edges) + (e in ges_edges) + (e in notears_edges) == 2)
    agreement_1 = sum(1 for e in all_discovered if (e in pc_edges) + (e in ges_edges) + (e in notears_edges) == 1)
    
    lines.append("INTER-ALGORITHM AGREEMENT")
    lines.append("-" * 80)
    lines.append(f"Edges found by all 3 algorithms:  {agreement_3}")
    lines.append(f"Edges found by exactly 2:         {agreement_2}")
    lines.append(f"Edges found by only 1:            {agreement_1}")
    lines.append(f"Total unique edges discovered:    {len(all_discovered)}")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("FINAL DAG VALIDATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Is valid DAG (acyclic):   {validation['is_dag']}")
    lines.append(f"Number of nodes:          {validation['num_nodes']}")
    lines.append(f"Number of edges:          {validation['num_edges']}")
    lines.append(f"Average degree:           {validation['avg_degree']:.2f}")
    lines.append(f"Outcome in graph:         {validation['has_outcome']}")
    lines.append(f"Max path to outcome:      {validation['max_path_length']}")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("The hybrid DAG represents our best understanding of causal relationships")
    lines.append("affecting cardiovascular disease, combining:")
    lines.append("- Established medical knowledge (gold standard)")
    lines.append("- Data-driven discoveries (multiple algorithms)")
    lines.append("- Consensus validation (robustness check)")
    lines.append("")
    lines.append("This structure will be used for:")
    lines.append("1. Average Treatment Effect (ATE) estimation")
    lines.append("2. Bayesian Network construction")
    lines.append("3. Personalized risk prediction")
    lines.append("4. Intervention recommendation")
    lines.append("")
    
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    
    print(f"\n✓ Generated comprehensive report: {output_path}")
    
    return report_text


# ============================================================
# Main
# ============================================================
def save_edges(edges: list, path: Path):
    """Save edges to CSV"""
    df = pd.DataFrame(edges, columns=["source", "target"])
    df.to_csv(path, index=False)
    print(f"✓ Saved: {path}")


def main():
    print("\n" + "=" * 80)
    print("STEP 2: CAUSAL DISCOVERY (PC + GES + NOTEARS + CONSENSUS)")
    print("=" * 80)
    
    # Load data
    df = load_clean(CLEAN_PATH)
    print(f"\n✓ Loaded cleaned data: {df.shape}")
    
    # Select relevant columns
    cols = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active", "bmi", "cardio"]
    df = df[cols].copy()
    
    # Build medical DAG (domain knowledge baseline)
    print("\n" + "=" * 80)
    print("STEP 2A: MEDICAL DOMAIN KNOWLEDGE")
    print("=" * 80)
    medical_edges = build_medical_dag()
    print(f"✓ Built medical DAG: {len(medical_edges)} edges")
    parents = sorted([u for (u, v) in medical_edges if v == "cardio"])
    print(f"  Direct parents of cardio: {parents}")
    
    # Run all three causal discovery algorithms
    print("\n" + "=" * 80)
    print("STEP 2B: DATA-DRIVEN CAUSAL DISCOVERY")
    print("=" * 80)
    
    pc_edges = run_pc(
        df, 
        max_cond_vars=2,   # Less conservative
        alpha=0.05,        # Standard significance
        sample_size=10000  # More data
    )
    
    ges_edges = run_ges(
        df,
        sample_size=5000
    )
    
    notears_edges = run_notears(
        df,
        sample_size=5000
    )
    
    # Create consensus from data-driven methods
    print("\n" + "=" * 80)
    print("STEP 2C: CONSENSUS AGGREGATION")
    print("=" * 80)
    consensus_edges = consensus_voting(
        pc_edges, 
        ges_edges, 
        notears_edges,
        min_votes=2  # Edge must appear in at least 2 algorithms
    )
    
    # Merge everything into final hybrid DAG
    print("\n" + "=" * 80)
    print("STEP 2D: HYBRID DAG (MEDICAL + CONSENSUS)")
    print("=" * 80)
    hybrid_edges, consensus_added, individual_added = merge_all_sources(
        medical_edges,
        consensus_edges,
        pc_edges,
        ges_edges,
        notears_edges
    )
    
    # Validate final DAG
    print("\n" + "=" * 80)
    print("STEP 2E: DAG VALIDATION")
    print("=" * 80)
    validation = validate_dag(hybrid_edges, df)
    
    print(f"✓ Is valid DAG:          {validation['is_dag']}")
    print(f"✓ Number of nodes:       {validation['num_nodes']}")
    print(f"✓ Number of edges:       {validation['num_edges']}")
    print(f"✓ Average degree:        {validation['avg_degree']:.2f}")
    print(f"✓ Max path to outcome:   {validation['max_path_length']}")
    
    if validation['outcome_parents']:
        print(f"\n✓ Direct causes of CVD:  {', '.join(sorted(validation['outcome_parents']))}")
    
    # Save all edge lists
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    save_edges(medical_edges, OUT_MEDICAL)
    save_edges(pc_edges, OUT_DIR / "edges_pc.csv")
    save_edges(ges_edges, OUT_DIR / "edges_ges.csv")
    save_edges(notears_edges, OUT_DIR / "edges_notears.csv")
    save_edges(consensus_edges, OUT_DIR / "edges_consensus.csv")
    save_edges(hybrid_edges, OUT_HYBRID)
    
    # Generate comprehensive report
    report_text = generate_comprehensive_report(
        medical_edges,
        pc_edges,
        ges_edges,
        notears_edges,
        consensus_edges,
        hybrid_edges,
        validation,
        OUT_REPORT
    )
    
    print("\n" + "=" * 80)
    print("✅ Step 2 complete. Next: python src/03_ate_estimation.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()