#!/usr/bin/env python3
# ============================================================
# visualize_dag.py
# Beautiful Multi-layer DAG Visualization
# Shows the causal structure with clear hierarchical layers
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
from pathlib import Path

# -------------------------
# Configuration
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"

# -------------------------
# Layer Definitions (Medical Domain Knowledge)
# -------------------------
LAYERS = {
    "Demographics": {
        "nodes": ["age", "gender"],
        "color": "#FF6B6B",  # Red
        "description": "Immutable traits"
    },
    "Physical": {
        "nodes": ["height", "weight"],
        "color": "#4ECDC4",  # Teal
        "description": "Body measurements"
    },
    "Lifestyle": {
        "nodes": ["smoke", "alco", "active"],
        "color": "#95E1D3",  # Light teal
        "description": "Modifiable behaviors"
    },
    "Body Composition": {
        "nodes": ["bmi"],
        "color": "#F38181",  # Light red
        "description": "Derived metrics"
    },
    "Metabolic": {
        "nodes": ["ap_hi", "ap_lo", "cholesterol", "gluc"],
        "color": "#FFC857",  # Yellow
        "description": "Health indicators"
    },
    "Outcome": {
        "nodes": ["cardio"],
        "color": "#AA4465",  # Dark red
        "description": "Disease status"
    }
}

# Friendly names for display
FRIENDLY_NAMES = {
    "age": "Age",
    "gender": "Gender",
    "height": "Height",
    "weight": "Weight",
    "bmi": "BMI",
    "smoke": "Smoking",
    "alco": "Alcohol",
    "active": "Activity",
    "ap_hi": "Systolic BP",
    "ap_lo": "Diastolic BP",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "cardio": "CVD"
}

# -------------------------
# Load edges
# -------------------------
def load_edges(filename: str) -> list:
    """Load edges from CSV file"""
    path = OUT_DIR / filename
    if not path.exists():
        print(f"⚠️  File not found: {path}")
        return []
    
    df = pd.read_csv(path)
    edges = list(zip(df["source"], df["target"]))
    return edges


# -------------------------
# Layered Layout
# -------------------------
def compute_layered_layout(edges: list, layers: dict) -> dict:
    """
    Compute node positions in a clear layered layout
    """
    pos = {}
    
    # Layer spacing
    x_spacing = 3.0  # Horizontal space between layers
    y_spacing = 1.5  # Vertical space between nodes
    
    for layer_idx, (layer_name, layer_info) in enumerate(layers.items()):
        nodes = layer_info["nodes"]
        
        # Center nodes vertically
        n_nodes = len(nodes)
        y_start = (n_nodes - 1) * y_spacing / 2
        
        x = layer_idx * x_spacing
        
        for node_idx, node in enumerate(nodes):
            y = y_start - node_idx * y_spacing
            pos[node] = (x, y)
    
    return pos


# -------------------------
# Beautiful DAG Visualization
# -------------------------
def visualize_dag(edges: list, output_file: str, title: str = "Causal DAG"):
    """
    Create beautiful layered visualization of DAG
    """
    if not edges:
        print(f"⚠️  No edges to visualize for {title}")
        return
    
    # Create graph
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # Get all nodes (including isolated ones)
    all_nodes = set()
    for layer_info in LAYERS.values():
        all_nodes.update(layer_info["nodes"])
    G.add_nodes_from(all_nodes)
    
    # Compute layout
    pos = compute_layered_layout(edges, LAYERS)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(-0.5, len(LAYERS) * 3 - 0.5)
    ax.set_ylim(-6, 6)
    
    # Draw layer backgrounds
    for layer_idx, (layer_name, layer_info) in enumerate(LAYERS.items()):
        x = layer_idx * 3
        y_min = min([pos[n][1] for n in layer_info["nodes"] if n in pos]) - 0.5
        y_max = max([pos[n][1] for n in layer_info["nodes"] if n in pos]) + 0.5
        height = y_max - y_min
        
        # Draw semi-transparent background
        rect = FancyBboxPatch(
            (x - 0.4, y_min), 0.8, height,
            boxstyle="round,pad=0.05",
            facecolor=layer_info["color"],
            edgecolor="none",
            alpha=0.15,
            zorder=0
        )
        ax.add_patch(rect)
    
    # Draw edges first (so they're behind nodes)
    for src, dst in edges:
        if src not in pos or dst not in pos:
            continue
        
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        
        # Determine edge type based on layers
        src_layer = get_node_layer(src)
        dst_layer = get_node_layer(dst)
        
        if dst == "cardio":
            # Edges to outcome are thicker and darker
            color = "#AA4465"
            width = 2.5
            alpha = 0.8
        elif src_layer == dst_layer:
            # Within-layer edges (rare) - dashed
            color = "#999999"
            width = 1.0
            alpha = 0.5
            linestyle = "dashed"
        else:
            # Normal edges
            color = "#666666"
            width = 1.5
            alpha = 0.6
        
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='-|>',
            mutation_scale=20,
            linewidth=width,
            color=color,
            alpha=alpha,
            connectionstyle="arc3,rad=0.1",
            zorder=1
        )
        ax.add_patch(arrow)
    
    # Draw nodes
    for node in G.nodes():
        if node not in pos:
            continue
        
        x, y = pos[node]
        
        # Get node color from layer
        layer_name = get_node_layer(node)
        if layer_name:
            color = LAYERS[layer_name]["color"]
        else:
            color = "#CCCCCC"
        
        # Node size based on importance
        if node == "cardio":
            size = 0.4
            edgewidth = 3
        else:
            # Size based on degree (connectivity)
            degree = G.degree(node)
            size = 0.25 + 0.05 * min(degree, 5)
            edgewidth = 2
        
        # Draw node circle
        circle = plt.Circle(
            (x, y), size,
            facecolor=color,
            edgecolor="white",
            linewidth=edgewidth,
            zorder=2
        )
        ax.add_patch(circle)
        
        # Add label
        label = FRIENDLY_NAMES.get(node, node)
        
        # Special styling for outcome
        if node == "cardio":
            fontsize = 14
            fontweight = "bold"
        else:
            fontsize = 11
            fontweight = "normal"
        
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color="white",
            zorder=3
        )
    
    # Add layer titles at top
    y_top = 5.5
    for layer_idx, (layer_name, layer_info) in enumerate(LAYERS.items()):
        x = layer_idx * 3
        ax.text(
            x, y_top, layer_name,
            ha="center", va="bottom",
            fontsize=13,
            fontweight="bold",
            color=layer_info["color"]
        )
        ax.text(
            x, y_top - 0.4, layer_info["description"],
            ha="center", va="top",
            fontsize=9,
            style="italic",
            color="#666666"
        )
    
    # Add statistics box
    stats_text = f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}"
    if "cardio" in G:
        parents = list(G.predecessors("cardio"))
        stats_text += f"  |  CVD Direct Causes: {len(parents)}"
    
    ax.text(
        0.5, -5.5, stats_text,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CCCCCC", alpha=0.8),
        transform=ax.transData
    )
    
    # Title
    ax.text(
        len(LAYERS) * 3 / 2, 6.5, title,
        ha="center", va="bottom",
        fontsize=18,
        fontweight="bold"
    )
    
    # Clean up axes
    ax.axis("off")
    plt.tight_layout()
    
    # Save
    output_path = OUT_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved visualization: {output_path}")
    plt.close()


def get_node_layer(node: str) -> str:
    """Get the layer name for a node"""
    for layer_name, layer_info in LAYERS.items():
        if node in layer_info["nodes"]:
            return layer_name
    return None


# -------------------------
# Comparison Visualization
# -------------------------
def visualize_comparison(output_file: str = "dag_comparison.png"):
    """
    Create side-by-side comparison of different DAGs
    """
    # Load all DAGs
    dags = {
        "⭐ Final Best\nDAG": load_edges("edges_final_best.csv"),
        "Hybrid DAG\n(Medical+Data)": load_edges("edges_hybrid.csv"),
        "Smart DAG\n(GES+Valid)": load_edges("edges_smart_dag.csv"),
        "Medical\nKnowledge": load_edges("edges_medical_dag.csv"),
        "GES\nAlgorithm": load_edges("edges_ges.csv"),
        "Consensus\n(2+ votes)": load_edges("edges_consensus.csv"),
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    
    for idx, (name, edges) in enumerate(dags.items()):
        ax = axes[idx]
        
        if not edges:
            ax.text(0.5, 0.5, "No edges found", 
                   ha="center", va="center", fontsize=14, color="#999999")
            ax.set_title(name, fontsize=14, fontweight="bold", pad=10)
            ax.axis("off")
            continue
        
        # Create graph
        G = nx.DiGraph()
        G.add_edges_from(edges)
        
        # Get all nodes
        all_nodes = set()
        for layer_info in LAYERS.values():
            all_nodes.update(layer_info["nodes"])
        G.add_nodes_from(all_nodes)
        
        # Simple spring layout for comparison
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
        # Draw
        node_colors = []
        for node in G.nodes():
            layer_name = get_node_layer(node)
            if layer_name:
                node_colors.append(LAYERS[layer_name]["color"])
            else:
                node_colors.append("#CCCCCC")
        
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=800,
            edgecolors="white",
            linewidths=2
        )
        
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color="#666666",
            width=1.5,
            alpha=0.6,
            arrowsize=15,
            arrowstyle="-|>"
        )
        
        # Labels
        labels = {node: FRIENDLY_NAMES.get(node, node) for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_size=8,
            font_color="white",
            font_weight="bold"
        )
        
        # Stats
        parents = list(G.predecessors("cardio")) if "cardio" in G else []
        stats = f"{G.number_of_edges()} edges | {len(parents)} → CVD"
        
        ax.text(
            0.5, -0.15, stats,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
        
        ax.set_title(name, fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")
    
    plt.suptitle(
        "Causal Discovery Comparison: All Algorithms",
        fontsize=20, fontweight="bold", y=0.98
    )
    
    plt.tight_layout()
    
    output_path = OUT_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved comparison: {output_path}")
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    print("\n" + "=" * 80)
    print("DAG VISUALIZATION")
    print("=" * 80 + "\n")
    
    # Visualize each DAG individually with layered layout
    print("Creating detailed visualizations...\n")
    
    dags_to_visualize = [
        ("edges_final_best.csv", "dag_final_best_layered.png", "⭐ FINAL BEST DAG (Recommended)"),
        ("edges_hybrid.csv", "dag_hybrid_layered.png", "Hybrid DAG (Medical + Consensus)"),
        ("edges_smart_dag.csv", "dag_smart_layered.png", "Smart DAG (GES + Validation)"),
        ("edges_medical_dag.csv", "dag_medical_layered.png", "Medical Domain Knowledge"),
        ("edges_ges.csv", "dag_ges_layered.png", "GES Algorithm Discovery"),
        ("edges_consensus.csv", "dag_consensus_layered.png", "Consensus (2+ algorithms)"),
    ]
    
    for input_file, output_file, title in dags_to_visualize:
        edges = load_edges(input_file)
        if edges:
            visualize_dag(edges, output_file, title)
        else:
            print(f"⚠️  Skipping {output_file} (no edges)")
    
    # Create comparison visualization
    print("\nCreating comparison visualization...")
    visualize_comparison("dag_comparison.png")
    
    print("\n" + "=" * 80)
    print("✅ Visualization complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 dag_final_best_layered.png   - ⭐ BEST MODEL (use this!)")
    print("  📊 dag_hybrid_layered.png       - Alternative approach")
    print("  📊 dag_comparison.png           - All approaches compared")
    print("  📊 Other individual visualizations")
    print("\n")


if __name__ == "__main__":
    main()