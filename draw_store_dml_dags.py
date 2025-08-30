
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["figure.dpi"] = 160

def draw_dag(nodes, edges, pos, title, outfile):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Create labels with line breaks for multi-word labels
    labels = {}
    for node in nodes:
        if ' ' in node:
            words = node.split(' ')
            if len(words) > 1:
                mid = len(words) // 2
                labels[node] = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            else:
                labels[node] = node
        else:
            labels[node] = node
    node_size = 6000
    fig = plt.figure(figsize=(11, 8.5))
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='fancy', arrowsize=20, width=2, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_shape='s', node_size=node_size)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=14)
    plt.axis('off')
    plt.title(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

# Basic
nodes_basic = [
    "Income", "Urbanicity", "Competitor Presence", "Brand Awareness",
    "Store Opening (T)", "Loyalty Signup (Y)"
]
pos_basic = {
    "Income": (-2.5,  1.0),
    "Urbanicity": (-2.5,  0.2),
    "Competitor Presence": (-2.5, -0.6),
    "Brand Awareness": (-2.5, -1.4),
    "Store Opening (T)": (-0.2, -0.2),
    "Loyalty Signup (Y)": (2.0, -0.2),
}
edges_basic = [
    ("Income", "Store Opening (T)"),
    ("Urbanicity", "Store Opening (T)"),
    ("Competitor Presence", "Store Opening (T)"),
    ("Brand Awareness", "Store Opening (T)"),
    ("Income", "Loyalty Signup (Y)"),
    ("Urbanicity", "Loyalty Signup (Y)"),
    ("Competitor Presence", "Loyalty Signup (Y)"),
    ("Brand Awareness", "Loyalty Signup (Y)"),
    ("Store Opening (T)", "Loyalty Signup (Y)"),
]
draw_dag(nodes_basic, edges_basic, pos_basic,
         "Basic Confounded Model — Adjust {Income, Urbanicity, Competitor, Brand Awareness}",
         "dag_basic.png")

# Mediators
nodes_med = nodes_basic + ["Foot Traffic", "In-Store Experience", "Local Buzz / PR"]
pos_med = pos_basic | {
    "Foot Traffic": (0.9,  0.6),
    "In-Store Experience": (0.9, -0.2),
    "Local Buzz / PR": (0.9, -1.0),
}
edges_med = edges_basic + [
    ("Store Opening (T)", "Foot Traffic"),
    ("Foot Traffic", "Loyalty Signup (Y)"),
    ("Store Opening (T)", "In-Store Experience"),
    ("In-Store Experience", "Loyalty Signup (Y)"),
    ("Store Opening (T)", "Local Buzz / PR"),
    ("Local Buzz / PR", "Loyalty Signup (Y)"),
]
draw_dag(nodes_med, edges_med, pos_med,
         "With Mediators — Do NOT adjust for mediators when estimating the total effect",
         "dag_mediators.png")

# Instrument & Collider
nodes_ic = nodes_basic + ["Zoning Shock / Lease Opportunity", "Regional KPI (post-T)"]
pos_ic = pos_basic | {
    "Zoning Shock / Lease Opportunity": (-0.2, 1.0),
    "Regional KPI (post-T)": (1.1, -1.0),
}
edges_ic = edges_basic + [
    ("Zoning Shock / Lease Opportunity", "Store Opening (T)"),
    ("Store Opening (T)", "Regional KPI (post-T)"),
    ("Competitor Presence", "Regional KPI (post-T)"),
]
draw_dag(nodes_ic, edges_ic, pos_ic,
         "Instrument & Collider",
         "dag_instrument_collider.png")
