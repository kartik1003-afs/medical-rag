import networkx as nx

def build_graph(relationships: list) -> nx.DiGraph:
    """Constructs a NetworkX directed graph from disease-drug relationships."""
    G = nx.DiGraph()
    for rel in relationships:
        disease = rel.get("disease", "").strip()
        drug = rel.get("drug", "").strip()
        if disease and drug:
            G.add_node(disease, type="Disease")
            G.add_node(drug, type="Drug")
            G.add_edge(disease, drug, label="treated_by")
    return G
