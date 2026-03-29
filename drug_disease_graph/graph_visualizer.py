import plotly.graph_objects as go
import networkx as nx

def visualize_graph_plotly(G: nx.DiGraph):
    """Visualizes the NetworkX graph interactively using Plotly."""
    if len(G.nodes) == 0:
        return None
        
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        node_type = G.nodes[node].get('type', 'Other')
        if node_type == 'Disease':
            node_color.append('salmon')
        else:
            node_color.append('lightblue')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=node_text,
        mode='markers+text',
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=20,
            line_width=2))
            
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='',
                title_font_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    x=0.5, y=-0.05,
                    showarrow=False,
                    text="Salmon = Disease | Light Blue = Drug (Edges represent 'treated_by')",
                    xref="paper", yref="paper"
                ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig
