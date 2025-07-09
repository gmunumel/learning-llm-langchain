import os


def store_graph(graph, graph_name, prefix=None):
    prefix = prefix or ""
    with open(f"{prefix}graphs/{graph_name}.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())


def display_graph(graph_name, prefix=None):
    prefix = prefix or ""
    os.system(f"feh {prefix}graphs/{graph_name}.png")


def show_graph(graph, graph_name, prefix=None):
    store_graph(graph, graph_name, prefix)
    display_graph(graph_name, prefix)
