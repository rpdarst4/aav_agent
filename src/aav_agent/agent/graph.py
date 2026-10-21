"""Wire the nodes into a LangGraph graph: llm -> (tools -> llm)* -> finalize."""

from __future__ import annotations

from types import SimpleNamespace

from .nodes import AgentState, Nodes


def build_graph(nodes: Nodes, api=None):
    """`api` (StateGraph, START, END) is injectable so the wiring can be tested without LangGraph."""
    if api is None:
        from langgraph.graph import END, START, StateGraph

        api = SimpleNamespace(StateGraph=StateGraph, START=START, END=END)
    g = api.StateGraph(AgentState)
    g.add_node("llm", nodes.llm)
    g.add_node("tools", nodes.tools)
    g.add_node("finalize", nodes.finalize)
    g.add_edge(api.START, "llm")
    g.add_conditional_edges("llm", nodes.route, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "llm")
    g.add_edge("finalize", api.END)
    return g.compile()
