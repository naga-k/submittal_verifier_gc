from langgraph.graph import StateGraph, END
from graph.schemas import RunState
from graph.nodes import classify_node, extract_checklist_node, verify_requirements_node

def build_workflow():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(RunState)
    
    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("extract_checklist", extract_checklist_node)
    graph.add_node("verify", verify_requirements_node)
    
    # Define the flow - remove the problematic conditional edge
    graph.set_entry_point("classify")
    graph.add_edge("classify", "extract_checklist")
    graph.add_edge("extract_checklist", "verify")
    graph.add_edge("verify", END)
    
    return graph.compile()