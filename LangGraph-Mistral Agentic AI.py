# app.py - Agentic AI Multi-Agent System with LangGraph + Mistral
import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI  # pip install langchain-mistralai
from typing import TypedDict, Annotated, Sequence, Dict, Any
from langchain_core.tools import tool
import operator
import json

import os

# Load Mistral API key from environment variables (set in Render dashboard)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment variables. Please set it in Render dashboard.")

llm = ChatMistralAI(
    model="mistral-nemo",  # Free tier model
    api_key=MISTRAL_API_KEY,
    temperature=0.7,
    max_tokens=1000
)

# Define shared state
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], operator.add]
    research: str
    code: str
    validation: str
    final_output: str

# Tools for agents
@tool
def research_tool(query: str) -> str:
    """Simulate web research (in production, integrate web_search tool)."""
    # Placeholder - replace with real tool call if needed
    return f"Researched '{query}': Key findings include [simulated results]."

@tool
def code_gen_tool(description: str) -> str:
    """Generate Python code based on description."""
    return f"# Generated code for '{description}'\nimport numpy as np\n# Example: simple function\ndef example_func():\n    return 'Hello from agent!'"

@tool
def validate_code_tool(code: str) -> str:
    """Validate and test code (simulate execution)."""
    # Placeholder - use code_execution tool in production
    return f"Validated code: No errors found. Output: [simulated run]."

# Agent functions (parallel execution nodes)
def research_agent(state: AgentState) -> AgentState:
    prompt = f"Research this goal: {state['messages'][-1].content}. Summarize key findings."
    response = llm.invoke([HumanMessage(content=prompt)])
    state["research"] = response.content
    return state

def code_agent(state: AgentState) -> AgentState:
    prompt = f"Based on research '{state['research]}', generate Python code for the goal."
    response = llm.invoke([HumanMessage(content=prompt)])
    state["code"] = response.content
    return state

def validator_agent(state: AgentState) -> AgentState:
    prompt = f"Validate code: {state['code']}. Test and suggest fixes if needed."
    response = llm.invoke([HumanMessage(content=prompt)])
    state["validation"] = response.content
    state["final_output"] = f"Research: {state['research']}\nCode: {state['code']}\nValidation: {state['validation']}"
    return state

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("research", research_agent)
workflow.add_node("code", code_agent)
workflow.add_node("validate", validator_agent)

# Parallel: Research and code run concurrently after entry
workflow.set_entry_point("research")
workflow.add_edge("research", "code")  # Sequential for simplicity; use add_conditional_edges for true parallel
workflow.add_edge("code", "validate")
workflow.add_edge("validate", END)

# Compile
app = workflow.compile()

# Streamlit frontend for deployment
st.title("Agentic AI: Multi-Agent Research & Code Generator")
goal = st.text_area("Enter your goal (e.g., 'Research quantum error correction and generate Python code')", height=100)

if st.button("Run Agents"):
    if goal:
        initial_state = {"messages": [HumanMessage(content=goal)]}
        result = app.invoke(initial_state)
        st.subheader("Final Output")
        st.write(result["final_output"])
    else:
        st.warning("Enter a goal to start the agents.")

# Footer
st.markdown("---")
st.markdown("Built with LangGraph + Mistral AI | Deployed on Render | Open Source on GitHub")
