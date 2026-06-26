from typing import TypedDict

from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ------------------------------------------
# Gemini Model
# ------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


# ------------------------------------------
# Graph State
# ------------------------------------------

class GraphState(TypedDict):
    text: str
    topics: str
    explanation: str
    summary: str


# ------------------------------------------
# Node 1: Identify Topics
# ------------------------------------------

def identify_topics(state: GraphState):

    prompt = ChatPromptTemplate.from_template("""
Identify the main topics from the following document.

Document:
{text}
""")

    chain = prompt | llm

    result = chain.invoke({
        "text": state["text"][:12000]
    })

    state["topics"] = result.content

    return state


# ------------------------------------------
# Node 2: Explain Topics
# ------------------------------------------

def explain_topics(state: GraphState):

    prompt = ChatPromptTemplate.from_template("""
Explain these topics clearly.

Topics:
{topics}
""")

    chain = prompt | llm

    result = chain.invoke({
        "topics": state["topics"]
    })

    state["explanation"] = result.content

    return state


# ------------------------------------------
# Node 3: Final Summary
# ------------------------------------------

def generate_summary(state: GraphState):

    prompt = ChatPromptTemplate.from_template("""
Create a final structured summary using the explanation below.

Explanation:
{explanation}

Include:

1. Introduction

2. Key Topics

3. Important Insights

4. Conclusion
""")

    chain = prompt | llm

    result = chain.invoke({
        "explanation": state["explanation"]
    })

    state["summary"] = result.content

    return state


# ------------------------------------------
# Build LangGraph Workflow
# ------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node("identify_topics", identify_topics)
workflow.add_node("explain_topics", explain_topics)
workflow.add_node("generate_summary", generate_summary)

workflow.set_entry_point("identify_topics")

workflow.add_edge("identify_topics", "explain_topics")
workflow.add_edge("explain_topics", "generate_summary")
workflow.add_edge("generate_summary", END)

graph = workflow.compile()


# ------------------------------------------
# Function to call from app.py
# ------------------------------------------

def generate_langgraph_summary(text):

    result = graph.invoke({

        "text": text,

        "topics": "",

        "explanation": "",

        "summary": ""

    })

    return result["summary"]