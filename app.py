import streamlit as st
from utils import extract_text

from langchain_summary import generate_langchain_summary
from rag_summary import generate_rag_summary
from langgraph_summary import generate_langgraph_summary

# -----------------------------------------
# Streamlit Page Config
# -----------------------------------------

st.set_page_config(
    page_title="AI Document Summarizer",
    layout="wide"
)

st.title("📄 AI Document Summarizer")
st.markdown("### Compare **LangChain**, **RAG**, and **LangGraph** for PDF Summarization")

uploaded_file = st.file_uploader(
    "📤 Upload a PDF",
    type=["pdf"]
)

# -----------------------------------------
# MAIN
# -----------------------------------------

if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        text = extract_text(uploaded_file)

        if len(text.strip()) == 0:
            st.error("Could not extract text from PDF.")
            st.stop()

        st.success("✅ PDF Loaded Successfully!")

        # -----------------------------------------
        # Generate Summaries
        # -----------------------------------------

        langchain_result = generate_langchain_summary(text)

        rag_result = generate_rag_summary(text)

        langgraph_result = generate_langgraph_summary(text)

    # -----------------------------------------
    # Display Results
    # -----------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔵 LangChain Summary")
        st.write(langchain_result)

    with col2:
        st.subheader("🟢 RAG Summary")
        st.write(rag_result)

    with col3:
        st.subheader("🟣 LangGraph Summary")
        st.write(langgraph_result)

    # -----------------------------------------
    # Comparative Analysis
    # -----------------------------------------

    st.divider()

    st.header("📊 Comparative Analysis")

    comparison = {
        "Approach": [
            "LangChain",
            "RAG",
            "LangGraph"
        ],

        "Working Principle": [
            "Direct LLM Summarization",
            "Retrieve Relevant Chunks + LLM",
            "Multi-step Graph Workflow"
        ],

        "Speed": [
            "⭐⭐⭐⭐⭐ Fast",
            "⭐⭐⭐⭐ Medium",
            "⭐⭐⭐ Moderate"
        ],

        "Accuracy": [
            "⭐⭐⭐ Good",
            "⭐⭐⭐⭐ Excellent",
            "⭐⭐⭐⭐⭐ Best"
        ],

        "Best Use Case": [
            "Small PDFs",
            "Large Documents",
            "Complex AI Applications"
        ]
    }

    st.table(comparison)

    # -----------------------------------------
    # Pros & Cons
    # -----------------------------------------

    st.header("✅ Pros & Cons")

    pros_cons = {
        "Approach": [
            "LangChain",
            "RAG",
            "LangGraph"
        ],

        "Pros": [
            "Very Fast\nEasy to Build\nSimple Architecture",

            "High Accuracy\nRelevant Retrieval\nLess Hallucination",

            "Structured Workflow\nMulti-step Reasoning\nHighly Scalable"
        ],

        "Cons": [
            "May miss important context",

            "Slightly slower because retrieval is performed",

            "Most complex implementation"
        ]
    }

    st.table(pros_cons)

    # -----------------------------------------
    # Recommendation
    # -----------------------------------------

    st.header("🏆 Which Approach Should You Choose?")

    recommendation = {
        "Scenario": [
            "Quick Summary",
            "Small PDF",
            "Research Paper",
            "Large PDF",
            "Enterprise AI System"
        ],

        "Recommended": [
            "LangChain",
            "LangChain",
            "RAG",
            "RAG",
            "LangGraph"
        ],

        "Reason": [
            "Fastest",

            "Simple",

            "Best Retrieval",

            "Handles Context Better",

            "Supports Complex AI Workflows"
        ]
    }

    st.table(recommendation)

    st.success("✅ Analysis Completed Successfully!")

    st.markdown("---")

    st.subheader("📌 Final Conclusion")

    st.info(
        """
### 🔵 LangChain
- Fastest approach
- Simple implementation
- Best for small documents

### 🟢 RAG
- Retrieves relevant information before summarizing
- Better accuracy for large PDFs
- Reduces hallucinations

### 🟣 LangGraph
- Uses a workflow of multiple AI steps
- Best reasoning capability
- Ideal for enterprise and production AI systems

🏆 **Overall Recommendation:**

- Small PDFs → **LangChain**
- Large PDFs → **RAG**
- Complex AI Workflows → **LangGraph**
"""
    )