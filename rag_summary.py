from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from utils import create_documents
from utils import create_vector_store
from utils import retrieve_context


# -----------------------------------------
# Gemini Model
# -----------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


# -----------------------------------------
# Generate RAG Summary
# -----------------------------------------

def generate_rag_summary(text):

    # Step 1: Split into chunks
    documents = create_documents(text)

    # Step 2: Build Retriever
    retriever = create_vector_store(documents)

    # Step 3: Retrieve relevant chunks
    context = retrieve_context(
        retriever,
        "Summarize this document"
    )

    # Step 4: Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant.

Use ONLY the retrieved context below.

Context:
{context}

Generate a structured summary with:

1. Introduction
2. Key Topics
3. Important Points
4. Conclusion

Do not add information outside the context.
"""
    )

    # Step 5: LangChain Chain
    chain = prompt | llm

    # Step 6: Invoke
    response = chain.invoke(
        {
            "context": context
        }
    )

    return response.content