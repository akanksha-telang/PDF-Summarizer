from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template("""
You are an expert document summarizer.

Summarize the following document in a clear and structured way.

Document:
{document}

Include:
- Overview
- Key Points
- Important Findings
- Final Conclusion
""")

chain = prompt | llm


def generate_langchain_summary(text):
    response = chain.invoke(
        {
            "document": text[:15000]
        }
    )

    return response.content