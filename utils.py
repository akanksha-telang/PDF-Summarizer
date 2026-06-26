from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------
# Extract text from uploaded PDF
# ---------------------------------------------------

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ---------------------------------------------------
# Split document into LangChain Documents
# ---------------------------------------------------

def create_documents(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    documents = []

    for chunk in chunks:
        documents.append(
            Document(page_content=chunk)
        )

    return documents


# ---------------------------------------------------
# TF-IDF Retriever (No FAISS, No Embeddings)
# ---------------------------------------------------

class TFIDFRetriever:

    def __init__(self, documents):

        self.documents = documents

        self.texts = [doc.page_content for doc in documents]

        self.vectorizer = TfidfVectorizer()

        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, k=4):

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(query_vector, self.matrix).flatten()

        ranked = scores.argsort()[::-1]

        results = []

        for index in ranked[:k]:
            results.append(self.documents[index])

        return results


# ---------------------------------------------------
# Create Retriever
# ---------------------------------------------------

def create_vector_store(documents):

    return TFIDFRetriever(documents)


# ---------------------------------------------------
# Retrieve Context
# ---------------------------------------------------

def retrieve_context(retriever, query):

    docs = retriever.retrieve(query, k=4)

    context = ""

    for doc in docs:

        context += doc.page_content

        context += "\n\n"

    return context