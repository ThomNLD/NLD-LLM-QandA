import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Suppress the huggingface/tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv(".env")
openai_api_key = os.getenv("OPENAI_API_KEY")


def create_vector_db(
    csv_file_path="./src/codebasics_faqs.csv",
    source_column="prompt",
    vector_store_path="./docs/faiss_vector_store",
    encoding="Windows-1252",
):
    """
    Create or load a FAISS vector store from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.
        source_column (str): Column name in the CSV to use as the source text.
        vector_store_path (str): Path to save or load the vector store.
        encoding (str): Encoding of the CSV file.

    Returns:
        FAISS: The vector store object.
    """
    # Load the data from the CSV file
    loader = CSVLoader(
        file_path=csv_file_path, source_column=source_column, encoding=encoding
    )
    data = loader.load()

    # Initialize the HuggingFace embeddings model
    embeddings_model = HuggingFaceEmbeddings()

    # Check if the vector store already exists
    if os.path.exists(vector_store_path):
        # Load the existing vector store
        vectordb = FAISS.load_local(
            vector_store_path,
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True,
        )
    else:
        # Create a new vector store from the documents
        vectordb = FAISS.from_documents(documents=data, embedding=embeddings_model)
        # Save the vector store to disk
        vectordb.save_local(vector_store_path)

    return vectordb


def get_qa_chain(
    vector_store_path="./docs/faiss_vector_store",
    openai_api_key=openai_api_key,
    prompt_name="rlm/rag-prompt",
    search_kwargs={"k": 2},
):
    """
    Get the RAG QA chain using the specified vector store and OpenAI LLM.

    Args:
        vector_store_path (str): Path to the vector store.
        openai_api_key (str): OpenAI API key.
        prompt_name (str): Name of the prompt to pull from LangChain hub.
        search_kwargs (dict): Search arguments for the retriever.

    Returns:
        callable: The RAG chain callable.
    """
    # Initialize the HuggingFace embeddings model
    embeddings_model = HuggingFaceEmbeddings()

    # Load the vector store
    vectordb = FAISS.load_local(
        vector_store_path,
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True,
    )

    # Initialize a retriever from the vector store
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

    # Initialize the OpenAI LLM (ChatGPT model)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

    # Pull the prompt template from the LangChain hub
    prompt = hub.pull(prompt_name)

    # Define a function to format the documents for the prompt
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Define the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Streamlit UI
st.title("Codebasics Q&A 🌱")

# Button to create the knowledgebase
if st.button("Create Knowledgebase"):
    create_vector_db()
    st.success("Knowledgebase created successfully!")

# Input for the question
question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = ""
    for chunk in chain.stream(question):
        response += chunk
    st.header("Answer")
    st.write(response)
