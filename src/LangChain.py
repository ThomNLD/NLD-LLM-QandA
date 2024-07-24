import os
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
load_dotenv("../.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Specify the encoding when loading the CSV file
loader = CSVLoader(
    file_path="../docs/codebasics_faqs.csv",
    source_column="prompt",
    encoding="Windows-1252",
)

# Load the data from the CSV file
data = loader.load()

# Initialize the HuggingFace embeddings model
embeddings_model = HuggingFaceEmbeddings()

# Create a FAISS vector store from the loaded documents
vectordb = FAISS.from_documents(documents=data, embedding=embeddings_model)

# Initialize a retriever from the vector store
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# Retrieve documents based on a query
docs = retriever.invoke("How about job placement support?")

# Initialize the OpenAI LLM (ChatGPT model)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# Pull the prompt template from the LangChain hub
prompt = hub.pull("rlm/rag-prompt")


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

# Invoke the RAG chain with a question and stream the results
question = "Is does course helpful for my resume, and why?"
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
