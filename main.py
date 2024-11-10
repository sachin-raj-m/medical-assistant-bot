import streamlit as st
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain import PromptTemplate
from langchain.llms import HuggingFaceHub
import pinecone
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from dotenv import load_dotenv


# Function to load text data and convert it to documents
def load_text_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    # Return documents, each containing the entire text file
    return [Document(page_content=content, metadata={})]


# Initialize Pinecone client
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment='gcp-starter'
)

# Define Index Name
index_name = "langchain-demo"

# Checking Index
if index_name not in pinecone.list_indexes():
    # Create new Index
    pinecone.create_index(name=index_name, metric="cosine", dimension=768)

# Define the repo ID and connect to Mixtral model on Huggingface
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.8, "top_k": 50},
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
)

template = """
You are a medical assistant bot. The Humans will ask you a questions about their medical condition, symptoms, treatment options, and medications. 
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
Keep the answer concise.  

Context: {context}
Question: {question}
Answer: 
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Embeddings and text splitter setup
embeddings = HuggingFaceEmbeddings()


# Function to reload documents based on the selected context source
def reload_documents():
    return load_text_data('healthcare_info_processed.txt')


# Function for creating RAG chain after context reload
def create_rag_chain(documents):
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)

    docsearch = Pinecone.from_documents(
        docs, embeddings, index_name=index_name)

    rag_chain = (
        {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Streamlit app setup
st.set_page_config(page_title="Medical Diagnose Bot")
with st.sidebar:
    st.title('Medical Diagnose Bot')

# Load documents from the text file
documents = reload_documents()

# Create the RAG chain based on the loaded documents
rag_chain = create_rag_chain(documents)

# Function for generating LLM response
def generate_response(input):
    result = rag_chain.invoke(input)
    return result


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome, let's diagnose and help you!"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer from medical database.."):
            response = generate_response(input)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
