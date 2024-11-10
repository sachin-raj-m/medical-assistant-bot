import streamlit as st
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain import PromptTemplate
from langchain.llms import HuggingFaceHub
import pinecone
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

loader = TextLoader('augmented_natural_healthcare_dataset.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()


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
    docsearch = Pinecone.from_documents(
        docs, embeddings, index_name=index_name)
else:
    # Link to the existing index
    docsearch = Pinecone.from_existing_index(index_name, embeddings)


# Define the repo ID and connect to Mixtral model on Huggingface
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.8, "top_k": 50},
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
)


template = """
You are a medical assistant bot. The Humans will ask you a questions about their medical condition,symptoms, treatment options and medications. 
Use following piece of context which is provided to answer the question. 
If you don't know the answer, just say you don't know. 
Keep the answer within 2 sentences and concise.

Context: {context}
Question: {question}
Answer: 

"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)


rag_chain = (
    {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Import dependencies here
class ChatBot():
    load_dotenv()
    loader = TextLoader('augmented_natural_healthcare_dataset.txt')
    documents = loader.load()

    # The rest of the code here

    rag_chain = (
        {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# Outside ChatBot() class
# bot = ChatBot()
# input = input("Ask me anything: ")
# result = bot.rag_chain.invoke(input)
# print(result)


bot = ChatBot()

st.set_page_config(page_title="Medical Diagnose Bot")
with st.sidebar:
    st.title('Medical Diagnose Bot')

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return result


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome, let's diagonose and help you!"}]

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
