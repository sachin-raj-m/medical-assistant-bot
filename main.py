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

# Create a custom output parser to ensure we only get the answer text


class AnswerOnlyOutputParser(StrOutputParser):
    def parse(self, output: str) -> str:
        # I have given answer at the end of prompt template
        answer = output.split('Answer:')[-1].strip()
        return answer

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

# Prepare the prompt template
template = """
You are a medical assistant bot. The Humans will ask you a questions about their medical condition, symptoms, treatment options and medications. 
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

# Load documents from text file
documents = load_text_data('healthcare_info_processed.txt')

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

# Create Pinecone index from documents
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# Create the RAG chain with the custom answer-only output parser
rag_chain = (
    {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | AnswerOnlyOutputParser()  # Use the custom output parser here
)

# Function for generating LLM response


def generate_response(input):
    result = rag_chain.invoke(input)
    return result  # The AnswerOnlyOutputParser will ensure only the answer is returned


# Streamlit app setup
st.set_page_config(page_title="Medical Diagnose Bot", page_icon="💉")
st.title("Medical Diagnose Bot 💉")

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

            # DEBUGGING: Retrieve the context and log or display it for debugging purposes
            # context = rag_chain.input_variables["context"]
            # st.write(f"**DEBUGGING: Retrieved Context:**\n{context}")
            # print(f"**DEBUGGING: Retrieved Context:**\n{context}")

            st.write(response)  # This will now only display the answer
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)


# Add custom CSS for styling
st.markdown("""
    <style>
    .chat-message {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #1A1A19;
        align-self: flex-end;
    }
    .assistant-message {
        background-color: #4A628A;
       
        align-self: flex-start;
    }
    .message-box {
        max-width: 700px;
        margin: 0 auto;
    }
    .header {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .button-container {
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 1000;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar buttons
with st.sidebar:
    st.header("Chat Options")
    if st.button("Start New Chat"):
        # Preserve the current chat history
        if "messages" in st.session_state:
            if "previous_chats" not in st.session_state:
                st.session_state.previous_chats = []
            st.session_state.previous_chats.append(st.session_state.messages)
        st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's diagnose and help you!"}]
    
    if st.button("Previous Chats"):
        if "previous_chats" in st.session_state and st.session_state.previous_chats:
            for idx, chat in enumerate(st.session_state.previous_chats):
                st.write(f"### Chat {idx + 1}")
                for message in chat:
                    st.markdown(f'<div class="chat-message {message["role"]}-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.write("No previous chats available.")
