# Medical Diagnose Bot Documentation

## Overview
This project implements a **Medical Diagnose Bot** using **Streamlit**, **LangChain**, **Pinecone**, and **HuggingFace**. The bot leverages a **Retrieval-Augmented Generation (RAG)** pipeline to answer medical questions based on a pre-loaded healthcare document. It provides users with real-time diagnosis support by querying a medical knowledge base and generating concise answers.

## Features
- **Medical Diagnosis**: Answers user queries related to medical conditions, symptoms, treatments, and medications.
- **RAG (Retrieval-Augmented Generation)**: Uses Pinecone for document retrieval and HuggingFace models for question answering.
- **Interactive UI**: A user-friendly interface built with Streamlit that supports chat-like interactions.
- **Previous Chats**: Stores and allows users to view previous conversations and also to clear the history of the conversations.
- **Disclaimer**: Displays a disclaimer to ensure the bot’s responses are informational and not a substitute for professional medical advice.

## Tech Stack
- **Streamlit**: For building the interactive web app.
- **LangChain**: For managing LLM chains and retrieval-augmented generation.
- **Pinecone**: For vector-based document search and retrieval.
- **HuggingFace**: For using the `Mixtral-8x7B-Instruct` model for generating responses.
- **dotenv**: For loading API keys securely from environment variables.
- **Python**: Main programming language.

## File Structure
``` bash
medical-diagnose-bot/
├── app.py                    # Main Streamlit app file
├── healthcare_info_processed.txt  # Text file containing healthcare-related documents
├── .env                      # Environment file for API keys
├── requirements.txt          # List of project dependencies
└── README.md                 # This documentation
```

## Setup Instructions

### 1. Install Dependencies
Ensure you have **Python 3.7+** installed. Then, install the required libraries by running:

```bash
pip install streamlit langchain pinecone-client huggingface-hub python-dotenv
```

### 2. Create .env File
Create a .env file in the root directory of your project and add the following API keys:

```bash
PINECONE_API_KEY=your_pinecone_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```


### 3. Run the Application
Once you have installed the dependencies and set up the environment variables, you can start the Streamlit app by running the following command:

```bash
streamlit run app.py
```
Visit http://localhost:8501 in your browser to interact with the bot.

