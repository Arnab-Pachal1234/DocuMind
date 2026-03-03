# 📚 DocuMind -- RAG PDF Chatbot (Gemini + LangChain)

DocuMind is a Retrieval-Augmented Generation (RAG) chatbot that allows
users to upload PDF documents and ask intelligent questions based
strictly on the uploaded content.

Built using:

-   Streamlit (Frontend UI)
-   LangChain 1.x (LLM orchestration)
-   Google Gemini 1.5 Flash (LLM)
-   Google Generative AI Embeddings
-   FAISS (Vector Database)
-   PyPDF2 (PDF Processing)

------------------------------------------------------------------------

## 🚀 Features

-   📂 Upload multiple PDF files
-   ✂ Automatic text chunking
-   🧠 Embedding generation using Google Embeddings
-   📊 FAISS vector similarity search
-   🤖 Gemini-powered contextual answers
-   ⚡ Fast and lightweight Streamlit UI
-   🔒 Strict context-based answering (No hallucination mode)

------------------------------------------------------------------------

## 🏗️ Project Architecture

User Upload PDFs\
→ Extract Text\
→ Split into Chunks\
→ Generate Embeddings\
→ Store in FAISS\
→ Similarity Search\
→ Inject Context into Prompt\
→ Gemini Generates Answer

This follows a proper RAG (Retrieval-Augmented Generation) pipeline.

------------------------------------------------------------------------

## 📦 Installation Guide

### 1️⃣ Clone the Repository

``` bash
git clone https://github.com/your-username/documind.git
cd documind
```

### 2️⃣ Create Virtual Environment

``` bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

``` bash
pip install -U streamlit
pip install -U langchain langchain-core langchain-community
pip install -U langchain-google-genai
pip install -U langchain-text-splitters
pip install -U faiss-cpu
pip install -U PyPDF2
pip install -U python-dotenv
```

------------------------------------------------------------------------

## 🔑 Environment Variables

Create a `.env` file in the root directory:

    GOOGLE_API_KEY=your_google_api_key_here

Get your API key from: https://makersuite.google.com/app/apikey

------------------------------------------------------------------------

## ▶️ Running the Application

``` bash
streamlit run app.py
```

Then open the browser link shown in terminal.

------------------------------------------------------------------------

## 🧠 How It Works

### Step 1: Upload PDFs

Upload one or multiple PDF documents via sidebar.

### Step 2: Processing

-   Extracts text using PyPDF2
-   Splits text using RecursiveCharacterTextSplitter
-   Creates embeddings using GoogleGenerativeAIEmbeddings
-   Stores vectors in FAISS

### Step 3: Ask Questions

User question → Similarity search → Context injection → Gemini answer

The model strictly answers only from retrieved context.

------------------------------------------------------------------------

## 📁 Project Structure

    DocuMind/
    │
    ├── app.py
    ├── .env
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🔒 Safety & Design

-   No hallucinated answers
-   Strict context enforcement
-   Session-based vector storage
-   Clean modular architecture
-   Compatible with LangChain 1.x

------------------------------------------------------------------------

## 📌 Tech Stack

  Component       Technology
  --------------- ----------------------
  Frontend        Streamlit
  LLM             Gemini 1.5 Flash
  Embeddings      Google Generative AI
  Vector DB       FAISS
  PDF Parsing     PyPDF2
  Orchestration   LangChain 1.x

------------------------------------------------------------------------

## 💡 Future Improvements

-   Persistent FAISS storage
-   Streaming LLM responses
-   Chat history memory
-   Deployment on Streamlit Cloud
-   Docker containerization

------------------------------------------------------------------------

## 👨‍💻 Author

Arnab Pachai\
B.Tech CSE Student\
RAG & LLM Application Developer

------------------------------------------------------------------------

## ⭐ If You Like This Project

Give it a ⭐ on GitHub and feel free to fork and improve!

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License.
