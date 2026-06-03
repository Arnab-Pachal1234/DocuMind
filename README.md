# 📚 DocuMind — RAG PDF Chatbot

DocuMind is a full-stack **Retrieval-Augmented Generation (RAG)** PDF chatbot that allows users to upload PDF documents and ask intelligent questions based strictly on the uploaded content.

The application extracts text from PDFs, chunks the content, generates embeddings, stores vectors in **Qdrant**, retrieves only the most relevant chunks for each question, and generates answers using **OpenAI**.

---

## 🚀 Live Architecture

```text
React Frontend
    ↓
FastAPI Backend
    ↓
PDF Text Extraction
    ↓
Text Chunking
    ↓
OpenAI Embeddings
    ↓
Qdrant Vector Database
    ↓
Relevant Chunk Retrieval
    ↓
OpenAI Chat Model
    ↓
MongoDB Chat History
```

---

## 🧠 Updated Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite + Tailwind CSS |
| Backend | FastAPI |
| LLM | OpenAI GPT Model |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Database | Qdrant |
| Database | MongoDB Atlas |
| PDF Processing | pypdf |
| RAG Orchestration | LangChain |
| Backend Deployment | Render |
| Frontend Deployment | Vercel |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Hub |

---

## ✨ Features

- 📂 Upload one or multiple PDF files
- ✂️ Automatic text extraction and chunking
- 🧠 OpenAI embedding generation
- 🔍 Semantic search using Qdrant Vector DB
- 💬 Ask questions from uploaded PDF content
- 📌 Retrieves only relevant chunks for each question
- 🤖 OpenAI-powered contextual answering
- 🧾 MongoDB-based persistent chat history
- 🔗 Shareable thread-based chat sessions
- 🌐 React frontend with beautiful modern UI
- 🐳 Dockerized FastAPI backend
- 🚀 Backend auto-deployment using Render
- ⚡ Frontend auto-deployment using Vercel
- 🔁 CI/CD pipeline using GitHub Actions

---

## 🏗️ Project Structure

```text
DocuMind/
│
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── schemas.py
│   ├── pdf_utils.py
│   ├── vector_store.py
│   ├── rag_service.py
│   ├── chat_service.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   └── .env.example
│
├── documind_frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   └── .env.example
│
└── .github/
    └── workflows/
        ├── docker-image.yml
        └── frontend-deploy.yml
```

---

## 🔄 RAG Pipeline

```text
1. User uploads PDF
2. FastAPI receives PDF
3. pypdf extracts text
4. Text is split into chunks
5. OpenAI creates embeddings for each chunk
6. Chunks + metadata are stored in Qdrant
7. User asks a question
8. Question is embedded using OpenAI
9. Qdrant retrieves relevant chunks using semantic search
10. OpenAI generates an answer from only those chunks
11. Chat history is saved in MongoDB
```

---

## ⚙️ Backend Setup

### 1. Go to backend folder

```bash
cd backend
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install backend dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Create `.env` file

Create a `.env` file inside the `backend/` folder:

```env
OPENAI_API_KEY=your_openai_api_key

MONGO_USER=your_mongodb_username
MONGO_PASS=your_mongodb_password
MONGO_CLUSTER=cluster0.xxxxx.mongodb.net

QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=documind_chunks

BACKEND_URL=http://localhost:8000
```

### 5. Run backend locally

```bash
uvicorn main:app --reload
```

Backend will run at:

```text
http://localhost:8000
```

Swagger API docs:

```text
http://localhost:8000/docs
```

---

## 🔌 Backend API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Backend health/home route |
| `GET` | `/health` | Health check route |
| `POST` | `/create-thread` | Creates a new chat thread |
| `POST` | `/upload-pdf` | Uploads and processes PDF files |
| `POST` | `/ask` | Asks a question from uploaded PDFs |
| `GET` | `/history/{thread_id}` | Gets chat history for a thread |
| `GET` | `/thread/{thread_id}` | Gets thread information |
| `DELETE` | `/history/{thread_id}` | Deletes chat history for a thread |

---

## 🎨 Frontend Setup

### 1. Go to frontend folder

```bash
cd documind_frontend
```

### 2. Install dependencies

```bash
npm install
```

### 3. Create `.env` file

Create a `.env` file inside the `documind_frontend/` folder:

```env
VITE_API_URL=http://localhost:8000
```

For production:

```env
VITE_API_URL=https://your-render-backend.onrender.com
```

### 4. Run frontend locally

```bash
npm run dev
```

Frontend will run at:

```text
http://localhost:5173
```

---

## 🐳 Docker Backend Setup

### Build Docker image locally

```bash
docker build -t documind-backend .
```

### Run Docker container locally

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your_openai_key" \
  -e MONGO_USER="your_mongo_user" \
  -e MONGO_PASS="your_mongo_password" \
  -e MONGO_CLUSTER="your_mongo_cluster" \
  -e QDRANT_URL="your_qdrant_url" \
  -e QDRANT_API_KEY="your_qdrant_api_key" \
  -e QDRANT_COLLECTION_NAME="documind_chunks" \
  documind-backend
```

Then open:

```text
http://localhost:8000/docs
```

---

## 🚀 Deployment

### Backend Deployment — Render

The backend is deployed on **Render** using a Docker image.

Recommended backend deployment flow:

```text
GitHub Push
    ↓
GitHub Actions
    ↓
Docker Build
    ↓
Push Image to Docker Hub
    ↓
Render Deploy Hook
    ↓
Render Pulls Latest Image
    ↓
Backend Live
```

Render environment variables required:

```env
OPENAI_API_KEY=your_openai_api_key
MONGO_USER=your_mongodb_username
MONGO_PASS=your_mongodb_password
MONGO_CLUSTER=cluster0.xxxxx.mongodb.net
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=documind_chunks
BACKEND_URL=https://your-render-backend.onrender.com
```

---

### Frontend Deployment — Vercel

The React frontend is deployed on **Vercel**.

Vercel settings:

```text
Framework Preset: Vite
Root Directory: documind_frontend
Build Command: npm run build
Output Directory: dist
Install Command: npm install
```

Vercel environment variable:

```env
VITE_API_URL=https://your-render-backend.onrender.com
```

---

## 🔁 CI/CD Pipeline

### Backend CI/CD

Workflow file:

```text
.github/workflows/docker-image.yml
```

Backend CI/CD flow:

```text
Push to main
    ↓
Build Docker image
    ↓
Push to Docker Hub
    ↓
Trigger Render deploy hook
```

Required GitHub Secrets:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
RENDER_DEPLOY_HOOK_URL
```

---

### Frontend CI/CD

Workflow file:

```text
.github/workflows/frontend-deploy.yml
```

Frontend CI/CD flow:

```text
Push frontend changes
    ↓
Install dependencies
    ↓
Build React app
    ↓
Trigger Vercel deploy hook
```

Required GitHub Secrets:

```text
VITE_API_URL
VERCEL_DEPLOY_HOOK_URL
```

---

## 🧪 Testing Flow

### 1. Start backend

```bash
uvicorn main:app --reload
```

### 2. Start frontend

```bash
npm run dev
```

### 3. Open frontend

```text
http://localhost:5173
```

### 4. Test the app

1. Upload a text-based PDF.
2. Wait for successful processing.
3. Ask a document-specific question.
4. Check the answer and retrieved chunk indexes.
5. Verify chat history is preserved.

---

## ⚠️ Important Notes

### PDF Text Extraction

This project uses `pypdf`, which works best with **text-based PDFs**.

If a PDF is scanned or image-based, text extraction may fail. For scanned PDFs, OCR support can be added later using tools like:

- Tesseract OCR
- PyMuPDF
- AWS Textract
- Google Document AI

### Qdrant Point IDs

Qdrant point IDs must be either:

```text
unsigned integers
or valid UUID strings
```

This project uses UUID strings for safe point insertion.

### Qdrant Payload Indexing

Filtering by thread requires a payload index on:

```text
metadata.thread_id
```

The backend creates this index automatically.

### Avoid Duplicate Chunks

When uploading again for the same thread, old chunks should be deleted before adding new chunks to avoid duplicate retrieval.

---

## 🧠 Example Questions

After uploading a PDF, try asking:

```text
What is the title of this document?
```

```text
What is the main objective of this document?
```

```text
Summarize the methodology.
```

```text
What are the key findings?
```

```text
Explain this document in simple words.
```

---

## 🔒 Safety & Design

- Answers are generated only from retrieved document chunks.
- The model is instructed not to guess.
- Chat history is stored per thread.
- Qdrant stores chunk embeddings and metadata.
- MongoDB stores persistent chat messages.
- Environment secrets are not committed to GitHub.
- Docker image excludes `.env` using `.dockerignore`.

---

## 📦 Production Improvements

Future improvements can include:

- Streaming responses
- User authentication
- PDF storage using Cloudinary or AWS S3
- OCR for scanned PDFs
- Better citation support
- Chunk preview in frontend
- Admin dashboard
- Rate limiting
- Multi-user workspace support
- Better error handling and retry logic
- Background processing for large PDFs

---

## 👨‍💻 Author

**Arnab Pachal**  
B.Tech CSE Student  
RAG & LLM Application Developer

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub and feel free to fork and improve it.

---

## 📜 License

This project is licensed under the MIT License.
