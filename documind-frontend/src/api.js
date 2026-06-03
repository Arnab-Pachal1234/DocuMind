import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL;

if (!API_BASE_URL) {
  console.error("VITE_API_URL is missing. Check your .env file.");
}

const API = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

export const createThread = async () => {
  try {
    const response = await API.post("/create-thread");
    return response.data;
  } catch (error) {
    console.error("Create thread error:", error);
    throw error;
  }
};

export const uploadPDF = async (files, threadId) => {
  try {
    const formData = new FormData();

    Array.from(files).forEach((file) => {
      formData.append("files", file);
    });

    formData.append("thread_id", threadId);

    const response = await API.post("/upload-pdf", formData);

    return response.data;
  } catch (error) {
    console.error("Upload PDF error:", error);
    throw error;
  }
};

export const askQuestion = async (threadId, question) => {
  try {
    const response = await API.post("/ask", {
      thread_id: threadId,
      question,
    });

    return response.data;
  } catch (error) {
    console.error("Ask question error:", error);
    throw error;
  }
};

export const getHistory = async (threadId) => {
  try {
    const response = await API.get(`/history/${threadId}`);
    return response.data;
  } catch (error) {
    console.error("Get history error:", error);
    throw error;
  }
};