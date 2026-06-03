import { useEffect, useRef, useState } from "react";
import {
  Upload,
  Send,
  Brain,
  FileText,
  Copy,
  CheckCircle2,
  Loader2,
  Sparkles,
  MessageSquareText,
  Link as LinkIcon,
} from "lucide-react";

import {
  createThread,
  uploadPDF,
  askQuestion,
  getHistory,
} from "./api";

function App() {
  const [threadId, setThreadId] = useState("");
  const [messages, setMessages] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [question, setQuestion] = useState("");

  const [isCreatingThread, setIsCreatingThread] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [copied, setCopied] = useState(false);

  const chatEndRef = useRef(null);

  const backendUrl = import.meta.env.VITE_API_URL;
  const shareUrl = `${window.location.origin}/?thread=${threadId}`;

  useEffect(() => {
    initializeThread();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const initializeThread = async () => {
    try {
      setIsCreatingThread(true);

      const params = new URLSearchParams(window.location.search);
      const existingThread = params.get("thread");

      if (existingThread) {
        setThreadId(existingThread);

        try {
          const historyData = await getHistory(existingThread);
          setMessages(historyData.history || []);
        } catch (error) {
          console.error("Failed to load history:", error);
        }
      } else {
        const data = await createThread();
        setThreadId(data.thread_id);

        window.history.replaceState(
          null,
          "",
          `/?thread=${data.thread_id}`
        );
      }
    } catch (error) {
      alert("Failed to create thread. Please check backend URL.");
      console.error(error);
    } finally {
      setIsCreatingThread(false);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFiles(e.target.files);
    setUploadStatus("");
  };

  const handleUpload = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      alert("Please select at least one PDF file.");
      return;
    }

    try {
      setIsUploading(true);
      setUploadStatus("");

      const result = await uploadPDF(selectedFiles, threadId);

      setUploadStatus(
        `Uploaded successfully. ${result.chunks_created} chunks stored in Qdrant.`
      );
    } catch (error) {
      console.error(error);
      alert(error.response?.data?.detail || "PDF upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleAsk = async (e) => {
    e.preventDefault();

    if (!question.trim()) return;

    const userMessage = {
      role: "user",
      content: question,
    };

    setMessages((prev) => [...prev, userMessage]);

    const currentQuestion = question;
    setQuestion("");

    try {
      setIsAsking(true);

      const result = await askQuestion(threadId, currentQuestion);

      const assistantMessage = {
        role: "assistant",
        content: result.answer,
        chunks_used: result.chunks_used,
        used_chunk_indexes: result.used_chunk_indexes,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error(error);

      const errorMessage = {
        role: "assistant",
        content:
          error.response?.data?.detail ||
          "Something went wrong while generating the answer.",
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsAsking(false);
    }
  };

  const copyShareLink = async () => {
    await navigator.clipboard.writeText(shareUrl);
    setCopied(true);

    setTimeout(() => {
      setCopied(false);
    }, 1500);
  };

  if (isCreatingThread) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center text-white">
        <div className="flex flex-col items-center gap-4">
          <div className="h-16 w-16 rounded-3xl bg-indigo-600 flex items-center justify-center shadow-xl shadow-indigo-500/30">
            <Brain size={34} />
          </div>
          <div className="flex items-center gap-2 text-slate-300">
            <Loader2 className="animate-spin" size={20} />
            Starting DocuMind...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-32 left-20 h-72 w-72 rounded-full bg-indigo-600/20 blur-3xl" />
        <div className="absolute bottom-0 right-10 h-96 w-96 rounded-full bg-cyan-500/10 blur-3xl" />
      </div>

      <div className="relative min-h-screen flex flex-col lg:flex-row">
        <aside className="w-full lg:w-[360px] border-r border-white/10 bg-white/5 backdrop-blur-xl p-5 lg:p-6 flex flex-col gap-5">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/30">
              <Brain size={28} />
            </div>

            <div>
              <h1 className="text-2xl font-bold tracking-tight">DocuMind</h1>
              <p className="text-sm text-slate-400">AI PDF RAG Assistant</p>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-5 shadow-xl">
            <div className="flex items-center gap-2 mb-4">
              <FileText className="text-indigo-400" size={20} />
              <h2 className="font-semibold">Document Center</h2>
            </div>

            <label className="group flex flex-col items-center justify-center border-2 border-dashed border-slate-700 hover:border-indigo-400 rounded-2xl p-6 cursor-pointer transition bg-slate-950/50">
              <Upload className="text-slate-400 group-hover:text-indigo-400 mb-3" />
              <p className="text-sm text-slate-300 font-medium">
                Select PDF files
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Upload one or multiple PDFs
              </p>

              <input
                type="file"
                accept="application/pdf"
                multiple
                onChange={handleFileChange}
                className="hidden"
              />
            </label>

            {selectedFiles.length > 0 && (
              <div className="mt-4 space-y-2">
                {Array.from(selectedFiles).map((file, index) => (
                  <div
                    key={index}
                    className="text-xs bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-slate-300 truncate"
                  >
                    {file.name}
                  </div>
                ))}
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={isUploading}
              className="mt-4 w-full h-11 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-900 disabled:text-slate-400 transition font-semibold flex items-center justify-center gap-2"
            >
              {isUploading ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Upload size={18} />
                  Submit & Process
                </>
              )}
            </button>

            {uploadStatus && (
              <div className="mt-4 text-xs text-emerald-300 bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-3 flex gap-2">
                <CheckCircle2 size={16} />
                <span>{uploadStatus}</span>
              </div>
            )}
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-5 shadow-xl">
            <div className="flex items-center gap-2 mb-4">
              <LinkIcon className="text-cyan-400" size={20} />
              <h2 className="font-semibold">Share Thread</h2>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
              <p className="text-xs text-slate-500 mb-1">Thread URL</p>
              <p className="text-xs text-slate-300 break-all">{shareUrl}</p>
            </div>

            <button
              onClick={copyShareLink}
              className="mt-4 w-full h-10 rounded-xl bg-slate-800 hover:bg-slate-700 transition text-sm font-medium flex items-center justify-center gap-2"
            >
              {copied ? (
                <>
                  <CheckCircle2 size={16} className="text-emerald-400" />
                  Copied
                </>
              ) : (
                <>
                  <Copy size={16} />
                  Copy Link
                </>
              )}
            </button>
          </div>

          <div className="mt-auto rounded-2xl border border-white/10 bg-slate-900/50 p-4">
            <p className="text-xs text-slate-500">Backend</p>
            <p className="text-xs text-slate-300 break-all mt-1">
              {backendUrl}
            </p>

            <p className="text-xs text-slate-500 mt-4">Thread ID</p>
            <p className="text-xs text-slate-300 break-all mt-1">
              {threadId}
            </p>
          </div>
        </aside>

        <main className="flex-1 flex flex-col min-h-screen">
          <header className="border-b border-white/10 bg-slate-950/60 backdrop-blur-xl px-5 lg:px-8 py-5">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
              <div>
                <div className="flex items-center gap-2 text-indigo-300 text-sm font-medium">
                  <Sparkles size={16} />
                  Context-Aware Document Chat
                </div>
                <h2 className="text-2xl lg:text-3xl font-bold mt-1">
                  Ask your PDF anything
                </h2>
              </div>

              <div className="flex items-center gap-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 px-4 py-2 text-sm text-emerald-300">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                Render Backend Connected
              </div>
            </div>
          </header>

          <section className="flex-1 overflow-y-auto px-5 lg:px-8 py-6">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center">
                <div className="max-w-xl text-center">
                  <div className="mx-auto h-20 w-20 rounded-3xl bg-white/10 border border-white/10 flex items-center justify-center mb-6">
                    <MessageSquareText size={38} className="text-indigo-300" />
                  </div>

                  <h3 className="text-2xl font-bold mb-3">
                    Start by uploading a PDF
                  </h3>

                  <p className="text-slate-400 leading-relaxed">
                    DocuMind will chunk your PDF, store embeddings in Qdrant,
                    and answer your question using only the most relevant
                    document chunks.
                  </p>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-8 text-left">
                    <FeatureCard title="Upload" text="Add your PDF files." />
                    <FeatureCard title="Retrieve" text="Find relevant chunks." />
                    <FeatureCard title="Answer" text="Ask with context." />
                  </div>
                </div>
              </div>
            ) : (
              <div className="max-w-4xl mx-auto space-y-5">
                {messages.map((message, index) => (
                  <ChatMessage key={index} message={message} />
                ))}

                {isAsking && (
                  <div className="flex justify-start">
                    <div className="rounded-2xl rounded-bl-md bg-white/10 border border-white/10 px-4 py-3 text-slate-300 flex items-center gap-2">
                      <Loader2 size={17} className="animate-spin" />
                      Thinking with your document...
                    </div>
                  </div>
                )}

                <div ref={chatEndRef} />
              </div>
            )}
          </section>

          <form
            onSubmit={handleAsk}
            className="border-t border-white/10 bg-slate-950/80 backdrop-blur-xl p-4 lg:p-6"
          >
            <div className="max-w-4xl mx-auto flex gap-3">
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask something from your uploaded PDF..."
                className="flex-1 h-13 rounded-2xl bg-slate-900 border border-slate-700 focus:border-indigo-400 outline-none px-5 text-slate-100 placeholder:text-slate-500"
              />

              <button
                disabled={isAsking}
                className="h-13 px-5 md:px-7 rounded-2xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-900 disabled:text-slate-400 transition font-semibold flex items-center gap-2"
              >
                {isAsking ? (
                  <Loader2 size={19} className="animate-spin" />
                ) : (
                  <Send size={19} />
                )}
                <span className="hidden md:inline">Ask</span>
              </button>
            </div>
          </form>
        </main>
      </div>
    </div>
  );
}

function FeatureCard({ title, text }) {
  return (
    <div className="rounded-2xl bg-white/5 border border-white/10 p-4">
      <h4 className="font-semibold text-slate-200">{title}</h4>
      <p className="text-sm text-slate-500 mt-1">{text}</p>
    </div>
  );
}

function ChatMessage({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[90%] md:max-w-[78%] rounded-3xl px-5 py-4 shadow-lg ${
          isUser
            ? "bg-indigo-600 text-white rounded-br-md"
            : "bg-white/10 border border-white/10 text-slate-100 rounded-bl-md"
        }`}
      >
        <div className="text-xs mb-2 opacity-70 font-medium">
          {isUser ? "You" : "DocuMind"}
        </div>

        <p className="whitespace-pre-wrap leading-relaxed text-sm">
          {message.content}
        </p>

        {!isUser && message.chunks_used && (
          <div className="mt-4 pt-3 border-t border-white/10 text-xs text-slate-400">
            Used {message.chunks_used} relevant chunks
            {message.used_chunk_indexes?.length > 0 && (
              <span> · Indexes: {message.used_chunk_indexes.join(", ")}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;