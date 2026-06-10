import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";
import { createServer as createViteServer } from "vite";

// Load environment configurations
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Middleware securely parsing JSON request envelopes
app.use(express.json({ limit: "50mb" }));

// Initialize Google GenAI client securely on server-side
const geminiApiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;

// Secure guard for Gemini Client
const ai = geminiApiKey 
  ? new GoogleGenAI({
      apiKey: geminiApiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    })
  : null;

// Secure API endpoint for querying RAG context using Gemini
app.post("/api/rag/query", async (req, res) => {
  try {
    const { question, context } = req.body;

    if (!question) {
      return res.status(400).json({ error: "Missing 'question' parameter inside request body." });
    }

    if (!ai) {
      return res.status(500).json({ 
        error: "Google Gemini API key is not configured. Please add the GEMINI_API_KEY inside Settings > Secrets." 
      });
    }

    // Build strict grounding prompts to avoid hallucinations
    const promptTemplate = `You are a professional RAG assistant answering user questions based solely on the provided context retrieved from an uploaded PDF.

Context:
${context || "No context retrieved."}

Question:
${question}

Instructions:
1. Answer the question using ONLY the provided context. Do not use outside knowledge, general training assumptions, or hallucinate.
2. Be objective, precise, and direct.
3. If the answer cannot be found in the provided context, you MUST reply EXACTLY:
"The uploaded PDF does not contain enough information to answer this question."
Do not attempt to explain further, suggest guesses, or formulate a speculative answer.

Answer:`;

    // Query high-speed, high-efficiency Gemini 3.5 Flash Model
    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: promptTemplate,
      config: {
        temperature: 0.0, // Force strict adherence to context facts
      }
    });

    return res.json({ 
      answer: response.text || "The uploaded PDF does not contain enough information to answer this question." 
    });

  } catch (error: any) {
    console.error("Gemini RAG Query Error:", error);
    return res.status(500).json({ 
      error: `An error occurred while calling the Gemini API: ${error.message || error}` 
    });
  }
});

// Configure Vite middleware or serve built static output
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
    console.log("Vite Development Server middleware mounted.");
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
    console.log("Production static build delivery mounted.");
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Fullstack application running on http://localhost:${PORT}`);
  });
}

startServer();
