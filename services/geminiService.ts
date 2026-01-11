import { GoogleGenAI } from "@google/genai";
import { TrainingStats } from "../types";

const getAIClient = () => {
  if (!process.env.API_KEY) {
    throw new Error("API Key not found");
  }
  return new GoogleGenAI({ apiKey: process.env.API_KEY });
};

export const explainArchitecture = async (layers: any[], stats: TrainingStats) => {
  try {
    const ai = getAIClient();
    const prompt = `
      You are an expert AI tutor for neural networks.
      Explain the current neural network architecture and its training status to a student.
      
      Architecture:
      ${JSON.stringify(layers)}
      
      Current Training Stats:
      Epoch: ${stats.epoch}
      Loss: ${stats.loss.toFixed(4)}
      Accuracy: ${(stats.accuracy * 100).toFixed(1)}%
      
      Keep the explanation concise, encouraging, and highlight one interesting thing about this configuration or current performance.
      Do not use markdown formatting excessively, keep it readable for a chat bubble.
    `;

    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: prompt,
    });
    
    return response.text;
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "I'm having trouble connecting to my brain right now. Please check your API key.";
  }
};

export const chatWithTutor = async (history: { role: string, text: string }[], userMessage: string, contextData: any) => {
  try {
    const ai = getAIClient();
    
    // Construct chat history with system context
    const chat = ai.chats.create({
      model: 'gemini-3-flash-preview',
      config: {
        systemInstruction: `You are NeuroVis, a helpful and friendly 3D neural network visualization assistant. 
        Your goal is to teach users about neural networks based on the current simulation they are looking at.
        
        Current Simulation Context:
        ${JSON.stringify(contextData)}
        
        Answer user questions specifically about this network or general neural network concepts. Keep answers short and conversational.`
      }
    });

    // We can't easily pre-fill history in the SDK 'create' method in the same way as some other APIs, 
    // but we can just send the new message if we treat it as a fresh turn or send history in the prompt if needed.
    // For simplicity in this demo, we will just send the user message to a new chat session, 
    // or manually reconstruct history if we wanted to maintain multi-turn state robustly. 
    // Given the SDK structure, let's just send the message directly with the system instruction providing context.

    const result = await chat.sendMessage({ message: userMessage });
    return result.text;

  } catch (error) {
    console.error("Gemini Chat Error:", error);
    return "Sorry, I couldn't process that request.";
  }
};