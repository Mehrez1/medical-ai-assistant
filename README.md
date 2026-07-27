🏥 MedGuide AI — Intelligent Medical Triage Assistant

Academic Project — Generative AI Module, 2026

An AI-powered medical triage assistant that analyzes patient-reported symptoms, assesses urgency levels, and recommends the appropriate healthcare professional — built on a Retrieval-Augmented Generation (RAG) pipeline and a large language model agent.

Overview

MedGuide AI helps patients understand their symptoms before seeking care. Through a guided conversational interface, the system:

Detects symptoms from free-text input using NLP techniques
Assesses urgency on a four-level scale (Mild → Normal → Urgent → Emergency)
Recommends a specialist from 16 categories of healthcare professionals
Generates a medical summary the patient can share with a doctor
Grounds its reasoning in a curated medical knowledge base via RAG, reducing hallucination risk

Disclaimer: This tool is an academic prototype and does not replace professional medical advice, diagnosis, or treatment. Users should always consult a qualified healthcare provider.

Key Features
Feature	Description
🩺 Symptom Detection	Extracts symptoms, severity, and duration from natural language
⚠️ Urgency Assessment	Classifies cases across 4 severity levels based on symptom combinations
👨‍⚕️ Specialist Routing	Matches patients to 1 of 16 professional categories (GP, ER, specialists)
📚 RAG-Grounded Answers	Retrieves relevant medical context before generating any recommendation
📝 Shareable Summary	Produces a downloadable report for the consulting physician
💬 Conversational Memory	Maintains context across the dialogue for coherent follow-ups
Architecture
User Input (free text)
        │
        ▼
┌───────────────────────┐
│  Symptom Extraction    │  NLP / pattern matching
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  RAG Retrieval         │  ChromaDB + HuggingFace embeddings
│  (relevant medical docs)│
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Agent Tools           │  evaluate_urgency() · orient_to_specialist()
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  LLM Reasoning (Groq)  │  Llama 3.3 70B — contextualized generation
└──────────┬────────────┘
           ▼
    Structured Output
  (Urgency · Specialist · Summary)

The system is orchestrated with LangChain, which coordinates symptom analysis, tool calls, retrieval, and generation into a single conversational agent.

Tech Stack
Layer	Technology	Purpose
LLM	Groq API — Llama 3.3 70B (fallback: Mixtral 8x7B)	Medical reasoning and natural-language generation
Orchestration	LangChain	Agent logic, tool routing, conversation memory
Retrieval	ChromaDB 0.4.17	Vector store for medical knowledge base
Embeddings	HuggingFace all-MiniLM-L6-v2 (384-dim)	Semantic search over medical documents
Interface	Streamlit 1.28.1	Interactive web front end
Config	python-dotenv	Environment variable management
How It Works — Example

Input:

"High fever 40°C, difficulty breathing, extreme fatigue"

Pipeline output:

Field	Value
Detected symptoms	fever, breathing difficulty, fatigue
Urgency level	🔴 Urgent
Recommended action	Emergency services
Specialist	Emergency room

A milder case:

"Slight cough, nothing serious"

Field	Value
Urgency level	🟢 Mild
Specialist	General practitioner
Advice	Rest and monitor symptoms
Knowledge Base

The RAG layer is grounded in a curated medical dataset covering:

18 documented medical conditions
100+ associated symptoms
13+ symptom categories (fever, cough, pain, etc.)
16 professional specialist categories for routing
Project Status
 LangChain-based conversational agent
 RAG pipeline with ChromaDB + embeddings
 Urgency classification engine
 Specialist routing logic
 Full Streamlit interface
 Dual LLM support (Groq / Replicate)
 Error handling and logging
What This Project Demonstrates
Generative AI applied to medical reasoning
Agentic design — decision-making combining tools and LLM inference
RAG for grounding LLM output in domain-specific knowledge
Modular architecture separating agent logic, retrieval, tools, and UI
NLP for structured information extraction from free text
License

This project was developed for academic purposes as part of a Generative AI course (2026).

Author

Built by Mehrez1.
