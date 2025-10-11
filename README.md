# NFS: AI-Powered Document Q&A and Collaboration Platform

<p align="center">
  <img src="https://placehold.co/800x400/667eea/ffffff?text=NFS+Dashboard&font=sans" alt="NFS Application Dashboard">
</p>

NFS (Next-Generation File System) is a comprehensive web application built with **Streamlit** that transforms static documents into an interactive learning experience. It leverages a **Retrieval-Augmented Generation (RAG)** model to allow users to "chat" with their PDF documents, asking questions and receiving context-aware answers.

The platform also features a **full-fledged community forum**, creating a collaborative space where students can ask course-related questions and teachers can provide answers, fostering an interactive educational environment.

---

## 📋 Table of Contents
- [✨ Core Features](#-core-features)  
- [🛠️ Technology Stack](#️-technology-stack)  
- [🚀 Getting Started](#-getting-started)  
- [📖 How to Use](#-how-to-use)  
- [📂 Project File Structure](#-project-file-structure)  

---

## ✨ Core Features

### 🤖 AI-Powered Document Q&A
- Upload PDF documents and ask complex questions.
- The system retrieves relevant text chunks and provides concise, accurate answers based on the document's content.

### 🎓 Interactive Community Forum
- Students can post questions on a public forum for clarification on topics.  
- Teachers can browse and respond to student questions, building a persistent knowledge base.  
- Notifications: Teachers are notified in the sidebar about pending questions that need a response.

### 🔐 Role-Based Access Control
The application supports **three distinct user roles** with specific permissions:
- **Administrator**: Manages all users and can create teacher accounts.  
- **Teacher**: Can upload and delete PDF documents, and answer questions in the forum.  
- **Student**: Can query documents and participate in the forum by asking questions.

### 📄 Content Management
- An intuitive interface for teachers to upload and manage the library of documents available for Q&A.

### 🔒 Secure Authentication
- Complete login, signup, and session management system ensures data and permissions are handled securely.

---

## 🛠️ Technology Stack

**Frontend:** Streamlit  

**Backend & AI:**  
- **Vector Database:** ChromaDB for efficient similarity search  
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`) for creating text embeddings  
- **Text Processing:** PyMuPDF (`fitz`) for PDF text extraction & NLTK for text chunking  
- **User & Forum Database:** SQLite for structured data like user credentials, forum posts, and replies  
- **Language:** Python 3.9+  

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 or higher  
- pip (Python package installer)  
- Git  

### 2. Installation
```bash
git clone https://github.com/your-username/nfs-rag-forum.git
cd nfs-rag-forum
