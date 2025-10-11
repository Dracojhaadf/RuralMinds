NFS: AI-Powered Document Q&A and Collaboration Platform
<p align="center">
<img src="https://www.google.com/search?q=https://placehold.co/800x400/667eea/ffffff%3Ftext%3DNFS%2520Dashboard%26font%3Dsans" alt="NFS Application Dashboard">
</p>

NFS (Next-Generation File System) is a comprehensive web application built with Streamlit that transforms static documents into an interactive learning experience. It leverages a Retrieval-Augmented Generation (RAG) model to allow users to "chat" with their PDF documents, asking questions and receiving context-aware answers.

The platform also features a full-fledged community forum, creating a collaborative space where students can ask course-related questions and teachers can provide answers, fostering an interactive educational environment.

📋 Table of Contents
✨ Core Features

🛠️ Technology Stack

🚀 Getting Started

📖 How to Use

📂 Project File Structure

✨ Core Features
🤖 AI-Powered Document Q&A: Upload PDF documents and ask complex questions. The system retrieves relevant text chunks and provides concise, accurate answers based on the document's content.

🎓 Interactive Community Forum:

Students can post questions on a public forum for clarification on topics.

Teachers can browse and respond to student questions, building a persistent knowledge base.

Notifications: Teachers are notified in the sidebar about pending questions that need a response.

🔐 Role-Based Access Control: The application supports three distinct user roles with specific permissions:

Administrator: Manages all users and can create teacher accounts.

Teacher: Can upload and delete PDF documents, and answer questions in the forum.

Student: Can query documents and participate in the forum by asking questions.

📄 Content Management: An intuitive interface for teachers to upload and manage the library of documents available for Q&A.

🔒 Secure Authentication: A complete login, signup, and session management system ensures that data and permissions are handled securely.

🛠️ Technology Stack
Frontend: Streamlit

Backend & AI:

Vector Database: ChromaDB for efficient similarity search.

Embeddings: Sentence-Transformers (all-MiniLM-L6-v2) for creating text embeddings.

Text Processing: PyMuPDF (fitz) for PDF text extraction & NLTK for text chunking.

User & Forum Database: SQLite for structured data like user credentials, forum posts, and replies.

Language: Python 3.9+

🚀 Getting Started
Follow these instructions to set up and run the NFS platform on your local machine.

1. Prerequisites
Python 3.9 or higher

pip (Python package installer)

Git

2. Installation
Clone the repository:

git clone [https://github.com/your-username/nfs-rag-forum.git](https://github.com/your-username/nfs-rag-forum.git)
cd nfs-rag-forum

Create and activate a virtual environment:

Windows:

python -m venv venv
.\venv\Scripts\activate

macOS / Linux:

python3 -m venv venv
source venv/bin/activate

Install the required packages:

pip install -r requirements.txt

Download NLTK Data: The application will automatically attempt to download the punkt tokenizer on the first run.

3. Configuration
The default administrator credentials are set in app.py:

Username: administrator

Password: admin

It is highly recommended to change these for any production-like environment.

4. Running the Application
Once the installation is complete, start the Streamlit application with the following command:

streamlit run app.py

The application will be accessible in your web browser, typically at http://localhost:8501.

📖 How to Use
👤 Administrator
Navigate to the "Admin Login" page.

Use the default credentials (administrator/admin) to log in.

From the admin panel, you can view all registered users, delete users, and create new "Teacher" accounts.

👨‍🏫 Teacher
Log in with your teacher credentials.

In the sidebar, use the file uploader to add new PDF documents to the system.

Navigate to the Forum tab to view student questions. Click "View" to see a discussion and post your reply. A notification will appear in the sidebar if questions are unanswered.

🧑‍🎓 Student
Create a new account or log in.

In the Document Q&A tab, select a document from the dropdown menu and use the chat interface to ask questions about it.

In the Forum tab, ask a new question or view existing discussions.

📂 Project File Structure
This project is composed of a main Streamlit application and several supporting components.

nfs-rag-forum/
│
├─── 🚀 MAIN STREAMLIT APPLICATION
│    ├── 📄 app.py            # Main Streamlit UI and application logic
│    ├── ⚙️ backend.py         # Core logic for RAG, PDF processing, etc.
│    └── 🔑 auth.py            # User authentication & SQLite database management
│
├─── 📦 DATA & DATABASES
│    ├── 🗃️ users.db           # SQLite database for users and forum
│    ├── 📁 chroma_db/         # Directory for the ChromaDB vector database
│    ├── 📁 source_folder/    # Place PDFs here for bulk ingestion
│    ├── 📁 static/            # For storing static assets like videos
│    │   └── videos/
│    └── 📁 captions/          # Stores generated captions for videos
│
├─── 🛠️ SUPPORTING SCRIPTS & COMPONENTS
│    ├── 📜 ingest.py          # Standalone script to bulk-process PDFs
│    ├── 🌐 form.py            # A separate Flask app for student forms
│    ├── 📄 form-frontend.html # HTML templates for the Flask app
│    ├── 📄 data-view-frontend.html
│    └── 📄 frontend.html
│
├─── 🧪 TESTING
│    ├── 📁 test/
│    └── 📁 test_system/
│
├─── 🗑️ LEGACY (Replaced by SQLite DB)
│    ├── 📄 forum_db.json
│    ├── 📄 student_interactions.json
│    └── 📄 users_db.json
│
└─── ⚙️ CONFIGURATION
     ├── 📄 requirements.txt   # List of Python dependencies for pip
     └── 📖 README.md          # This file
