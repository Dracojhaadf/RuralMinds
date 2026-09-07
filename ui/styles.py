import streamlit as st

def inject_custom_css():
    """Inject custom styling, typography, and animations into the Streamlit app."""
    st.markdown("""
        <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Chat Input Styling */
        .stChatInputContainer {
            padding-bottom: 20px;
        }
        
        /* Message Styling */
        .stChatMessage {
            background-color: transparent !important;
            border: none !important;
        }
        
        /* User Message */
        div[data-testid="stChatMessage"]:nth-child(odd) {
            background-color: transparent;
        }
        
        /* Assistant Message */
        div[data-testid="stChatMessage"]:nth-child(even) {
            background-color: #444654; /* ChatGPT dark grey */
        }
        
        /* Avatar Styling */
        .stChatMessage .stChatMessageAvatar {
            background-color: #10a37f; /* OpenAI Green */
            color: white;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #202123;
            color: white;
        }
        
        /* General Button Styling */
        .stButton>button {
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.1);
            background-color: #343541;
            color: white;
            transition: all 0.2s;
        }
        
        .stButton>button:hover {
            border-color: #10a37f;
            color: #10a37f;
        }
        
        /* Primary Button */
        div.stButton > button[kind="primary"] {
            background-color: #10a37f;
            border: none;
            color: white;
        }
        
        div.stButton > button[kind="primary"]:hover {
            background-color: #1a7f64;
        }
        
        /* Text Inputs */
        .stTextInput > div > div > input {
            background-color: #40414f;
            color: white;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 4px;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #10a37f;
            box-shadow: 0 0 0 1px #10a37f;
        }

        /* THINKING ANIMATION */
        .thinking-container {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 10px;
        }
        
        .thinking-text {
            font-size: 14px;
            color: #b4b4b4;
            font-style: italic;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 0.5; }
            50% { opacity: 1; }
            100% { opacity: 0.5; }
        }

        /* Square with Circle Orbit Animation */
        .loader-box {
            position: relative;
            width: 20px;
            height: 20px;
            border: 2px solid #10a37f;
            border-radius: 2px;
        }
        
        .loader-circle {
            position: absolute;
            width: 6px;
            height: 6px;
            background-color: white;
            border-radius: 50%;
            top: -3px;
            left: -3px;
            animation: orbit 2s linear infinite;
        }
        
        @keyframes orbit {
            0% { top: -3px; left: -3px; }
            25% { top: -3px; left: 17px; }
            50% { top: 17px; left: 17px; }
            75% { top: 17px; left: -3px; }
            100% { top: -3px; left: -3px; }
        }
        </style>
    """, unsafe_allow_html=True)
