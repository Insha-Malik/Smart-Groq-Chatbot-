# smart_groq_chatbot

import os, re
from uuid import uuid4
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# ─────────────────────────────
# Load API Key
# ─────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in .env")
    st.stop()

# ─────────────────────────────
# Page setup
# ─────────────────────────────
st.set_page_config("Smart Groq Chatbot 🤖", "🤖", layout="wide")
st.title("Smart Groq Chatbot 🤖")

# ─────────────────────────────
# Sessions
# ─────────────────────────────
if "sessions" not in st.session_state: st.session_state.sessions = {}
if "current_session" not in st.session_state: st.session_state.current_session = "default"
sid = st.session_state.current_session

if sid not in st.session_state.sessions:
    st.session_state.sessions[sid] = {"history": InMemoryChatMessageHistory(), "title": "New Chat"}

# ─────────────────────────────
# Helper: Generate short 2-word filename
# ─────────────────────────────
def generate_short_filename(text, max_words=2):
    text = re.sub(r"[^\w\s]","",text.lower())  # lowercase & remove punctuation
    words = text.split()
    stopwords = {"i","you","me","is","the","a","an","can","could","would","should","it","how","my","your","please","want","to","know"}
    meaningful = [w for w in words if w not in stopwords]
    if not meaningful:
        meaningful = words
    short_name = "_".join(meaningful[:max_words])
    return short_name if short_name else "chat"

# ─────────────────────────────
# Current session history
# ─────────────────────────────
history = st.session_state.sessions[sid]["history"]

# ─────────────────────────────
# Sidebar controls
# ─────────────────────────────
with st.sidebar:
    st.subheader("⚙ Controls")

    # New session
    if st.button("New Conversation"):
        new_id = f"session_{uuid4()}"
        st.session_state.sessions[new_id] = {"history": InMemoryChatMessageHistory(), "title": "New Chat"}
        st.session_state.current_session = new_id
        st.session_state.pop("first_user_msg", None)

    # Previous chats with horizontal Clear/Delete buttons
    st.markdown("### 💬 Previous Chats")
    for k, v in st.session_state.sessions.items():
        st.markdown(f"{v['title']}")
        cols = st.columns([1,1])
        with cols[0]:
            if st.button("Clear", key=f"clear_{k}"):
                st.session_state.sessions[k]["history"] = InMemoryChatMessageHistory()
                st.session_state.sessions[k]["title"] = "New Chat"
                if st.session_state.current_session == k:
                    st.session_state.pop("first_user_msg", None)
        with cols[1]:
            if st.button("Delete", key=f"delete_{k}"):
                del st.session_state.sessions[k]
                if st.session_state.current_session == k:
                    st.session_state.current_session = "default"
                    st.session_state.pop("first_user_msg", None)

    # AI & display settings
    max_tokens = {"Short":100,"Medium":200,"Long":300}[st.selectbox("Answer Length", ["Short","Medium","Long"], 1)]
    temperature = {"Precise":0.2,"Balanced":0.5,"Creative":0.9}[st.radio("Response Style", ["Precise","Balanced","Creative"], 1)]
    dark_mode = st.radio("Theme", ["Light", "Dark"], 0)
    model_name = st.selectbox("AI Model", ["deepseek-r1-distill-llama-70b","gemma2-9b-it","llama-3.1-8b-instant"], 2)
    system_prompt = st.text_area("System Instructions", "You are a professional smart assistant. Answer clearly and concisely.")

    # ─────────────────────────────
    # Export / Save Conversation in Sidebar
    # ─────────────────────────────
    st.markdown("### 💾 Save Conversation")
    if history.messages:
        if "first_user_msg" not in st.session_state:
            # Use first human message to generate short name
            user_msgs = [m.content for m in history.messages if getattr(m,"type", None)=="human"]
            if user_msgs:
                st.session_state.first_user_msg = generate_short_filename(user_msgs[0])
            else:
                st.session_state.first_user_msg = f"chat_{sid}"

        # Editable filename input
        chat_name = st.text_input("File Name:", value=st.session_state.first_user_msg)

        # Build plain text export
        plain_text = ""
        for m in history.messages:
            role = "User" if getattr(m,"type", None)=="human" else "Assistant"
            plain_text += f"{role}: {m.content}\n\n"

        # Download button
        st.download_button(
            "Save Conversation",
            data=plain_text,
            file_name=chat_name,
            mime="text/plain",
            key=f"download_{sid}"
        )

# ─────────────────────────────
# Dark/Light CSS
# ─────────────────────────────
dark_css = """
<style>
body, .stApp, .main, .block-container{background:#0E1117 !important; color:#FAFAFA !important;}
[data-testid="stSidebar"]{background:#1B1F2B !important; color:#FAFAFA !important;}
.stChatMessage, .stChatMessage *{background:#222 !important; color:#FAFAFA !important;}
button, .stButton button{background:#444 !important; color:#FAFAFA !important; border:1px solid #555 !important;}
input, textarea, select, option, .stTextInput input, .stTextArea textarea, .stSelectbox, .stSlider, .stRadio, .stCheckbox{background:#262730 !important; color:#FAFAFA !important; border:1px solid #444 !important;}
h1,h2,h3,h4,h5,h6,p,span,label{color:#FAFAFA !important;}
a{color:#81A1C1 !important;}
table, th, td{background:#1E2028 !important; color:#FAFAFA !important; border-color:#444 !important;}
::placeholder{color:#AAA !important;}
</style>
"""
light_css = dark_css.replace("#0E1117","#FFF").replace("#1B1F2B","#F5F5F5").replace("#222","#F1F0F0").replace("#444","#E0E0E0").replace("#262730","#FFF").replace("#FAFAFA","#000").replace("#81A1C1","#1E90FF").replace("#1E2028","#F9F9F9").replace("#AAA","#555")
st.markdown(dark_css if dark_mode=="Dark" else light_css, unsafe_allow_html=True)

# ─────────────────────────────
# Render previous messages
# ─────────────────────────────
for m in history.messages:
    role = getattr(m,"type",None) or getattr(m,"role","")
    st.chat_message("user" if role=="human" else "assistant").write(m.content)

# ─────────────────────────────
# Handle user input
# ─────────────────────────────
user_input = st.chat_input("Ask me anything...")
if user_input:
    st.chat_message("user").write(user_input)
    if "first_user_msg" not in st.session_state:
        st.session_state.first_user_msg = generate_short_filename(user_input)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            llm = ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens)
            prompt = ChatPromptTemplate.from_messages([("system","{system_prompt}"), MessagesPlaceholder("history"),("human","{input}")])
            chain = prompt | llm | StrOutputParser()
            chat_with_history = RunnableWithMessageHistory(
                chain,
                lambda s_id: st.session_state.sessions[s_id]["history"],
                input_messages_key="input",
                history_messages_key="history"
            )
            response_text = chat_with_history.invoke({"input":user_input,"system_prompt":system_prompt},{"configurable":{"session_id":sid}})
        except Exception as e:
            st.error(f"Model error: {e}")
            response_text = ""
        typed = ""
        for ch in response_text: typed += ch; placeholder.markdown(typed)