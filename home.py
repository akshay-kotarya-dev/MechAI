import streamlit as st
import os
import dotenv
from llamaindex.indexing import load_index
from llamaindex.llm import connect_llm
from llamaindex.querying import query_index
from llamaindex.embeddings import set_emebed_model

dotenv.load_dotenv()
db_index_name = os.getenv("DB_INDEX_NAME")

st.set_page_config(
    page_title="MechAI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* ── Global ── */
.stApp { background: #212121 !important; color: #ececec; }
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #171717 !important;
    border-right: 1px solid #2f2f2f !important;
    min-width: 260px !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #ececec !important; }

/* Hide collapse/expand button entirely — sidebar is always open */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stSidebarNavItems"] { display: none !important; }

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: #2a2a2a !important;
    color: #ececec !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px !important;
    width: 100% !important;
    font-size: 0.88rem !important;
    padding: 10px 14px !important;
    text-align: left !important;
    margin-bottom: 4px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #333 !important;
    border-color: #555 !important;
}

/* ── Main background ── */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Chat container ── */
.chat-container {
    max-width: 720px;
    margin: 0 auto;
    padding: 24px 24px 180px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 12vh 24px 3rem;
}
.empty-state h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #ececec;
    margin-bottom: 6px;
}
.empty-state p { color: #8e8ea0; font-size: 0.95rem; margin-bottom: 2rem; }
.chips {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    max-width: 480px;
    margin: 0 auto;
}
.chip {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 12px;
    padding: 14px 16px;
    text-align: left;
    cursor: default;
    color: #c5c5d2;
    font-size: 0.84rem;
    line-height: 1.5;
}
.chip b { display: block; color: #ececec; font-size: 0.88rem; margin-bottom: 3px; }

/* ── Messages ── */
.msg-wrap { display: flex; gap: 14px; margin-bottom: 26px; align-items: flex-start; }
.msg-wrap.user { flex-direction: row-reverse; }

.av {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: 700;
    flex-shrink: 0; margin-top: 3px;
}
.av.u { background: #5436DA; color: #fff; }
.av.b { background: #19c37d; color: #fff; font-size: 1rem; }

.bubble {
    max-width: 80%;
    font-size: 0.93rem;
    line-height: 1.75;
    color: #ececec;
}
.bubble.u {
    background: #2f2f2f;
    border: 1px solid #3a3a3a;
    border-radius: 18px 18px 4px 18px;
    padding: 11px 16px;
}
.bubble.b { background: transparent; padding-top: 4px; }

/* ── Thinking dots ── */
.dots { display: flex; gap: 5px; padding: 10px 0; }
.d {
    width: 7px; height: 7px; border-radius: 50%;
    background: #8e8ea0;
    animation: bop 1.2s infinite ease-in-out;
}
.d:nth-child(2) { animation-delay:.2s }
.d:nth-child(3) { animation-delay:.4s }
@keyframes bop {
    0%,80%,100% { transform:translateY(0); opacity:.35; }
    40% { transform:translateY(-7px); opacity:1; }
}

/* ── Fixed input bar ── */
.input-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    z-index: 999;
    background: linear-gradient(to top, #212121 75%, transparent);
    padding: 12px 24px 20px;
}
.input-inner-wrap {
    max-width: 480px;
    margin: 0 auto;
    background: #2f2f2f;
    border: 1px solid #3f3f3f;
    border-radius: 14px;
    padding: 4px 8px 4px 16px;
    display: flex;
    align-items: center;
}
.input-footer {
    text-align: center;
    font-size: 0.7rem;
    color: #565869;
    margin-top: 8px;
    max-width: 480px;
    margin-left: auto; margin-right: auto;
}

/* ── Form & input overrides ── */
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
div[data-testid="stForm"] > div { gap: 0 !important; }

.stTextInput > label { display: none !important; }
.stTextInput > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.stTextInput input {
    background: transparent !important;
    border: none !important;
    color: #ececec !important;
    font-size: 0.95rem !important;
    padding: 12px 0 !important;
    box-shadow: none !important;
    caret-color: #ececec;
}
.stTextInput input::placeholder { color: #565869 !important; }

/* ── Send button — hidden, Enter key is used instead ── */
.stFormSubmitButton { display: none !important; }

/* Loading overlay */
.loading-screen {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 80vh; gap: 16px;
}
.loading-screen p { color: #8e8ea0; font-size: 0.9rem; }
.spin {
    width: 36px; height: 36px;
    border: 3px solid #2f2f2f;
    border-top-color: #19c37d;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── st.chat_message styling ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}
[data-testid="stChatMessageContent"] {
    background: transparent !important;
    color: #ececec !important;
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
}
/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: #2f2f2f !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 11px 16px !important;
}
/* Avatar icons */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    background: transparent !important;
    border: none !important;
}
/* LaTeX rendering */
.katex { color: #ececec !important; font-size: 1rem !important; }
.katex-display { overflow-x: auto; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline (cached after first load) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_pipeline():
    set_emebed_model()
    llm = connect_llm()
    index = load_index(db_index_name)
    return index, llm


with st.spinner("Starting MechAI — loading models & connecting to knowledge base..."):
    index, llm = init_pipeline()

# ── Session state ───────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ MechAI")
    st.divider()

    if st.button("✏️  New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.caption("MODEL")
    st.markdown("🟢 **Llama 3.3 · 70B · Groq**")

    st.divider()

    st.caption("KNOWLEDGE BASE")
    st.markdown("📚 **Tribology**")
    st.caption("I.M. Hutchings, 1st Ed.")

    st.divider()

    st.caption("POWERED BY")
    st.markdown("""
⚡ Groq API  
🔍 Pinecone  
🤗 HuggingFace  
🦙 LlamaIndex  
🎈 Streamlit
    """)


# ── Main content ─────────────────────────────────────────────────────────────
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("""
    <div class='empty-state'>
        <h1>⚙️ MechAI</h1>
        <p>Your intelligent mechanical engineering assistant</p>
        <div class='chips'>
            <div class='chip'><b>Reynolds equation</b>Explain hydrodynamic lubrication</div>
            <div class='chip'><b>Contact stress</b>Hertz contact formula for cylinders</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for turn in st.session_state.chat_history:
        # User message
        with st.chat_message("user"):
            st.markdown(turn['question'])

        # Bot reply or thinking dots
        with st.chat_message("assistant"):
            if turn["answer"] is None:
                st.markdown("""
                <div class='dots'>
                    <div class='d'></div><div class='d'></div><div class='d'></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(turn['answer'])

st.markdown("</div>", unsafe_allow_html=True)

# ── Resolve pending answer ────────────────────────────────────────────────
if st.session_state.chat_history and st.session_state.chat_history[-1]["answer"] is None:
    resp = query_index(index, st.session_state.chat_history[-1]["question"], llm)
    st.session_state.chat_history[-1]["answer"] = resp.response
    st.rerun()

# ── Input bar (fixed bottom) ──────────────────────────────────────────────
st.markdown("<div class='input-bar'><div class='input-inner-wrap'>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Message",
        placeholder="Message MechAI...  (Press Enter ↵ to send)",
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("Send", use_container_width=False)

st.markdown("""
    </div>
    <div class='input-footer'>MechAI may make mistakes — verify critical formulas independently.</div>
</div>
""", unsafe_allow_html=True)

if submitted and user_input.strip():
    st.session_state.chat_history.append({"question": user_input.strip(), "answer": None})
    st.rerun()
