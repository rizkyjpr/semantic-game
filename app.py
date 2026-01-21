import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from scipy.spatial.distance import cosine
import random
import requests
import textwrap

from logic import submit_guess, get_score, get_ai_hint, reset_game, COMMON_NOUNS

# --- CONFIGURATION ---
st.set_page_config(page_title="Semantic Mystery Game", page_icon="🔮")


# --- MODEL LOADING ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_oracle_model(api_token):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="conversational",
        huggingfacehub_api_token=api_token,
    )
    return ChatHuggingFace(llm=llm_endpoint)


model_embed = load_embedding_model()

api_token = st.secrets["api_token"]
oracle_llm = load_oracle_model(api_token)


# --- INITIALIZATION SESSION STATE ---
if "target_word" not in st.session_state:
    st.session_state.target_word = random.choice(COMMON_NOUNS).lower()
    st.session_state.history = []
    st.session_state.hint_history = []
    st.session_state.win = False
    st.session_state.gave_up = False

# --- DYNAMIC VARIABLES (Calculated on every rerun) ---
hint_count = len(st.session_state.hint_history)
MAX_HINTS = 3

# --- USER INTERFACE ---
st.title("🔮 Semantic Mystery")

with st.expander("📖 How to Play & Game Rules"):
    help_text = textwrap.dedent(
        """
    ### *Can you bridge the gap between thought and essence?*

    In this world, we do not care about letters or spelling. We care about **meaning**. 

    **How to Play:**
    1. **Submit a Noun**: Enter an English noun. The AI calculates the meaning distance between your word and the hidden target.
    2. **Decode the Score**:
        * 🧊 **0% - 30% (Cold)**: You are drifting in the void. Your word shares no soul with the target.
        * 🕯️ **31% - 70% (Warm)**: You've caught a scent. The context is similar, but the essence is different.
        * 🔥 **71% - 99% (Hot)**: You are standing at the doorstep! You are remarkably close.
    3. **Consult the Oracle**: Stuck? Ask the Oracle for a cryptic hint. Just be warned—she doesn't like being disturbed, and her tongue is as sharp as her mind.
    """
    )

    st.markdown(help_text)

st.write("---")


# SIDEBAR
with st.sidebar:
    st.header("Menu")
    if st.button("🔄 Change new word"):
        reset_game()
        st.rerun()

    st.write("---")
    if st.checkbox("Cheat Mode (Show answer)"):
        st.info(f"The answer is: {st.session_state.target_word}")

# SHOW PROGRESS & HISTORY
if st.session_state.history:
    best_score = st.session_state.history[0]["score"]
    st.metric("Highest Score", f"{best_score:.2%}")
    st.progress(min(max(float(best_score), 0.0), 1.0))
    st.write("")

    with st.expander(f"📜 View Guess History"):
        for h in st.session_state.history:
            # kedekatan?
            if h["score"] > 0.7:
                label = " 🔥 (Hot) — "
            elif h["score"] > 0.4:
                label = " 🟠 (Warm) — "
            else:
                label = " ❄️ (Cold) — "

            st.write(f"{label} **{h['word']}** — {h['score']:.2%}")

    st.write("---")

# INPUT USER
col_input, col_btn = st.columns([4, 1])

with col_input:
    st.text_input(
        "Input your guess:",
        key="current_guess",
        on_change=lambda: submit_guess(model_embed),
        disabled=st.session_state.win,
        label_visibility="collapsed",
    )

with col_btn:
    if st.button("Submit", use_container_width=True, disabled=st.session_state.win):
        submit_guess(model_embed)


# GIVEUP BUTTON
if not st.session_state.win:
    if st.button("🏳️ Give Up"):
        st.session_state.win = True
        st.session_state.gave_up = True
        st.rerun()

# ORACLE BUTTON
if st.button("🔮 Seek Oracle Guidance", disabled=(hint_count >= MAX_HINTS)):
    found_hint = False
    last_error = "Unknown Error"
    for i in range(3):
        with st.spinner(f"Whispering to the void... (Trial {i+1}/3)"):
            hint = get_ai_hint(oracle_llm, st.session_state.target_word)
            if "Error" not in hint:
                # append to history
                st.session_state.hint_history.append(hint)
                found_hint = True
                st.rerun()
                break
            else:
                last_error = hint

    if not found_hint:
        st.error(f"The Oracle is silent. Reason: {last_error}")

if hint_count < MAX_HINTS:
    st.caption(f"You have {MAX_HINTS - hint_count} guidance tokens left.")
else:
    st.caption("🚫 *The Oracle is no longer listening.*")

if st.session_state.hint_history:
    st.markdown("---")
    st.markdown("### 📜 The Oracle's Ledger")

    # SHOWING ALL HINT
    with st.expander(f"📜 Oracle's Whispers ({hint_count}/{MAX_HINTS})"):
        for h in reversed(st.session_state.hint_history):
            with st.chat_message("assistant", avatar="🔮"):
                st.markdown(
                    f'<div class="oracle-box-style">"{h}"</div>', unsafe_allow_html=True
                )

# ENDGAME
if st.session_state.win:
    if st.session_state.gave_up:
        st.warning(
            f"You gave up! The answer is: **{st.session_state.target_word.upper()}**"
        )
    else:
        st.balloons()
        st.success(
            f"🎊 You Won! The answer is: **{st.session_state.target_word.upper()}**"
        )

    if st.button("Play again"):
        reset_game()
        st.rerun()
