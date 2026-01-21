import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from scipy.spatial.distance import cosine
import random
import requests

# Page Configuration
st.set_page_config(page_title="Semantic Mystery Game", page_icon="🔮")

# Noun Lists
COMMON_NOUNS = [
    "Airplane",
    "Apple",
    "Backpack",
    "Bicycle",
    "Camera",
    "Coffee",
    "Computer",
    "Diamond",
    "Dolphin",
    "Elephant",
    "Forest",
    "Guitar",
    "Hammer",
    "Hospital",
    "Island",
    "Jungle",
    "Kitchen",
    "Laptop",
    "Library",
    "Mountain",
    "Notebook",
    "Ocean",
    "Orange",
    "Piano",
    "Pizza",
    "Planet",
    "Rainbow",
    "Restaurant",
    "River",
    "Rocket",
    "School",
    "Submarine",
    "Telephone",
    "Telescope",
    "Umbrella",
    "Volcano",
    "Waterfall",
    "Window",
    "Zebra",
]

if "hint_history" not in st.session_state:
    st.session_state.hint_history = []

hint_count = len(st.session_state.hint_history)

max_hints = 3


# Load Model
@st.cache_resource
def load_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


model = load_model()

api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

endpoint_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    temperature=0.7,
    huggingfacehub_api_token=api_token,
)

llm = ChatHuggingFace(llm=endpoint_llm)

# Inizialization session state
if "target_word" not in st.session_state:
    st.session_state.target_word = random.choice(COMMON_NOUNS).lower()
    st.session_state.history = []
    st.session_state.win = False
    st.session_state.gave_up = False


# Game Logic


def submit_guess():
    guess = st.session_state.current_guess.lower().strip()
    if guess and not st.session_state.win:
        # Words to vector
        v_target = model.embed_query(st.session_state.target_word)
        v_guess = model.embed_query(guess)

        # Calculating score
        score = 1 - cosine(v_target, v_guess)

        # Add to history
        if not any(h["word"] == guess for h in st.session_state.history):
            st.session_state.history.append({"word": guess, "score": score})

        # Sorting history
        st.session_state.history = sorted(
            st.session_state.history, key=lambda x: x["score"], reverse=True
        )

        # Winning condition
        if score > 0.80 or guess == st.session_state.target_word:
            st.session_state.win = True
            st.session_state.gave_up = False

    # Reset input column
    st.session_state.current_guess = ""


def get_ai_hint(target_word):
    API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
    headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACEHUB_API_TOKEN']}"}

    prompt = f"Give a one-sentence cryptic tsundere riddle for the word: '{target_word}'. Don't mention the word."

    payload = {
        "inputs": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "parameters": {"temperature": 0.8, "max_new_tokens": 100},
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        output = response.json()
        # Mengambil teks hasil generate saja
        hint = output[0]["generated_text"].split("assistant\n")[-1].strip()
        return hint
    except Exception as e:
        # List pesan ngambek Oracle
        annoyed_remarks = [
            f"Error: {e}"
            # "Hmph! I'm not in the mood to talk to you right now. Go away!",
            # "The void is too noisy... Don't disturb my meditation!",
            # "Tch, do you think my wisdom is free? Try again when you're less annoying.",
            # "I've said enough for today. My throat hurts from explaining things to idiots.",
        ]
        return random.choice(annoyed_remarks)


def reset_game():
    st.session_state.target_word = random.choice(COMMON_NOUNS).lower()

    st.session_state.history = []

    st.session_state.hint_history = []

    st.session_state.win = False
    st.session_state.gave_up = False

    st.rerun()


# User Interface
st.title("🔮 Semantic Mystery")

with st.expander("📖 How to Play & Game Rules"):
    st.markdown(
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

st.write("---")


# Sidebar
with st.sidebar:
    st.header("Menu")
    if st.button("🔄 Change new word"):
        reset_game()
        st.rerun()

    st.write("---")
    if st.checkbox("Cheat Mode (Show answer)"):
        st.info(f"The answer is: {st.session_state.target_word}")

# Show progress bar and history
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

# Input User
st.text_input(
    "Input your guess:",
    key="current_guess",
    on_change=submit_guess,
    disabled=st.session_state.win,
)

# Give up button
if not st.session_state.win:
    if st.button("🏳️ Give Up"):
        st.session_state.win = True
        st.session_state.gave_up = True
        st.rerun()

# Oracle Button
if st.button("🔮 Seek Oracle Guidance", disabled=(hint_count >= max_hints)):
    found_hint = False
    for i in range(3):
        with st.spinner(f"Whispering to the void... (Trial {i+1}/3)"):
            hint = get_ai_hint(st.session_state.target_word)
            if "Error" not in hint:
                # append to history
                st.session_state.hint_history.append(hint)
                found_hint = True
                st.rerun()
                break

    if not found_hint:
        st.error("Tch! The void is too noisy right now. Try again.")

if hint_count < max_hints:
    st.caption(f"You have {max_hints - hint_count} guidance tokens left.")
else:
    st.caption("🚫 *The Oracle is no longer listening.*")

if st.session_state.hint_history:
    st.markdown("---")
    st.markdown("### 📜 The Oracle's Ledger")

    # Showing all hint
    with st.expander(f"📜 Oracle's Whispers ({hint_count}/{max_hints})"):
        for h in reversed(st.session_state.hint_history):
            with st.chat_message("assistant", avatar="🔮"):
                st.markdown(
                    f'<div class="oracle-box-style">"{h}"</div>', unsafe_allow_html=True
                )

# Endgame
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
