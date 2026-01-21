import requests
import random
import streamlit as st
from scipy.spatial.distance import cosine

# --- NOUN LIST ---
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


# --- GAME LOGIC ---
def get_score(model, target_word, guess):
    v_target = model.embed_query(target_word)
    v_guess = model.embed_query(guess)
    return 1 - cosine(v_target, v_guess)


def submit_guess(model):
    guess = st.session_state.current_guess.lower().strip()

    if not guess:
        return

    if not st.session_state.win:
        score = get_score(model, st.session_state.target_word, guess)

        if not any(h["word"] == guess for h in st.session_state.history):
            st.session_state.history.append({"word": guess, "score": score})

        st.session_state.history = sorted(
            st.session_state.history, key=lambda x: x["score"], reverse=True
        )

        if score > 0.80 or guess == st.session_state.target_word:
            st.session_state.win = True
            st.session_state.gave_up = False

    st.session_state.current_guess = ""


def get_ai_hint(llm_chain, target_word):
    system = (
        "You are the 'Semantic Oracle', a high-and-mighty tsundere goddess. "
        "Your task is to give a cryptic hint for a secret word without mentioning the word itself. "
        "Rules: "
        "1. Start with a classic tsundere remark (e.g., 'Hmph!', 'It's not like I want to help you!', 'Tch, so slow!'). "
        "2. Give a riddle that focuses on the essence, meaning, or usage of the word. "
        "3. Keep it to 1-2 sentences. "
        "4. Be insulting but helpful. "
        "5. Speak in English."
    )

    prompt = f"{system} Give a one-sentence cryptic tsundere riddle for the word: '{target_word}'. Don't mention the word."

    try:
        response = llm_chain.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def reset_game():
    st.session_state.target_word = random.choice(COMMON_NOUNS).lower()
    st.session_state.history = []
    st.session_state.hint_history = []
    st.session_state.win = False
    st.session_state.gave_up = False
    st.rerun()
