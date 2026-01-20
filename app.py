import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from scipy.spatial.distance import cosine
import random

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Semantic Mystery Game", page_icon="🧩")

# 2. Daftar Kata Benda Umum (Bisa kamu tambah sesukamu)
COMMON_NOUNS = [
    "Airplane", "Apple", "Backpack", "Bicycle", "Camera", "Coffee", "Computer", 
    "Diamond", "Dolphin", "Elephant", "Forest", "Guitar", "Hammer", "Hospital", 
    "Island", "Jungle", "Kitchen", "Laptop", "Library", "Mountain", "Notebook", 
    "Ocean", "Orange", "Piano", "Pizza", "Planet", "Rainbow", "Restaurant", 
    "River", "Rocket", "School", "Submarine", "Telephone", "Telescope", "Umbrella", 
    "Volcano", "Waterfall", "Window", "Zebra"
]

# 3. Load Model Embedding (Lokal & Gratis)
@st.cache_resource
def load_model():
    # Menggunakan model all-MiniLM-L6-v2 yang ringan untuk MacBook M1
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

model = load_model()

# 4. Inisialisasi Session State (Memori Game)
if 'target_word' not in st.session_state:
    st.session_state.target_word = random.choice(COMMON_NOUNS).lower()
    st.session_state.history = []
    st.session_state.win = False
    st.session_state.gave_up = False

# 5. Fungsi Logika Game
def submit_guess():
    guess = st.session_state.current_guess.lower().strip()
    if guess and not st.session_state.win:
        # Proses merubah kata menjadi vektor angka
        v_target = model.embed_query(st.session_state.target_word)
        v_guess = model.embed_query(guess)
        
        # Hitung skor kedekatan (0 sampai 1)
        score = 1 - cosine(v_target, v_guess)
        
        # Simpan ke riwayat jika belum pernah ditebak
        if not any(h['word'] == guess for h in st.session_state.history):
            st.session_state.history.append({"word": guess, "score": score})
        
        # Urutkan riwayat dari skor tertinggi
        st.session_state.history = sorted(st.session_state.history, key=lambda x: x['score'], reverse=True)

        # Cek jika tebakan benar (toleransi skor > 0.80)
        if score > 0.80 or guess == st.session_state.target_word:
            st.session_state.win = True
            st.session_state.gave_up = False
    
    # Reset kolom input
    st.session_state.current_guess = ""

def reset_game():
    st.session_state.target_word = random.choice(COMMON_NOUNS).lower()
    st.session_state.history = []
    st.session_state.win = False
    st.session_state.gave_up = False

# --- TAMPILAN UI ---
st.title("🧩 Semantic Mystery")
st.write("Tebak kata benda rahasia! AI akan menilai seberapa dekat makna tebakanmu.")

# Sidebar untuk Kontrol
with st.sidebar:
    st.header("Menu")
    if st.button("🔄 Ganti Kata Baru"):
        reset_game()
        st.rerun()
    
    st.write("---")
    if st.checkbox("Cheat Mode (Lihat Jawaban)"):
        st.info(f"Jawabannya adalah: {st.session_state.target_word}")

# Input User
st.text_input(
    "Masukkan tebakanmu (Bahasa Inggris):", 
    key="current_guess", 
    on_change=submit_guess,
    disabled=st.session_state.win
)

# Tombol Give Up
if not st.session_state.win:
    if st.button("🏳️ Menyerah (Give Up)"):
        st.session_state.win = True
        st.session_state.gave_up = True
        st.rerun()

# 6. Menampilkan Hasil
if st.session_state.win:
    if st.session_state.gave_up:
        st.warning(f"Kamu menyerah! Jawabannya adalah: **{st.session_state.target_word.upper()}**")
    else:
        st.balloons()
        st.success(f"🎊 MENANG! Jawabannya adalah: **{st.session_state.target_word.upper()}**")
    
    if st.button("Main Lagi"):
        reset_game()
        st.rerun()

# Menampilkan Progress Bar & Riwayat
if st.session_state.history:
    st.write("---")
    best_score = st.session_state.history[0]['score']
    st.metric("Skor Tertinggi", f"{best_score:.2%}")
    st.progress(min(max(float(best_score), 0.0), 1.0))

    st.write("### Riwayat Tebakan:")
    for h in st.session_state.history:
        # Memberikan label suhu berdasarkan skor
        if h['score'] > 0.7:
            label = "🔥 Panas!"
        elif h['score'] > 0.4:
            label = "🟠 Hangat"
        else:
            label = "❄️ Dingin"
            
        st.write(f"{label} **{h['word']}** — {h['score']:.2%}")