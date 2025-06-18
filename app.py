from __future__ import annotations

import io
from pathlib import Path
from typing import List

import librosa
import librosa.display  
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

#  configuration 
RAW_AUDIO_DIR = Path(
    "/Users/pietrocaprioglio/Documents/GitHub/AI-audio-visual-generative/test_audio_files"
)

DEFAULT_N_MELS = 128
DEFAULT_FIGSIZE = (10, 5)

st.set_page_config(
    page_title="Audio ▶︎ Spectrogram Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Audio ▶︎ Spectrogram Viewer")


#  helper functions 
def list_audio_files(folder: Path) -> List[Path]:
    """Return a sorted list of audio file paths in *folder*."""
    exts = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


@st.cache_data(show_spinner="Computing spectrogram…")
def load_audio_and_melspec(
    path: Path,
    n_mels: int = DEFAULT_N_MELS,
) -> bytes:
    """Load *path*, compute mel‑spectrogram, return it as PNG bytes."""
    y, sr = librosa.load(path)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, frameon=False, dpi=200)
    img = librosa.display.specshow(S_db, sr=sr, y_axis="log", x_axis="time", cmap="magma", ax=ax)
    ax.set_title('Mel Spectogram', fontsize=20)    ## Comment this line if you want to remove the title
    fig.colorbar(img, ax=ax, format=f'%0.2f')              ## Comment this line if you want to remove the axis

    # Save to buffer w/out borders
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


#  sidebar 
st.sidebar.header("Settings")

# Choose (or type) the folder with audio files
chosen_dir = st.sidebar.text_input("Audio folder")
audio_dir = Path(chosen_dir)
if not audio_dir.is_dir():
    st.sidebar.error("Directory not found — please enter an existing folder path.")
    st.stop()

files = list_audio_files(audio_dir)
if not files:
    st.sidebar.warning("No audio files found in the selected folder.")
    st.stop()

selected_file = st.sidebar.selectbox("Pick a track", files, format_func=lambda p: p.name)

n_mels = st.sidebar.slider("Number of mel bands", min_value=64, max_value=256, step=8, value=DEFAULT_N_MELS)

# main panel 
with st.container():
    st.subheader("Listen to your track")
    audio_bytes = selected_file.read_bytes()
    st.audio(audio_bytes, format="audio/wav")  # Streamlit guesses format from header

with st.container():
    st.subheader("Check the Mel‑spectrogram")
    spec_png = load_audio_and_melspec(selected_file, n_mels=n_mels)
    st.image(spec_png)

