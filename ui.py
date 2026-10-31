"""Streamlit front end.   streamlit run ui.py   (API must be running)"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Audio Retrieval", page_icon="🎧", layout="wide")
st.title("🎧 Audio Retrieval")

# ---- Sidebar: upload + library -------------------------------------------
with st.sidebar:
    st.header("Add audio")
    up = st.file_uploader("Audio file", type=["mp3", "wav", "m4a", "flac", "ogg"])
    title = st.text_input("Title (optional)")
    speaker = st.text_input("Speaker (optional)", value="Unknown")
    topic = st.text_input("Topic (optional)", value="uncategorized")
    if st.button("Upload & index", disabled=up is None):
        r = requests.post(f"{API}/upload", files={"file": (up.name, up.getvalue())},
                          data={"title": title, "speaker": speaker, "topic": topic})
        st.success("Uploaded. Transcribing in the background...") if r.ok else st.error(r.text)

    st.header("Library")
    if st.button("Refresh"):
        st.rerun()
    try:
        for ep in requests.get(f"{API}/files", timeout=10).json():
            icon = {"ready": "✅", "processing": "⏳", "failed": "❌"}.get(ep["status"], "•")
            st.write(f"{icon} **{ep['title']}**  ({ep['segment_count']} chunks)")
            if ep["error"]:
                st.caption(ep["error"])
    except requests.RequestException:
        st.error("Can't reach the API. Is uvicorn running?")

# ---- Main: ask ------------------------------------------------------------
question = st.text_input("Ask a question about your audio")
if question:
    with st.spinner("Searching..."):
        data = requests.post(f"{API}/query", json={"question": question}, timeout=180).json()
    st.markdown(f"### Answer\n{data['answer']}")
    if data.get("intent"):
        st.caption(f"Query type: {data['intent']}")
    for s in data["sources"]:
        m, sec = divmod(s["timestamp_sec"], 60)
        with st.expander(f"{s['title']} — {s['speaker']} @ {m}:{sec:02d}"):
            st.write(s["text"])
            audio = requests.get(f"{API}/files/{s['episode_id']}/audio", timeout=60)
            if audio.ok:
                st.audio(audio.content, start_time=s["timestamp_sec"])
