
# app.py

import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from text_extractor import extract_text
from nlp_utils import preprocess_text, analyze_sections, readability_score, sentiment_scores, SECTION_CRITERIA, extract_keywords
import joblib
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from supabase import create_client, Client

# Load .env
load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL') or ""
SUPABASE_KEY = os.getenv('SUPABASE_KEY') or ""
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="PitchPerfect AI", layout="wide")

# --- Session state for theme and auth ---
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = {'name': '', 'email': '', 'id': ''}
if 'selected_analysis' not in st.session_state:
    st.session_state.selected_analysis = None

# --- Theme CSS ---
if st.session_state.dark_mode:
    st.markdown('''<style>body, .block-container, .sidebar .sidebar-content {background: #18191A !important; color: #F8FAFC !important;} .sidebar .sidebar-content {border-right: 1.5px solid #222 !important;} </style>''', unsafe_allow_html=True)
else:
    st.markdown('''<style>.block-container {padding-top: 2rem;} .sidebar .sidebar-content {background: #f8fafc;} .sidebar .sidebar-content {border-right: 1.5px solid #eaf4ff;} .sidebar .sidebar-content {min-width: 270px;} </style>''', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:0.7em;margin-bottom:1.5em;'>
        <span style='font-size:2.1rem;'>🚀</span>
        <span style='font-size:1.3rem;font-weight:700;color:#007acc;'>PitchPerfect AI</span>
                </div>
    <div style='color:#888;font-size:0.98rem;margin-bottom:2.2em;'>AI-powered pitch deck analysis</div>
    """, unsafe_allow_html=True)
    if st.session_state.logged_in:
        if st.button("Upload New Deck", use_container_width=True):
            st.session_state.selected_analysis = None
        st.markdown("<div style='margin:1.5em 0 0.7em 0;font-weight:600;'>Past Analyses</div>", unsafe_allow_html=True)
        # Fetch analyses from Supabase for this user
        user_id = st.session_state.user.get('id', '')
        analyses = []
        if user_id:
            res = sb.table('analyses').select('*').eq('user_id', user_id).order('date', desc=True).execute()
            if res.data:
                analyses = res.data
        if analyses:
            for i, a in enumerate(analyses):
                label = f"{a['filename']} ({a['date'][:16]})"
                if st.button(label, key=f"analysis_{i}"):
                    st.session_state.selected_analysis = a
        else:
            st.info("No analyses yet", icon="⏳")
        st.markdown(f"""
        <div style='margin-top:2.5em;display:flex;align-items:center;gap:0.7em;background:#f5faff;padding:0.7em 1em;border-radius:10px;'>
            <span style='font-size:1.3rem;background:#eaf4ff;border-radius:50%;padding:0.3em 0.7em;color:#007acc;'>👤</span>
                            <div>
                <div style='font-weight:600;'>{st.session_state.user['name']}</div>
                <div style='font-size:0.95rem;color:#888;'>{st.session_state.user['email']}</div>
            </div>
                </div>
        """, unsafe_allow_html=True)
        if st.button("Sign Out", key="signout_btn"):
            sb.auth.sign_out()
            st.session_state.logged_in = False
            st.session_state.user = {'name': '', 'email': '', 'id': ''}
            st.session_state.selected_analysis = None
            st.rerun()
    st.markdown("<div style='margin:1.5em 0 0.7em 0;'><span style='color:#888;'>🌙</span> Dark Theme</div>", unsafe_allow_html=True)
    if st.button("Toggle Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# --- Main Upload Area ---
if not st.session_state.logged_in:
    st.stop()

st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;margin-top:2.5em;'>
        <div style='font-size:2.5rem;color:#007acc;margin-bottom:0.7em;'>⬆️</div>
        <div style='font-size:1.25rem;font-weight:600;margin-bottom:0.5em;'>PitchPerfect AI – Your Personal Pitch Deck Analyst</div>
        <div style='font-size:1.08rem;color:#007acc;margin-bottom:2em;'>Upload your deck and get AI-powered insights instantly</div>
        <div style='border:2px dashed #b3d8fd;border-radius:18px;padding:3em 4em;text-align:center;background:#f8fafc;margin-bottom:1.5em;'>
            <div style='font-size:2.5rem;color:#007acc;margin-bottom:0.7em;'>⬆️</div>
            <div style='font-size:1.2rem;font-weight:600;margin-bottom:0.5em;'>Drop your pitch deck here</div>
            <div style='color:#888;margin-bottom:1em;'>or</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Browse Files", type=["pdf", "pptx", "docx", "txt"], label_visibility="collapsed")

st.markdown("""
            </div>
        <div style='margin-top:1em;background:#fffbe6;border:1.5px solid #ffe066;padding:1em 1.5em;border-radius:10px;color:#b38f00;font-size:1.08rem;max-width:420px;'>
            <b>Supported formats:</b> PDF, PPTX, DOCX, TXT files up to 50MB
            </div>
    </div>
""", unsafe_allow_html=True)

# --- Show selected past analysis if any ---
if st.session_state.selected_analysis:
    a = st.session_state.selected_analysis
    st.subheader(f"Analysis for {a['filename']} ({a['date'][:16]})")
    st.write(a['summary'])
    st.download_button("Download Analysis as Text", a['summary'], file_name=f"{a['filename']}_analysis.txt")
    st.stop()

# --- If file uploaded, show analysis (rest of your analysis code below this) ---
if uploaded_file:
    filetype = os.path.splitext(uploaded_file.name)[1].lower()
    filesize = uploaded_file.size / 1024
    with st.container():
        st.markdown(f"<div style='background:#f5faff;padding:1em 1.5em;border-radius:12px;display:inline-block;margin-bottom:1em;'>"
                    f"<b>Filename:</b> {uploaded_file.name} &nbsp; | &nbsp; <b>Type:</b> {filetype.upper()} &nbsp; | &nbsp; <b>Size:</b> {filesize:.1f} KB"
                    f"</div>", unsafe_allow_html=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=filetype) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    text = extract_text(tmp_path, filetype)
    os.unlink(tmp_path)
    if not text:
        st.error("Could not extract text from the uploaded file.")
    else:
        st.subheader("Extracted Text Preview")
        st.code(text[:1000] + ("..." if len(text) > 1000 else ""), language=None)
        processed = preprocess_text(text)
        score, strengths, weaknesses, tips = analyze_sections(text)
        read_score = readability_score(text)
        sentiment = sentiment_scores(text)
        keywords = extract_keywords(text)
        # (Rest of the analysis and chart code as before)
        # ...
        # Save analysis summary for history in Supabase
        summary = f"Pitch Analysis for {uploaded_file.name}\n\nSection Coverage: {score}/10\nReadability: {read_score}\nSentiment: {sentiment}\n\nStrengths: {', '.join(strengths)}\nWeaknesses: {', '.join(weaknesses)}\nTips: {', '.join(tips)}\nKeywords: {', '.join(keywords)}\n"
        user_id = st.session_state.user.get('id', '')
        if user_id:
            sb.table('analyses').insert({
                'user_id': user_id,
                'filename': uploaded_file.name,
                'date': datetime.now().isoformat(),
                'summary': summary
            }).execute()
