from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from nlp_utils import preprocess_text, extract_keywords, sentiment_scores, readability_score, analyze_sections, SECTION_CRITERIA
from text_extractor import extract_text
import re

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only! Restrict in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_pitch(file: UploadFile = File(...)):
    print("File received:", file.filename)
    # Save uploaded file temporarily
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Extract text using extract_text function
    with open(tmp_path, 'rb') as f:
        text = extract_text(f, suffix.lower())
    os.remove(tmp_path)

    # Run analysis
    preprocessed = preprocess_text(text)
    keywords = extract_keywords(preprocessed)
    sentiment = sentiment_scores(text)
    readability = readability_score(text)
    section_score, strengths, weaknesses, tips = analyze_sections(text)

    # Generate investor feedback and maturity level (simple logic for now)
    if section_score >= 8:
        maturity = 'Series A Ready'
        investor_feedback = 'Impressive! Your pitch covers all key areas investors look for.'
    elif section_score >= 6:
        maturity = 'Seed Ready'
        investor_feedback = 'Good job! Strengthen your business model and traction sections for more impact.'
    elif section_score >= 4:
        maturity = 'Pre-seed Ready'
        investor_feedback = 'You have the basics. Clarify your problem, solution, and market size for investors.'
    else:
        maturity = 'Beginner'
        investor_feedback = 'Your pitch is missing several key sections. Focus on clearly stating the problem, solution, and team.'

    # Section-wise breakdown
    section_breakdown = []
    for section in SECTION_CRITERIA:
        found = False
        for kw in section['keywords']:
            if (len(kw.split()) == 1 and re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE)) or \
               (len(kw.split()) > 1 and kw in text.lower()):
                found = True
                break
        strength = 5 if found else 2
        missing = [] if found else [f"Missing: {section['name']}"]
        suggestion = section['tip']
        section_breakdown.append({
            'name': section['name'],
            'strength': strength,
            'missing': missing,
            'suggestion': suggestion
        })

    return {
        "keywords": keywords,
        "sentiment": sentiment,
        "readability": readability,
        "section_score": section_score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "tips": tips,
        "raw_text": text,
        "investor_feedback": investor_feedback,
        "maturity_level": maturity,
        "sections": section_breakdown
    } 