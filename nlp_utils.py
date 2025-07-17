import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

# --- Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# --- Keyword Extraction ---
def extract_keywords(text, top_n=15):
    try:
        vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english')
        X = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        return list(keywords)
    except Exception:
        return []

# --- Sentiment Analysis ---
def sentiment_scores(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores  # dict: {'neg':..., 'neu':..., 'pos':..., 'compound':...}

# --- Readability ---
def readability_score(text):
    try:
        score = textstat.flesch_reading_ease(text)
        return score
    except Exception:
        return None

# --- Core Section Analysis ---
SECTION_CRITERIA = [
    {
        'name': 'The Problem',
        'keywords': ['problem', 'challenge', 'pain point', 'unmet need'],
        'tip': 'Clearly state the problem or unmet need your startup addresses.'
    },
    {
        'name': 'The Solution',
        'keywords': ['solution', 'product', 'platform', 'our technology', 'we solve'],
        'tip': 'Describe your solution and how it addresses the problem.'
    },
    {
        'name': 'Market Size (TAM/SAM/SOM)',
        'keywords': ['market size', 'tam', 'sam', 'som', 'billion', 'million', 'industry'],
        'tip': 'Quantify the market opportunity (TAM/SAM/SOM).' 
    },
    {
        'name': 'Product/Demo',
        'keywords': ['how it works', 'demo', 'product features', 'technology'],
        'tip': 'Showcase your product, demo, or technology.'
    },
    {
        'name': 'Traction & Metrics',
        'keywords': ['traction', 'users', 'revenue', 'growth', 'mrr', 'arr', 'kpi', 'metrics'],
        'tip': 'Highlight traction, growth, and key metrics.'
    },
    {
        'name': 'Business Model',
        'keywords': ['business model', 'monetization', 'pricing', 'how we make money'],
        'tip': 'Explain how your startup makes money.'
    },
    {
        'name': 'Competitive Landscape',
        'keywords': ['competitors', 'competition', 'unique advantage', 'moat', 'differentiator'],
        'tip': 'Describe your competitors and your unique advantage.'
    },
    {
        'name': 'The Team',
        'keywords': ['team', 'founders', 'ceo', 'cto', 'advisors', 'experience'],
        'tip': 'Introduce your core team and their expertise.'
    },
    {
        'name': 'The Ask & Use of Funds',
        'keywords': ['ask', 'seeking', 'raising', 'investment', 'use of funds'],
        'tip': 'State your funding ask and how you will use the funds.'
    },
]

def analyze_sections(text):
    strengths = []
    weaknesses = []
    actionable_tips = []
    points = 0
    for section in SECTION_CRITERIA:
        found = False
        for kw in section['keywords']:
            # Use word boundary for single words, substring for phrases
            if (len(kw.split()) == 1 and re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE)) or \
               (len(kw.split()) > 1 and kw in text.lower()):
                found = True
                break
        if found:
            points += 1
            strengths.append(section['name'])
        else:
            weaknesses.append(section['name'])
            actionable_tips.append(section['tip'])
    score = round((points / len(SECTION_CRITERIA)) * 10, 1)
    return score, strengths, weaknesses, actionable_tips 