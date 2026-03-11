"""
Mood-Map Travel Curator
Psychology-Based Travel Recommendation System for Jammu & Kashmir
Powered by Google Gemini AI
"""

import streamlit as st
import google.generativeai as genai
import re
import os

# ============================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================

st.set_page_config(
    page_title="Mood-Map Travel Curator",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@400;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* Hero Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 20px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(100, 100, 255, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(255, 100, 200, 0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #e0c3fc, #8ec5fc, #f5d0fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    position: relative;
}
.hero-subtitle {
    color: rgba(255,255,255,0.65);
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.5px;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.7);
    padding: 0.3rem 1rem;
    border-radius: 50px;
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.1);
    position: relative;
}

/* Info Cards */
.info-card {
    background: linear-gradient(135deg, rgba(30, 27, 56, 0.9), rgba(20, 18, 40, 0.95));
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
.info-card h3 {
    color: #c4b5fd;
    font-size: 1.1rem;
    margin-bottom: 0.8rem;
    font-weight: 600;
}
.info-card p {
    color: rgba(255,255,255,0.6);
    font-size: 0.92rem;
    line-height: 1.7;
}

/* Mood Buttons */
.mood-btn-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    justify-content: center;
    margin: 1rem 0;
}

/* Result Card */
.result-card {
    background: linear-gradient(145deg, rgba(30, 27, 56, 0.95), rgba(15, 12, 41, 0.98));
    border: 1px solid rgba(142, 197, 252, 0.15);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    animation: fadeInUp 0.6s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-card h2 {
    background: linear-gradient(135deg, #e0c3fc, #8ec5fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Playfair Display', serif;
    margin-bottom: 0.5rem;
}
.result-card h3 {
    color: #c4b5fd;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 0.5rem;
    margin-top: 1.5rem;
}

/* Matched emotions tag */
.emotion-tag {
    display: inline-block;
    background: linear-gradient(135deg, rgba(142, 197, 252, 0.15), rgba(224, 195, 252, 0.15));
    border: 1px solid rgba(142, 197, 252, 0.25);
    color: #c4b5fd;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-size: 0.85rem;
    margin: 0.2rem;
    font-weight: 500;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #1a1744);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c4b5fd;
}

/* Footer */
.footer {
    text-align: center;
    color: rgba(255,255,255,0.3);
    font-size: 0.78rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 3rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# MODULE 3: EMOTION-LOCATION KNOWLEDGE BASE (Dataset Loader)
# ============================================================

@st.cache_data
def load_emotion_dataset(filepath: str) -> dict:
    """
    Parses jk_emotion_locations.txt into a structured dictionary.
    Returns: { "peaceful": { "description": "...", "locations": [{"name": ..., "detail": ...}, ...] }, ... }
    """
    dataset = {}
    current_emotion = None
    current_description = ""
    current_locations = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                # Detect "Emotion: XYZ"
                if line.lower().startswith("emotion:"):
                    # Save previous emotion block
                    if current_emotion:
                        dataset[current_emotion] = {
                            "description": current_description,
                            "locations": current_locations,
                        }
                    current_emotion = line.split(":", 1)[1].strip().lower()
                    current_description = ""
                    current_locations = []

                elif line.lower().startswith("description:"):
                    current_description = line.split(":", 1)[1].strip()

                elif line.lower().startswith("locations:"):
                    # Header line, skip
                    continue

                elif line.startswith("- "):
                    # Location entry like "- Yusmarg Meadows: description..."
                    entry = line[2:]
                    if ":" in entry:
                        name, detail = entry.split(":", 1)
                        current_locations.append(
                            {"name": name.strip(), "detail": detail.strip()}
                        )
                    else:
                        current_locations.append(
                            {"name": entry.strip(), "detail": ""}
                        )

            # Save last emotion block
            if current_emotion:
                dataset[current_emotion] = {
                    "description": current_description,
                    "locations": current_locations,
                }
    except FileNotFoundError:
        st.error(f"❌ Dataset file not found: `{filepath}`")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")

    return dataset


# ============================================================
# MODULE 2: EMOTION PROCESSING MODULE
# ============================================================

# Synonym mapping for fuzzy matching
EMOTION_SYNONYMS = {
    "peaceful": ["peaceful", "peace", "calm", "serene", "tranquil", "quiet", "still", "relaxed", "relax", "soothing", "zen"],
    "adventure": ["adventure", "adventurous", "thrill", "exciting", "excitement", "adrenaline", "daring", "bold", "explore", "extreme"],
    "reflection": ["reflection", "reflective", "introspective", "thoughtful", "thinking", "contemplate", "meditate", "ponder", "philosophical"],
    "romantic": ["romantic", "romance", "love", "intimate", "couple", "honeymoon", "date", "valentine", "passion", "together"],
    "spiritual": ["spiritual", "divine", "sacred", "holy", "prayer", "soul", "faith", "religious", "blessing", "devotion", "temple", "shrine"],
    "joyful": ["joyful", "joy", "happy", "happiness", "cheerful", "celebration", "celebrate", "fun", "delight", "bliss", "elated", "excited"],
    "overwhelmed": ["overwhelmed", "stressed", "stress", "burnout", "burnt", "exhausted", "tired", "anxious", "anxiety", "overworked", "drained", "pressure", "tense"],
    "curious": ["curious", "curiosity", "discover", "learn", "knowledge", "explore", "culture", "history", "heritage", "intellectual", "wonder"],
    "lonely": ["lonely", "alone", "isolated", "disconnected", "homesick", "miss", "solitary", "companionship", "warmth", "connection"],
    "energetic": ["energetic", "energy", "active", "dynamic", "lively", "vigorous", "vital", "sporty", "fitness", "athletic", "pumped"],
}


def preprocess_emotion(text: str) -> str:
    """Step 7: Preprocess emotional input — lowercase, remove punctuation, clean up."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    return text


def extract_emotion_keywords(text: str) -> list:
    """Extract individual words as potential emotional keywords."""
    stop_words = {
        "i", "me", "my", "myself", "we", "our", "feel", "feeling", "want",
        "need", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "a", "an", "the",
        "and", "but", "or", "for", "nor", "not", "so", "yet", "to", "of",
        "in", "on", "at", "by", "with", "from", "up", "out", "off",
        "some", "very", "really", "just", "like", "right", "now", "today",
        "bit", "little", "lot", "much", "more", "most", "too", "also",
    }
    words = text.split()
    return [w for w in words if w not in stop_words and len(w) > 1]


# ============================================================
# MODULE 4: RETRIEVAL MODULE
# ============================================================

def match_emotions(keywords: list, dataset: dict) -> list:
    """
    Step 8–9: Match user keywords against emotion categories using synonym mapping.
    Returns list of matched emotion category keys.
    """
    matched = []
    scores = {}

    for emotion_key, synonyms in EMOTION_SYNONYMS.items():
        score = 0
        for kw in keywords:
            if kw in synonyms:
                score += 2  # Exact synonym match
            else:
                # Partial match (e.g., "peacefulness" contains "peace")
                for syn in synonyms:
                    if syn in kw or kw in syn:
                        score += 1
                        break
        if score > 0 and emotion_key in dataset:
            scores[emotion_key] = score

    # Sort by score descending, take top matches
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    matched = [em for em, sc in sorted_emotions[:3]]  # Top 3

    # Fallback: if no match, return the most general/popular categories
    if not matched and dataset:
        matched = list(dataset.keys())[:2]

    return matched


def retrieve_locations(matched_emotions: list, dataset: dict) -> str:
    """Build a context string of all matched locations for the LLM prompt."""
    context_parts = []
    for emotion in matched_emotions:
        if emotion in dataset:
            info = dataset[emotion]
            locations_str = "\n".join(
                [f"  • {loc['name']}: {loc['detail']}" for loc in info["locations"]]
            )
            context_parts.append(
                f"Emotion Category: {emotion.title()}\n"
                f"Description: {info['description']}\n"
                f"Recommended Locations:\n{locations_str}"
            )
    return "\n\n".join(context_parts)


# ============================================================
# MODULE 5: PROMPT CONSTRUCTION MODULE
# ============================================================

def build_prompt(user_input: str, location_context: str, matched_emotions: list) -> str:
    """Construct the LLM prompt with emotional context and retrieved locations."""
    emotions_str = ", ".join([em.title() for em in matched_emotions])
    prompt = f"""You are a psychology-based travel advisor specializing in Jammu & Kashmir, India.
A traveler has described their emotional state as: "{user_input}"

Based on psychological analysis, their emotions align with these categories: {emotions_str}

Here are curated travel destinations that match their emotional needs:

{location_context}

Please provide a comprehensive, empathetic travel recommendation that includes:

1. **Emotional Understanding**: Acknowledge the traveler's emotional state and validate their feelings (2-3 sentences).

2. **Top 3 Recommended Destinations**: For each destination, explain:
   - Why this place psychologically suits their current emotional state
   - What specific experiences they should seek there
   - Best time to visit and practical tips

3. **Psychological Reasoning**: Explain the psychology behind why these environments help with their emotional state (e.g., nature therapy, adventure therapy, contemplative environments).

4. **Mini Itinerary**: A suggested 3-day travel itinerary incorporating the top destinations.

5. **Wellness Tips**: 2-3 mindfulness or wellness practices the traveler can combine with their trip.

Make your response warm, professional, and psychologically insightful. Use markdown formatting for readability."""

    return prompt


# ============================================================
# MODULE 6: AI RECOMMENDATION ENGINE (Gemini)
# ============================================================

def get_gemini_response(prompt: str, api_key: str) -> str:
    """Send prompt to Google Gemini and return the generated response."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ **API Error:** {str(e)}\n\nPlease check your API key and internet connection."


# ============================================================
# MODULE 1 & 7: USER INTERFACE & OUTPUT DISPLAY
# ============================================================

def main():
    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown("## 🔑 API Configuration")
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="Enter your Gemini API key...",
            help="Get your free API key from https://aistudio.google.com/apikey",
        )
        st.markdown("---")
        st.markdown("### 📖 How It Works")
        st.markdown(
            """
1. **Express your mood** — type how you feel or pick a quick mood
2. **AI analyzes** — your emotion is matched to curated destinations
3. **Get recommendations** — personalized travel plan with psychological insights
            """
        )
        st.markdown("---")
        st.markdown(
            """
### 🌍 About
**Mood-Map Travel Curator** uses psychology-based emotional mapping
combined with Google Gemini AI to recommend
travel destinations in **Jammu & Kashmir**.
            """
        )
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:rgba(255,255,255,0.35); font-size:0.75rem;'>"
            "Powered by Google Gemini AI<br>© 2026 Mood-Map Travel Curator"
            "</div>",
            unsafe_allow_html=True,
        )

    # ---------- HERO HEADER ----------
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-badge">✨ AI-Powered Psychology Travel</div>
            <div class="hero-title">🗺️ Mood-Map Travel Curator</div>
            <div class="hero-subtitle">
                Discover destinations in Jammu &amp; Kashmir that match your soul.<br>
                Tell us how you feel — we'll tell you where to go.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- LOAD DATASET ----------
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jk_emotion_locations.txt")
    dataset = load_emotion_dataset(dataset_path)

    if not dataset:
        st.error("⚠️ Could not load the emotion-location dataset. Please ensure `jk_emotion_locations.txt` is in the project directory.")
        st.stop()

    # ---------- EMOTION INPUT ----------
    st.markdown(
        '<div class="info-card">'
        "<h3>💭 How Are You Feeling?</h3>"
        "<p>Describe your emotional state in your own words, or click a quick mood below. "
        "Our AI will analyze your feelings and recommend the perfect destinations.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Quick mood buttons
    moods = {
        "🧘 Peaceful": "I feel peaceful and want calm nature",
        "⛰️ Adventure": "I want adventure and thrill",
        "🪞 Reflective": "I feel reflective and need time to think",
        "💕 Romantic": "I feel romantic and want a beautiful getaway",
        "🙏 Spiritual": "I seek spiritual peace and sacred experiences",
        "🎉 Joyful": "I feel joyful and want to celebrate life",
        "😮‍💨 Stressed": "I feel overwhelmed and stressed, I need escape",
        "🔍 Curious": "I feel curious and want to explore culture and history",
        "🤝 Connected": "I feel lonely and want warmth and community",
        "⚡ Energetic": "I feel energetic and want active experiences",
    }

    # Initialize session state
    if "mood_input" not in st.session_state:
        st.session_state.mood_input = ""
    if "recommendation" not in st.session_state:
        st.session_state.recommendation = None
    if "matched_emotions" not in st.session_state:
        st.session_state.matched_emotions = []

    st.markdown("**Quick Moods:**")
    cols = st.columns(5)
    for i, (label, value) in enumerate(moods.items()):
        with cols[i % 5]:
            if st.button(label, key=f"mood_{i}", use_container_width=True):
                st.session_state.mood_input = value
                st.session_state.recommendation = None

    # Text input
    user_input = st.text_area(
        "Or describe your feelings in your own words:",
        value=st.session_state.mood_input,
        height=100,
        placeholder="e.g., I feel stressed and overwhelmed. I need silence and nature to recharge...",
        key="emotion_text_area",
    )

    # ---------- GENERATE BUTTON ----------
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🌟 Get Travel Recommendations",
            use_container_width=True,
            type="primary",
        )

    # ---------- PROCESSING & RESULTS ----------
    if generate_clicked:
        if not api_key:
            st.warning("🔑 Please enter your Google Gemini API key in the sidebar to continue.")
            st.stop()

        if not user_input or len(user_input.strip()) < 3:
            st.warning("💭 Please describe your feelings or select a quick mood above.")
            st.stop()

        # Step 7: Preprocess
        processed = preprocess_emotion(user_input)

        # Extract keywords
        keywords = extract_emotion_keywords(processed)

        # Step 8: Match emotions
        matched = match_emotions(keywords, dataset)
        st.session_state.matched_emotions = matched

        # Step 9: Retrieve locations
        location_context = retrieve_locations(matched, dataset)

        # Step 10-11: Build prompt
        prompt = build_prompt(user_input, location_context, matched)

        # Step 12-13: Call AI
        with st.spinner("🔮 Analyzing your emotions and curating destinations..."):
            response = get_gemini_response(prompt, api_key)

        st.session_state.recommendation = response

    # ---------- DISPLAY RESULTS ----------
    if st.session_state.recommendation:
        # Show matched emotions
        if st.session_state.matched_emotions:
            st.markdown("#### 🎯 Matched Emotional Categories")
            tags_html = " ".join(
                [f'<span class="emotion-tag">{em.title()}</span>' for em in st.session_state.matched_emotions]
            )
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("")

        # Show AI Response
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("## 🌄 Your Personalized Travel Recommendations")
        st.markdown(st.session_state.recommendation)
        st.markdown("</div>", unsafe_allow_html=True)

        # Explore another mood
        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Explore Another Mood", use_container_width=True):
                st.session_state.mood_input = ""
                st.session_state.recommendation = None
                st.session_state.matched_emotions = []
                st.rerun()

    # ---------- FOOTER ----------
    st.markdown(
        '<div class="footer">'
        "🗺️ Mood-Map Travel Curator · Psychology-Based AI Travel Recommendations<br>"
        "Destinations curated for Jammu &amp; Kashmir · Powered by Google Gemini"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
