import streamlit as st
import requests
from ddgs import DDGS
from bs4 import BeautifulSoup
import webbrowser
import difflib
import time

# Optional: use scikit-learn for better similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# Configure page
st.set_page_config(
    page_title="STUDY ANALYZER",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for neon styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stApp {
        background: transparent;
    }
    
    .neon-text {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        background: linear-gradient(90deg, #ff00cc, #3333ff, #00ccff, #ff00cc);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        -webkit-animation: gradient 3s ease infinite;
        animation: gradient 3s ease infinite;
        text-align: center;
        font-size: 3.5rem !important;
        margin-bottom: 2rem;
    }
    
    @-webkit-keyframes gradient {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    .neon-card {
        background: rgba(15, 15, 35, 0.8);
        border: 1px solid #00ccff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 15px rgba(0, 204, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .neon-card:hover {
        box-shadow: 0 0 25px rgba(0, 204, 255, 0.6);
        transform: translateY(-2px);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #ff00cc, #3333ff);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(255, 0, 204, 0.5);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 2px solid #00ccff;
    }
    
    .match-score {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff00cc, #3333ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .reading-boy {
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Reading boy ASCII art with neon colors
reading_boy = """
<div class="reading-boy">
<pre style="color: #00ccff; font-family: monospace; font-size: 12px;">
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⢀⣠⣴⠾⠛⠉⠉⠉⠉⠛⠳⣦⣄⡀⠀⠀⠀⠀
    ⠀⠀⠀⢀⣴⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⣦⡀⠀⠀
    ⠀⠀⢀⣾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⡄⠀
    ⠀⢀⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⡀
    ⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀
    ⠀⠸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡿⠀
    ⠀⠀⢿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡇⠀
    ⠀⠀⠘⢷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡾⠁⠀
    ⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠟⠁⠀⠀
    ⠀⠀⠀⠀⠀⠈⠛⢶⣤⣀⠀⠀⠀⠀⣀⣤⠶⠛⠁⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠛⠛⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⠀
    📚 <span style="color: #ff00cc;">Analyzing content...</span> 🔍
</pre>
</div>
"""

def ddgs_search(query, max_result=5):
    try:
        with DDGS() as ddgs:
            result = ddgs.text(query, max_result=max_result)
            return result
    except Exception as e:
        st.error(f"An error occurred during DDGS search: {e}")
        return None

def content_extraction(url):
    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/58.0.3029.110 Safari/537.3'
            )
        }
        with requests.Session() as session:
            resp = session.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')

            # remove unwanted tags
            for element in soup(['script', 'style', 'footer', 'nav', 'aside']):
                element.decompose()

            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
            return content
    except Exception as e:
        st.error(f"An error occurred during content extraction: {e}")
        return None

def compute_similarity(text_a, text_b):
    """Return similarity percentage between two texts (0-100)."""
    if not text_a or not text_b:
        return 0.0
    try:
        if _HAS_SKLEARN:
            vect = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf = vect.fit_transform([text_a, text_b])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return float(max(0.0, min(1.0, sim))) * 100.0
        else:
            a = ' '.join(text_a.split())
            b = ' '.join(text_b.split())
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            return float(ratio) * 100.0
    except Exception:
        try:
            tokens_a = set(text_a.lower().split())
            tokens_b = set(text_b.lower().split())
            inter = tokens_a.intersection(tokens_b)
            denom = max(1, min(len(tokens_a), len(tokens_b)))
            return (len(inter) / denom) * 100.0
        except Exception:
            return 0.0

def main():
    # Header with neon text
    st.markdown('<div class="neon-text">STUDY ANALYZER</div>', unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Search Configuration")
        
        # Input fields
        query = st.text_input(
            "**What to search for:**",
            placeholder="Enter your search query...",
            help="This will be used to search the web"
        )
        
        user_text = st.text_area(
            "**Text to match against:**",
            placeholder="Paste the text you want to compare with search results...",
            height=150,
            help="The search results will be ranked by similarity to this text"
        )
        
        max_results = st.slider("**Number of results to analyze:**", 3, 10, 5)
        
        analyze_btn = st.button("🚀 ANALYZE SEARCH RESULTS", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Results Panel")
        
        if analyze_btn:
            if not query or not user_text:
                st.warning("⚠️ Please fill in both search query and text to match!")
            else:
                with st.spinner("🔄 Searching and analyzing content..."):
                    # Show reading boy animation
                    st.markdown(reading_boy, unsafe_allow_html=True)
                    
                    # Perform search
                    search_results = ddgs_search(query, max_result=max_results) or []
                    
                    if not search_results:
                        st.error("❌ No search results found. Please try a different query.")
                        return
                    
                    scored_results = []
                    progress_bar = st.progress(0)
                    
                    for i, result in enumerate(search_results):
                        url = result.get('href')
                        title = result.get('title') or url
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(search_results))
                        
                        with st.expander(f"🔍 Analyzing: {title[:80]}...", expanded=False):
                            st.write(f"**URL:** `{url}`")
                            
                            content = content_extraction(url)
                            if not content:
                                st.warning("❌ Could not extract content")
                                continue
                            
                            # Limit content size
                            doc = content[:8000]
                            score = compute_similarity(user_text, doc)
                            scored_results.append((score, url, title, doc))
                            
                            st.write(f"**Similarity Score:** `{score:.2f}%`")
                    
                    progress_bar.empty()
                    
                    if not scored_results:
                        st.error("❌ No searchable content found in results.")
                        return
                    
                    # Sort and display results
                    scored_results.sort(reverse=True, key=lambda x: x[0])
                    
                    # Best match
                    best_score, best_url, best_title, _ = scored_results[0]
                    
                    st.markdown("---")
                    st.markdown("### 🏆 Best Match")
                    st.markdown(f'<div class="neon-card">', unsafe_allow_html=True)
                    st.markdown(f'**🎯 Score:** <span class="match-score">{best_score:.2f}%</span>', unsafe_allow_html=True)
                    st.markdown(f'**📖 Title:** {best_title}')
                    st.markdown(f'**🔗 URL:** `{best_url}`')
                    
                    if st.button("🌐 OPEN BEST MATCH IN BROWSER", key="open_best"):
                        webbrowser.open(best_url)
                        st.success("✅ Opening best match in your browser!")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Top matches
                    st.markdown("### 📈 Top Matches")
                    for i, (score, url, title, _) in enumerate(scored_results[:5]):
                        st.markdown(f'<div class="neon-card">', unsafe_allow_html=True)
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f'**{i+1}. {title}**')
                            st.markdown(f'`{url}`')
                        with col_b:
                            st.markdown(f'<span class="match-score">{score:.2f}%</span>', unsafe_allow_html=True)
                            if st.button("🌐 Open", key=f"open_{i}"):
                                webbrowser.open(url)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.info("👆 Enter your search details and click 'ANALYZE SEARCH RESULTS' to begin!")
            st.markdown(reading_boy, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()