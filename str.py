import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import io

try:
    import bm25s
    bm25_available = True
except ImportError:
    bm25_available = False
    st.warning("BM25S is not installed. BM25 search will be skipped.")

# Configure Streamlit page
st.set_page_config(
    page_title="FAQ Semantic Search",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def download_nltk_data():
    """Download NLTK data with caching"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

@st.cache_resource
def load_models():
    """Load and cache models"""
    try:
        lemmatizer = WordNetLemmatizer()
        stopwords_set = set(stopwords.words('english'))
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return lemmatizer, stopwords_set, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def expand_with_synonyms(text, synonym_dict):
    """Expand text with synonyms for better matching"""
    tokens = text.lower().split()
    expanded = tokens[:]
    for token in tokens:
        if token in synonym_dict:
            expanded.extend(synonym_dict[token])
    return ' '.join(set(expanded))

def preprocess(text, lemmatizer, stopwords_set):
    """Preprocess text: tokenize, remove stopwords, lemmatize"""
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords_set]
    return ' '.join(tokens)

def normalize_scores(scores):
    """Normalize scores to 0-1 range"""
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score < 1e-7:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)

@st.cache_data
def process_faq_data(df, synonym_dict, _lemmatizer, _stopwords_set):
    """Process FAQ data with caching - Fixed with underscore prefixes"""
    # Create full text with synonyms
    df["fulltext"] = df.apply(
        lambda row: expand_with_synonyms(row["question"], synonym_dict) + " " +
                    expand_with_synonyms(row["answer"], synonym_dict),
        axis=1
    )
    
    # Preprocess texts
    faq_texts = df["fulltext"].apply(lambda x: preprocess(x, _lemmatizer, _stopwords_set)).tolist()
    return faq_texts

@st.cache_resource
def initialize_search_engines(df, faq_texts, _model):
    """Initialize search engines with caching - Fixed with underscore prefix"""
    # TF-IDF
    vectorizer = TfidfVectorizer()
    faq_vecs = vectorizer.fit_transform(faq_texts)
    
    # Semantic embeddings
    faq_embed = _model.encode(faq_texts, show_progress_bar=False)
    
    # BM25
    bm = None
    if bm25_available:
        try:
            tokenized_corpus = []
            for text in faq_texts:
                tokens = bm25s.tokenize(text)
                if isinstance(tokens, str):
                    tokens = tokens.split()
                elif not isinstance(tokens, list):
                    tokens = [str(tokens)]
                tokenized_corpus.append(tokens)
            
            bm = bm25s.BM25()
            bm.index(tokenized_corpus)
        except Exception as e:
            st.error(f"BM25 initialization failed: {e}")
    
    return vectorizer, faq_vecs, faq_embed, bm

def search_faqs(query, df, faq_texts, vectorizer, faq_vecs, faq_embed, bm, model, 
                lemmatizer, stopwords_set, synonym_dict, top_k=5, threshold=0.1):
    """Perform multi-algorithm FAQ search"""
    
    processed_query = preprocess(expand_with_synonyms(query, synonym_dict), lemmatizer, stopwords_set)
    if not processed_query:
        return []
    
    all_results = []
    
    # BM25 Search
    if bm is not None:
        try:
            query_tokens = bm25s.tokenize(processed_query)
            if isinstance(query_tokens, str):
                query_tokens = query_tokens.split()
            elif not isinstance(query_tokens, list):
                query_tokens = [str(query_tokens)]
            
            if query_tokens:
                res, scores = bm.retrieve([query_tokens], k=len(df))
                indices = res[0]
                scores_for_query = scores[0]
                
                filtered = [(df.iloc[indices[i]], scores_for_query[i], indices[i]) 
                           for i in range(len(indices)) if scores_for_query[i] > threshold]
                bm25_results = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]
                
                for entry, score, idx in bm25_results:
                    all_results.append(('BM25', entry, score, idx))
        except Exception as e:
            st.error(f"BM25 search error: {e}")
    
    # TF-IDF Search
    try:
        q_vec = vectorizer.transform([processed_query])
        sims = cosine_similarity(q_vec, faq_vecs).flatten()
        all_indices = np.where(sims > threshold)[0]
        sort_idx = all_indices[np.argsort(sims[all_indices])[::-1]]
        tfidf_filtered = [(df.iloc[i], sims[i], i) for i in sort_idx][:top_k]
        
        for entry, score, idx in tfidf_filtered:
            all_results.append(('TF-IDF', entry, score, idx))
    except Exception as e:
        st.error(f"TF-IDF search error: {e}")
    
    # Semantic Search
    try:
        q_emb = model.encode([processed_query])
        norm_faq_embed = faq_embed / np.linalg.norm(faq_embed, axis=1, keepdims=True)
        norm_q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        sims = np.dot(norm_faq_embed, norm_q_emb.T).squeeze()
        all_indices = np.where(sims > threshold)[0]
        sort_idx = all_indices[np.argsort(sims[all_indices])[::-1]]
        semantic_filtered = [(df.iloc[i], sims[i], i) for i in sort_idx][:top_k]
        
        for entry, score, idx in semantic_filtered:
            all_results.append(('Semantic', entry, score, idx))
    except Exception as e:
        st.error(f"Semantic search error: {e}")
    
    # Hybrid Search
    if bm is not None:
        try:
            q_emb = model.encode([processed_query])
            norm_faq_embed = faq_embed / np.linalg.norm(faq_embed, axis=1, keepdims=True)
            norm_q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
            emb_sims = np.dot(norm_faq_embed, norm_q_emb.T).squeeze()
            
            query_tokens = bm25s.tokenize(processed_query)
            if isinstance(query_tokens, str):
                query_tokens = query_tokens.split()
            elif not isinstance(query_tokens, list):
                query_tokens = [str(query_tokens)]
            
            if query_tokens:
                res, bm_scores = bm.retrieve([query_tokens], k=len(df))
                bm_arr = np.zeros(len(df))
                indices = res[0]
                scores_for_query = bm_scores[0]
                for i, idx in enumerate(indices):
                    bm_arr[idx] = scores_for_query[i]
                
                bm_arr = normalize_scores(bm_arr)
                emb_sims = normalize_scores(emb_sims)
                hybrid_scores = 0.3 * bm_arr + 0.7 * emb_sims
                
                all_indices = np.where(hybrid_scores > threshold)[0]
                sort_idx = all_indices[np.argsort(hybrid_scores[all_indices])[::-1]]
                hybrid_filtered = [(df.iloc[i], hybrid_scores[i], i) for i in sort_idx][:top_k]
                
                for entry, score, idx in hybrid_filtered:
                    all_results.append(('Hybrid', entry, score, idx))
        except Exception as e:
            st.error(f"Hybrid search error: {e}")
    
    # Sort all results by score
    all_results.sort(key=lambda x: x[2], reverse=True)
    return all_results

def main():
    st.title("🔍 FAQ Semantic Search System")
    st.markdown("**Upload your FAQ CSV and search using multiple algorithms: BM25, TF-IDF, Semantic, and Hybrid**")
    
    # Download NLTK data
    if not download_nltk_data():
        st.stop()
    
    # Load models
    lemmatizer, stopwords_set, model = load_models()
    if None in [lemmatizer, stopwords_set, model]:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    top_k = st.sidebar.slider("Results per algorithm", 1, 10, 5)
    threshold = st.sidebar.slider("Minimum score threshold", 0.0, 1.0, 0.1, 0.05)
    
    # Synonym dictionary
    synonym_dict = {
        'exempted': ['exemption', 'excluded', 'exclusions'],
        'mutual': ['funds', 'mf'],
        'prohibited': ['restrict', 'violate', 'violation', 'not permitted', 'breaches', 'banned'],
        'policy': ['policies', 'rule', 'rules', 'regulation', 'regulations'],
        'investment': ['invest', 'investing', 'investments'],
    }
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload FAQ CSV file",
        type=['csv'],
        help="CSV should have 'question' and 'answer' columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            if 'question' not in df.columns or 'answer' not in df.columns:
                st.error("CSV must contain 'question' and 'answer' columns")
                st.stop()
            
            st.success(f"✅ Loaded {len(df)} FAQ entries")
            
            # Show data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            
            # Process data - Fixed function call with underscore args
            with st.spinner("Processing FAQ data..."):
                faq_texts = process_faq_data(df, synonym_dict, lemmatizer, stopwords_set)
            
            # Initialize search engines - Fixed function call with underscore arg
            with st.spinner("Initializing search engines..."):
                vectorizer, faq_vecs, faq_embed, bm = initialize_search_engines(df, faq_texts, model)
            
            # Search interface
            st.markdown("## 🔍 Search FAQ")
            
            # Search methods info
            methods = ["TF-IDF", "Semantic"]
            if bm is not None:
                methods.extend(["BM25", "Hybrid"])
            st.info(f"**Available search methods:** {', '.join(methods)}")
            
            # Search input
            query = st.text_input("Enter your question:", placeholder="e.g., What are the investment policies?")
            
            if st.button("🔍 Search", type="primary") and query:
                with st.spinner("Searching..."):
                    results = search_faqs(
                        query, df, faq_texts, vectorizer, faq_vecs, faq_embed, bm, 
                        model, lemmatizer, stopwords_set, synonym_dict, top_k, threshold
                    )
                
                if results:
                    st.markdown("## 📋 Search Results")
                    
                    # Group results by algorithm
                    algorithms = {}
                    for alg, entry, score, idx in results:
                        if alg not in algorithms:
                            algorithms[alg] = []
                        algorithms[alg].append((entry, score, idx))
                    
                    # Display results in tabs
                    tabs = st.tabs(list(algorithms.keys()) + ["All Results"])
                    
                    # Individual algorithm tabs
                    for i, (alg, alg_results) in enumerate(algorithms.items()):
                        with tabs[i]:
                            for entry, score, idx in alg_results:
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"**Q:** {entry['question']}")
                                        st.markdown(f"**A:** {entry['answer']}")
                                    with col2:
                                        st.metric("Score", f"{score:.4f}")
                                    st.divider()
                    
                    # All results tab
                    with tabs[-1]:
                        for alg, entry, score, idx in results:
                            with st.container():
                                col1, col2, col3 = st.columns([2.5, 0.5, 1])
                                with col1:
                                    st.markdown(f"**Q:** {entry['question']}")
                                    st.markdown(f"**A:** {entry['answer']}")
                                with col2:
                                    st.badge(alg)
                                with col3:
                                    st.metric("Score", f"{score:.4f}")
                                st.divider()
                else:
                    st.warning("No results found. Try adjusting your query or lowering the threshold.")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
