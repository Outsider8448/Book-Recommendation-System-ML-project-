import streamlit as st
import pickle
import numpy as np

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Smart Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# =========================
# Load Model Files
# =========================
pt = pickle.load(open('pt.pkl', 'rb'))
 
import pickle
import gzip
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

with gzip.open(BASE_DIR / "books.pkl.gz", "rb") as f:
    books = pickle.load(f)

    
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))

# =========================
# Header
# =========================

st.title("📚 Smart Book Recommendation System")

st.markdown("""
Discover books tailored to your reading preferences using
**collaborative filtering** and **machine learning**.
""")

st.info("""
### How it works
• Analyzes reader rating patterns

• Finds books liked by similar readers

• Recommends books with related preferences
""")

# =========================
# Book Selection
# =========================

selected_book = st.selectbox(
    "Select a book you enjoyed",
    pt.index.values
)

# =========================
# Recommendation Button
# =========================

if st.button("🔍 Find Similar Books"):

    st.success(
        "✨ Great choice! Here are some books readers with similar tastes also enjoyed."
    )

    index = np.where(pt.index == selected_book)[0][0]

    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    st.subheader("📖 Recommended For You")

    col1, col2, col3, col4, col5 = st.columns(5)

    columns = [col1, col2, col3, col4, col5]

    for idx, item in enumerate(similar_items):

        temp_df = books[
            books['Book-Title'] == pt.index[item[0]]
        ].drop_duplicates('Book-Title')

        with columns[idx]:

            st.image(
                temp_df['Image-URL-M'].values[0],
                use_container_width=True
            )

            st.markdown(
                f"**{temp_df['Book-Title'].values[0]}**"
            )

            st.caption(
                f"✍️ {temp_df['Book-Author'].values[0]}"
            )

# =========================
# Footer
# =========================

st.markdown("---")

st.markdown(
    """
    **Built by Karan Yadav**  
    Machine Learning Project | Collaborative Filtering Recommendation Engine
    """
)