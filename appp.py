import pandas as pd
import numpy as np
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# -------------------- Load and preprocess data --------------------

@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    def convert(obj):
        return [i["name"] for i in ast.literal_eval(obj)][:5]

    def get_director(obj):
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                return i["name"]
        return ""

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert)
    movies["crew"] = movies["crew"].apply(get_director)

    movies["overview"] = movies["overview"].apply(lambda x: x.split())
    movies["crew"] = movies["crew"].apply(lambda x: [x])
    movies["tags"] = (
        movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    )

    new = movies[["movie_id", "title", "tags"]].copy()
    new["tags"] = new["tags"].apply(lambda x: " ".join(x).lower())
    return new


movies = load_data()

# -------------------- Vectorization --------------------
#import from scikit learn ML library
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

# -------------------- Helper Functions --------------------

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=YOUR_API_KEY&language=en-US"
        response = requests.get(url)
        data = response.json()
        poster_path = data.get("poster_path", "")
        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""
    except:
        return ""


def recommend(movie, n):
    try:
        idx = movies[movies["title"] == movie].index[0]
        distances = similarity[idx]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : n + 1]
        return [(movies.iloc[i[0]].title, movies.iloc[i[0]].movie_id) for i in movie_list]
    except:
        return []

# -------------------- Streamlit UI --------------------

st.title("🎬 Movie Recommendation System")
selected_movie = st.selectbox("Choose a movie", movies["title"].values)
num_recommendations = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5)

if st.button("Recommend"):
    results = recommend(selected_movie, num_recommendations)
    if results:
        st.subheader(f"Top {num_recommendations} Recommendations for **{selected_movie}**:")
        for i, (title, movie_id) in enumerate(results, 1):
            col1, col2 = st.columns([1, 4])
            with col1:
                poster_url = fetch_poster(movie_id)
                if poster_url:
                    st.image(poster_url, use_column_width=True)
            with col2:
                st.markdown(f"**{i}. {title}**")
    else:
        st.error("Movie not found. Please try a different title.")
