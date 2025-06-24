import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ----------------------------
# Step 1: Load and preprocess data
# ----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")

    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

    # Parse stringified lists into Python lists
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i["name"])
        return L[:5]

    def get_director(obj):
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                return i["name"]
        return ""

    movies.dropna(inplace=True)
    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert)
    movies["crew"] = movies["crew"].apply(get_director)

    movies["overview"] = movies["overview"].apply(lambda x: x.split())
    movies["crew"] = movies["crew"].apply(lambda x: [x])
    movies["tags"] = (
        movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    )
    new = movies[["movie_id", "title", "tags"]]
    new["tags"] = new["tags"].apply(lambda x: " ".join(x).lower())
    return new

movies = load_data()

# ----------------------------
# Step 2: Create vectorizer and similarity matrix
# ----------------------------
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()

similarity = cosine_similarity(vectors)

# ----------------------------
# Step 3: Recommend function
# ----------------------------
def recommend(movie):
    try:
        idx = movies[movies["title"] == movie].index[0]
    except:
        return []
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# ----------------------------
# Step 4: Streamlit UI
# ----------------------------
st.title("🎬 Movie Recommendation System")
selected_movie = st.selectbox("Choose a movie", movies["title"].values)

if st.button("Recommend"):
    results = recommend(selected_movie)
    if results:
        st.subheader("Top 5 Recommendations:")
        for i, title in enumerate(results, 1):
            st.write(f"{i}. {title}")
    else:
        st.write("Sorry, no recommendations found.")

