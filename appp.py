import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# Loading and preprocessing the data

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

    new = movies.loc[:, ["movie_id", "title", "tags"]].copy()
    new["tags"] = new["tags"].apply(lambda x: " ".join(x).lower())
    return new

movies = load_data()


# Creating vectorizer and similarity matrix

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()

similarity = cosine_similarity(vectors)


# Recommend function

def recommend(movie, n):
    try:
        idx = movies[movies["title"] == movie].index[0]
    except:
        return []
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : n + 1]
    return [movies.iloc[i[0]].title for i in movie_list]


# Streamlit for UI

st.title("🎬 Movie Recommendation System")
selected_movie = st.selectbox("Choose a movie", movies["title"].values)
num_recommendations = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5, step=1)

if st.button("Recommend"):
    results = recommend(selected_movie, num_recommendations)
    if results:
        st.subheader(f"Top {num_recommendations} Recommendations:")
        for i, title in enumerate(results, 1):
            st.write(f"{i}. {title}")
    else:
        st.write("Sorry, no recommendations found.")
