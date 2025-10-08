import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ------------------ Step 1: Load Data ------------------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
data = pd.merge(ratings, movies, on="movieId")

# ------------------ Step 2: Create User-Movie Matrix ------------------
user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating").fillna(0)

# ------------------ Step 3: Compute Similarity ------------------
similarity = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# ------------------ Step 4: Recommendation Function ------------------
def recommend_movies(user_id, num_recommendations=5):
    if user_id not in similarity_df.index:
        return None
    
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6].index
    similar_users_ratings = data[data['userId'].isin(similar_users)]
    seen_movies = data[data['userId'] == user_id]['title'].unique()
    
    recommendations = (
        similar_users_ratings[~similar_users_ratings['title'].isin(seen_movies)]
        .groupby('title')['rating']
        .mean()
        .sort_values(ascending=False)
        .head(num_recommendations)
    )
    return recommendations

# ------------------ Step 5: Streamlit GUI ------------------
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="centered")

st.title("🎥 Personalized Movie Recommendation System")
st.markdown("This app recommends movies based on what similar users liked!")

# Input user ID
user_id = st.number_input("Enter a User ID (1 - 600):", min_value=1, max_value=600, step=1)

# Number of recommendations
num_recs = st.slider("Number of recommendations:", 3, 15, 5)

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_movies(user_id, num_recs)
    if recommendations is None or recommendations.empty:
        st.error("❌ User not found or no recommendations available.")
    else:
        st.subheader(f"🎯 Top {num_recs} recommendations for User {user_id}:")
        for movie, rating in recommendations.items():
            st.write(f"⭐ **{movie}** — Avg Rating: {rating:.2f}")
