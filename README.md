# 🎬 Personalized Movie Recommendation System

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://personal-movie-reco.streamlit.app)

A **collaborative filtering-based movie recommendation system** that suggests movies based on what similar users enjoyed!

👉 **Live Demo:** https://personal-movie-reco.streamlit.app

---

## 📌 Overview

This system recommends movies by finding users with similar taste and suggesting movies they liked that you haven't seen yet. It uses **cosine similarity** on user-movie rating matrices to find taste patterns.

**How it works:**
1. Enter your User ID (1-600)
2. System finds users with similar movie preferences
3. Recommends highly-rated movies from those similar users
4. Filters out movies you've already watched

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/bivek127/movie-recommendation-system.git
cd movie-recommendation-system

# Install dependencies
pip install pandas numpy streamlit

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📂 Project Structure

```
├── app.py           # Streamlit application
├── movies.csv       # Movie metadata (movieId, title, genres)
├── ratings.csv      # User ratings (userId, movieId, rating, timestamp)
├── links.csv        # IMDB/TMDB links (optional)
└── tags.csv         # User tags (optional)
```

---

## 🧠 How the Algorithm Works

### 1. **User-Movie Matrix**
Creates a matrix where:
- Rows = Users
- Columns = Movies
- Values = Ratings (0 if not rated)

### 2. **Cosine Similarity**
Computes similarity between users using **pure NumPy** (no sklearn):

```python
def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / (norms + 1e-10)
    return X_normalized @ X_normalized.T
```

### 3. **Recommendation**
- Find top 5 most similar users
- Get movies they rated highly
- Filter out movies you've already seen
- Return top N recommendations

---

## 🎯 Features

- **No ML Libraries** - Uses pure NumPy for similarity calculations
- **Collaborative Filtering** - Finds users with similar taste
- **Interactive UI** - Adjust number of recommendations with slider
- **Fast Performance** - Efficient matrix operations
- **Real MovieLens Data** - Based on actual movie ratings

---

## 📊 Dataset

Uses the **MovieLens** dataset containing:
- **Movies:** ~9,700 movies with titles and genres
- **Ratings:** ~100,000 ratings from 600 users
- **Scale:** Ratings from 0.5 to 5.0 stars

---

## 💡 Usage Example

```python
# Get recommendations for User 1
recommendations = recommend_movies(user_id=1, num_recommendations=5)

# Output:
# Flash Gordon (1980) — Avg Rating: 5.00
# Summer of Sam (1999) — Avg Rating: 5.00
# Fisher King, The (1991) — Avg Rating: 5.00
```

---

## 🛠️ Tech Stack

- **Python 3.x** - Core language
- **Streamlit** - Web interface
- **Pandas** - Data manipulation
- **NumPy** - Similarity calculations

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add new recommendation algorithms (SVD, matrix factorization, etc.)
- Improve the UI/UX
- Add movie posters or metadata
- Optimize performance

---

## 📄 License

MIT License - Free to use and modify!

---

## 🙏 Credits

Dataset: [MovieLens](https://grouplens.org/datasets/movielens/)

---

**Made with ❤️ by [bivek127](https://github.com/bivek127)**
