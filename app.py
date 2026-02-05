from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import requests
import time

app = Flask(__name__)

# Your TMDB API key
API_KEY = "956df2a0d0459d950f60f0227537c4a6"

# Load CSV files
movies = pd.read_csv("movies.csv")
similarity = pd.read_csv("similarity.csv")

# Convert similarity to numeric matrix
similarity = similarity.apply(pd.to_numeric, errors='coerce').fillna(0)
similarity = similarity.values

# List of movie titles
movies_list = movies['title'].tolist()


# --------------------------
# Fetch movie poster function
# --------------------------
def fetch_poster(movie_id):
    """Fetch poster from TMDB safely."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        response = requests.get(url, timeout=8)
        data = response.json()

        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path

        return "https://via.placeholder.com/500x750?text=No+Image"

    except requests.exceptions.RequestException:
        time.sleep(1)
        return "https://via.placeholder.com/500x750?text=Image+Unavailable"


# --------------------------
# Recommendation Function
# --------------------------
def recommend(movie_name):
    movie_name = movie_name.strip()

    if movie_name not in movies_list:
        return [("Movie Not Found", "https://via.placeholder.com/500x750")]

    index = movies_list.index(movie_name)
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        movie_id = int(movies.iloc[i[0]].id)
        title = movies.iloc[i[0]].title
        poster = fetch_poster(movie_id)

        recommended_movies.append((title, poster))

    return recommended_movies


# --------------------------
# ROUTES
# --------------------------

@app.route("/")
def index():
    return render_template("index.html", movie_list=movies_list)


@app.route("/recommend", methods=["POST"])
def recommend_movies():
    movie_name = request.form["movie"]
    recommendations = recommend(movie_name)

    return render_template(
        "recommend.html",
        movie=movie_name,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)
