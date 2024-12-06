import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
import mysql.connector


def connect_to_database():
    """Connect to the MySQL database."""
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='1234',
        database='movie_recommendation_db'
    )


def fetch_data():
    """Fetch ratings data from the database."""
    conn = connect_to_database()
    query = "SELECT user_id, movie_id, rating FROM movie_ratings"
    df_ratings = pd.read_sql(query, conn)
    conn.close()
    return df_ratings


def fetch_movie_details():
    """Fetch movie details (name and genre)."""
    conn = connect_to_database()
    query = "SELECT movie_id, movie_name, genre FROM movie_details"
    df_movies = pd.read_sql(query, conn)
    conn.close()
    return df_movies


def recommend_movies(user_id, genre, genre_filter=None, top_n=5):
    """
    Generate recommendations for a user.
    If genre_filter is provided, filter the recommendations by genre.
    """
    ratings = fetch_data()
    movies = fetch_movie_details()

    # Filter movies based on genre
    if genre_filter:
        genre_movies = movies[movies['genre'].str.contains(genre_filter, case=False)]['movie_id']
    else:
        genre_movies = movies['movie_id']

    # Create user-item matrix (matrix of users and movies they've rated)
    user_movie_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating')

    # Fill missing values in the user-movie matrix with mean rating
    imputer = SimpleImputer(strategy='mean')
    filled_matrix = imputer.fit_transform(user_movie_matrix)

    # Compute cosine similarity between users
    similarity = cosine_similarity(filled_matrix)

    # Get the index of the user in the matrix
    user_index = user_id - 1
    similar_users = similarity[user_index].argsort()[-8:][::-1]  # Get 7 most similar users
    recommendations = []

    for similar_user in similar_users:
        user_ratings = user_movie_matrix.iloc[similar_user]
        top_movies = user_ratings[user_ratings.index.isin(genre_movies)].sort_values(ascending=False)[:top_n]
        recommended_movie_details = movies[movies['movie_id'].isin(top_movies.index)].to_dict(orient='records')
        recommendations.append({"user_id": similar_user + 1, "movies": recommended_movie_details})

    return recommendations
