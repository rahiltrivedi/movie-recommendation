import pandas as pd
import mysql.connector
import os

# Database connection details
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",  # Replace with your MySQL password
    "database": "movie_recommendation_db",
}

# File paths for the datasets
MOVIE_GENRE_FILE = "C:/Users/Asus/PycharmProjects/IndustryProject/u.genre"  # Path to u.genre.txt
MOVIE_ITEMS_FILE = "C:/Users/Asus/PycharmProjects/IndustryProject/u.item"  # Path to u.item


def connect_to_database():
    """Connect to the MySQL database."""
    return mysql.connector.connect(**DB_CONFIG)


def create_movie_details_table():
    """Create the `movie_details` table if it doesn't exist."""
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS movie_details (
            movie_id INT PRIMARY KEY,
            movie_name VARCHAR(255),
            genre VARCHAR(255)
        )
    """)
    conn.commit()
    conn.close()
    print("Table `movie_details` is ready.")


def load_movie_details():
    """Load movie details from the `u.item` and `u.genre` files into the database."""

    # Step 1: Check if the genre file exists
    if not os.path.exists(MOVIE_GENRE_FILE):
        print(f"Error: The file {MOVIE_GENRE_FILE} was not found.")
        return

    # Step 2: Load movie genre data from u.genre (if file exists)
    print(f"Loading {MOVIE_GENRE_FILE}...")
    genre_data = pd.read_csv(MOVIE_GENRE_FILE, sep="|", header=None, names=["genre", "movie_id"])

    # Step 3: Load movie items (movie names) data from u.item
    if not os.path.exists(MOVIE_ITEMS_FILE):
        print(f"Error: The file {MOVIE_ITEMS_FILE} was not found.")
        return
    print(f"Loading {MOVIE_ITEMS_FILE}...")
    movie_data = pd.read_csv(MOVIE_ITEMS_FILE, sep="|", header=None, encoding="latin-1",
                             names=["movie_id", "movie_name", *["col" + str(i) for i in range(22)]])

    # Select only the relevant columns from u.item (movie_id and movie_name)
    movie_data = movie_data[["movie_id", "movie_name"]]

    # Step 4: Merge both dataframes (movie names + genres)
    movie_details = pd.merge(movie_data, genre_data, on="movie_id", how="inner")

    # Step 5: Insert data into the database
    conn = connect_to_database()
    cursor = conn.cursor()
    for _, row in movie_details.iterrows():
        try:
            cursor.execute("""
                INSERT INTO movie_details (movie_id, movie_name, genre)
                VALUES (%s, %s, %s)
            """, (row["movie_id"], row["movie_name"], row["genre"]))
        except mysql.connector.Error as e:
            print(f"Error inserting row {row['movie_id']}: {e}")

    conn.commit()
    conn.close()
    print("Movie details loaded successfully.")


if __name__ == "__main__":
    create_movie_details_table()
    load_movie_details()