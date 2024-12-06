import pandas as pd
import mysql.connector

# Load ratings data
ratings = pd.read_csv('../u.data', sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Connect to MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',  # Replace with your username
    password='1234',  # Replace with your password
    database='movie_recommendation_db'
)
cursor = conn.cursor()

# Insert ratings data into the database
for _, row in ratings.iterrows():
    cursor.execute(
        "INSERT INTO movie_ratings (user_id, movie_id, rating) VALUES (%s, %s, %s)",
        (int(row['user_id']), int(row['movie_id']), float(row['rating']))
    )

conn.commit()
cursor.close()
conn.close()

print("Ratings data loaded successfully!")
