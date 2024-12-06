from flask import Flask, request, jsonify
from recommendation.recommend_movies import recommend_movies
import numpy as np

app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    """
    API endpoint to recommend movies.
    Expects query parameters: user_id (int), genre (str)
    Example: /recommend?user_id=1&genre=Action
    """
    # Get user_id and genre from the query parameters
    user_id = request.args.get("user_id", type=int)
    genre = request.args.get("genre", type=str)

    # Validate that user_id is provided
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    # Check if genre is provided, if not, set to None (or set to a default value)
    if not genre:
        genre = None

    # Call the recommend_movies function with the user_id and genre filter
    recommendations = recommend_movies(user_id, genre=genre)

    # Convert int64 or other non-serializable types to native Python types (e.g., int)
    recommendations = convert_int64_to_int(recommendations)

    # Structure the response to make it more user-friendly
    response = {
        "status": "success",
        "message": f"Top movie recommendations for user {user_id}",
        "recommendations": recommendations
    }

    return jsonify(response), 200

def convert_int64_to_int(data):
    """
    Recursively converts int64 types to native Python int types in lists or dictionaries.
    """
    if isinstance(data, list):
        return [convert_int64_to_int(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_int64_to_int(value) for key, value in data.items()}
    elif isinstance(data, np.int64):
        return int(data)
    else:
        return data

if __name__ == "__main__":
    app.run(debug=True)




# http://127.0.0.1:5000//recommend?user_id=1&top_n=5
# http://127.0.0.1:5000/recommend?user_id=1&genre=Action