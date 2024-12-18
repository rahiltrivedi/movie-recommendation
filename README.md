# Movie Recommendation System

A machine learning-based movie recommendation system using Flask, MySQL, and Pandas. This project provides movie recommendations based on user input, such as genre and user preferences.

## Features
- User-friendly interface via a Flask API
- Fetches movie data and genres from a MySQL database
- Returns top movie recommendations based on a given genre and user input

## Tech Stack
- **Backend**: Flask
- **Database**: MySQL
- **Libraries**: Pandas, NumPy, scikit-learn
- **Web API**: Flask RESTful API
- **Frontend**: Postman (for testing)

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/movie-recommendation-project.git
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the MySQL database**:
    - Ensure you have MySQL installed.
    - Set up your database schema based on the provided SQL script.

4. **Run the Flask app**:
    ```bash
    python app.py
    ```

    This will start the Flask API, which you can interact with using Postman or any other HTTP client.

## Usage
1. **POST Request**:
    - Endpoint: `/recommend`
    - Body: 
        ```json
        {
            "user_id": 1,
            "genre": "Action"
        }
        ```

2. The system will return the top 5 movie recommendations for the given genre.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Kaggle Movie Dataset (u.item, u.genre files)
- Flask Documentation
