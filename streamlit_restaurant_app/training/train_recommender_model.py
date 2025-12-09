import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import os

# Load and clean data
# ------------------------------
# Base paths relative to THIS script
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "zomato_pune_V002.csv")
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
data_path = os.path.join(BASE_DIR, "data", "zomato_pune_V002.csv")
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()
df = df.rename(columns={
    'Restaurant_Name': 'Restaurant Name',
    'Detail_address': 'Address',
    'Ratings_out_of_5': 'Rating',
    'Number of votes': 'Votes',
    'Locality': 'Locality'
})
df = df.dropna(subset=['Restaurant Name', 'Cuisines', 'Locality'])

# Combine features into a single string for each row
df['combined_text'] = df['Restaurant Name'] + " " + df['Cuisines'] + " " + df['Locality']

# Train TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_text'])

# Train KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(tfidf_matrix)

# Save vectorizer and model
# ...existing code...
joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
joblib.dump(knn, os.path.join(models_dir, "knn_recommender_model.pkl"))
# ...existing code...

print("Model and vectorizer saved successfully!")
