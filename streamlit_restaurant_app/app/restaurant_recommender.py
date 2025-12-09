import streamlit as st
import pandas as pd
import time
import urllib.parse
import joblib
import os


@st.cache_data
def load_data():
    # Get folder where this script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the CSV file (adjust ".." depending on folder structure)
    data_path = os.path.join(BASE_DIR, "..", "data", "zomato_pune_V002.csv")
    
    # Check if file exists
    if not os.path.exists(data_path):
        st.error(f"Data file not found at:\n{data_path}\nPlease check the file location.")
        return pd.DataFrame()  # return empty DataFrame to avoid crashes
    
    # Load CSV
    df = pd.read_csv(data_path)
    return df


def load_models():
    # Get the absolute path of the current script
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths to your model files
    vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
    knn_model_path = os.path.join(BASE_DIR, "models", "knn_recommender_model.pkl")
    
    # Check if files exist
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}")
    if not os.path.exists(knn_model_path):
        raise FileNotFoundError(f"KNN model not found at {knn_model_path}")
    
    # Load models
    vectorizer = joblib.load(vectorizer_path)
    knn_model = joblib.load(knn_model_path)
    
    return vectorizer, knn_model


# Load everything
df = load_data()
vectorizer, knn_model = load_models()

df["combined_text"] = df["Restaurant_Name"] + " " + df["Cuisines"] + " " + df["Locality"]
tfidf_matrix = vectorizer.transform(df["combined_text"])

# Custom typewriter effect
def typewriter_effect(text, speed=0.05):
    placeholder = st.empty()
    full_text = ""
    for char in text:
        full_text += char
        placeholder.markdown(f"## {full_text}")
        time.sleep(speed)

# App styling
st.markdown("<style>body { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center; color:#FF4B4B;'>🍽 Foodiefy: Restaurant Recommender</h1>", unsafe_allow_html=True)
typewriter_effect("Welcome to Foodiefy — your smart restaurant finder powered by machine learning.", speed=0.04)
st.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 Filter Restaurants")
localities = sorted(df["Locality"].dropna().unique())
cuisines = sorted(set(", ".join(df["Cuisines"].dropna()).split(", ")))

selected_locality = st.sidebar.selectbox("Select Locality", ["Any"] + localities)
selected_cuisine = st.sidebar.selectbox("Select Cuisine", ["Any"] + cuisines)

# Button logic
if st.sidebar.button("Find Restaurants"):
    with st.spinner("Finding the best spots for you..."):
        time.sleep(1.2)
        filtered_df = df.copy()

        if selected_locality != "Any":
            filtered_df = filtered_df[filtered_df["Locality"] == selected_locality]

        if selected_cuisine != "Any":
            filtered_df = filtered_df[filtered_df["Cuisines"].str.contains(selected_cuisine, case=False, na=False)]

        if filtered_df.empty:
            st.warning("😕 No restaurants found matching your criteria. Please try something else! ")
        else:
            st.success(f"✅ Found {len(filtered_df)} matching restaurants.")

            # Recommend similar restaurants to the first one
            query = filtered_df.iloc[0]["combined_text"]
            query_vec = vectorizer.transform([query])
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=6)

            st.markdown("### 🔍 Similar Recommendations:")
            for i in indices[0][1:]:  # Skip the first (itself)
                r = df.iloc[i]
                maps_url = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(r['Detail_address'])}"

                st.markdown(f"""
                <div style='background-color:#1f1f1f; padding:15px; margin-bottom:15px; border-radius:12px; box-shadow: 0 0 10px #333'>
                    <h3 style='color:#FF4B4B;'>{r['Restaurant_Name']}</h3>
                    <p><b>📍 Address:</b> {r['Detail_address']}</p>
                    <p><b>🍽 Cuisine:</b> {r['Cuisines']}</p>
                    <p><b>⭐ Rating:</b> {r['Ratings_out_of_5']} &nbsp;&nbsp; <b>🗳 Votes:</b> {r['Number of votes']}</p>
                    <a href='{maps_url}' target='_blank' style='color:#1E90FF; text-decoration: none;'>🔗 See on Google Maps</a>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("👈 Select options from the sidebar to begin your foodie journey!")

st.markdown("---")
st.markdown("<center>Made with ❤ using Streamlit | Foodiefy 🍴</center>", unsafe_allow_html=True)
