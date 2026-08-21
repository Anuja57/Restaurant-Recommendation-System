# 🍽️ Foodiefy: Restaurant Recommender System

This project is a content-based restaurant recommendation system built with **Streamlit** and **scikit-learn**. It suggests similar restaurants based on TF-IDF vector similarity and KNN search.

---

## 📁 Project Structure
```
streamlit_restaurant_app/
├── data/
│   └── zomato_pune_V002.csv              # Raw restaurant data
│
├── models/
│   ├── tfidf_vectorizer.pkl              # Saved TF-IDF vectorizer
│   └── knn_recommender_model.pkl         # Trained KNN model for similarity
│
├── app/
│   └── restaurant_recommender.py         # Streamlit app to run
│
├── training/
│   └── train_recommender_model.py        # Model training script
│
├── requirements.txt                      # Dependencies list
└── README.md                             # You're here!
```

---

## 🛠️ Installation

1. Clone the repo or download the files:
```bash
git clone <your-repo-url>
cd streamlit_restaurant_app
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training (One-Time Setup)
Before running the app, train the TF-IDF and KNN models:
```bash
cd training
python train_recommender_model.py
```
This creates:
- `models/tfidf_vectorizer.pkl`
- `models/knn_recommender_model.pkl`

---

## 🚀 Run the App
```bash
cd app
streamlit run restaurant_recommender.py
```

Then open the link in your browser (usually http://localhost:8501).

---

## ✅ Requirements
See `requirements.txt` for full list:
- streamlit
- pandas
- scikit-learn
- joblib

---

## 📬 Contact
Feel free to reach out if you have questions, ideas, or want to contribute!

