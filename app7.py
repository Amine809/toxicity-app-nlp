import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your training data (replace 'train.csv' with your actual filename)
df = pd.read_csv('train.csv')

# Assuming 'toxic' is the toxic comment label in your dataset
Toxic_comment_balanced_1 = df[df['toxic'] == 1].iloc[0:5000, :]
Toxic_comment_balanced_0 = df[df['toxic'] == 0].iloc[0:5000, :]
Toxic_comment_balanced = pd.concat([Toxic_comment_balanced_1, Toxic_comment_balanced_0])

# Train a simple model (Random Forest) for demonstration purposes
X = Toxic_comment_balanced.comment_text
y = Toxic_comment_balanced['toxic']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
X_train_fit = vectorizer.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=50)
model.fit(X_train_fit, y_train)

# Streamlit App
st.title("Toxic Comment Detector")

# Input text box for user comment
user_comment = st.text_area("Enter your comment:")

if st.button("Predict"):
    # Vectorize the user's comment
    user_comment_vect = vectorizer.transform([user_comment])

    # Predict toxicity
    toxicity_probability = model.predict_proba(user_comment_vect)[:, 1]

    # Display result
    st.write(f"Probability of toxicity: {toxicity_probability[0]:.2%}")
