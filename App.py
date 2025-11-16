import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 🧩 Step 1: Load and prepare dataset
news_df = pd.read_csv(r"C:\Users\khiza\Downloads\archive (8)\WELFake_Dataset.csv")
news_df = news_df.fillna(' ')

# 🧠 Download NLTK data (only first time)
nltk.download('stopwords')

# 🪶 Step 2: Text preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(title):
    stemmed_title = re.sub('[^a-zA-Z]', " ", title)
    stemmed_title = stemmed_title.lower()
    stemmed_title = stemmed_title.split()
    stemmed_title = [ps.stem(word) for word in stemmed_title if word not in stop_words]
    return " ".join(stemmed_title)

news_df['title'] = news_df['title'].apply(stemming)

# 🧰 Step 3: Feature extraction
x = news_df['title'].values
y = news_df['label'].values

vector = TfidfVectorizer()
vector.fit(x)
x = vector.transform(x)

# 🧩 Step 4: Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=1)

# 🧠 Step 5: Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# 🩵 Step 6: Streamlit UI
st.title('📰 Fake News Detector')
input_text = st.text_input('Enter a news article:')

if st.button('Check'):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Preprocess user input
        processed_text = stemming(input_text)
        vectorized_input = vector.transform([processed_text])
        prediction = model.predict(vectorized_input)

        if prediction[0] == 1:
            st.error("🚨 Fake News Detected!")
        else:
            st.success("✅ Real News Article!")



