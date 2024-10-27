import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# New project description
new_project_text = "I'm looking for a project that involves cloud computing with AWS, possibly in the financial services sector, and it should use a machine learning framework like TensorFlow."

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Preprocess new project text
cleaned_text = preprocess_text(new_project_text)

# TF-IDF for cosine similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_processed['Project Introduction'])
new_project_vector = vectorizer.transform([cleaned_text])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df_processed['Project Introduction'], df_processed['Tool Name'], test_size=0.3, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Function to get recommended tools based on indices
def get_recommendations(indices, df_processed):
    recommendations = []
    for index_list in indices:
        tools = df_processed['Tool Name'].iloc[index_list].tolist()
        recommendations.append(tools)
    return recommendations


# Label Encoding
label_encoder = LabelEncoder()
label_encoder.fit(df_processed['Tool Name'])  # Fit on the entire Tool Name column

y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)