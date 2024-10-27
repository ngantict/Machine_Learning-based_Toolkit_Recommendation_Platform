import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load the data
df_processed = pd.read_csv(r'D:\Study\B3\Internship Form 2024\Report\Processed data\toolkitproject_cleaned.csv')

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Toolkit prediction function
def predict_toolkits(new_project_text):
    cleaned_text = preprocess_text(new_project_text)
    
    # Ensure there are no NaN values in the column
    df_processed['Project Introduction'].fillna('', inplace=True)
    
    # Debug print to verify processed data
    print("Processed DataFrame:", df_processed[['Project Introduction']].head())

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_processed['Project Introduction'])
    new_project_vector = vectorizer.transform([cleaned_text])

    knn_tfidf = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn_tfidf.fit(tfidf_matrix)

    distances, indices = knn_tfidf.kneighbors(new_project_vector)

    # Get the recommended toolkits along with their details
    recommended_toolkits = df_processed.iloc[indices[0]]
    
    # Debug print to check recommended toolkits
    print("Recommended Toolkits:", recommended_toolkits)

    results = []
    for _, toolkit in recommended_toolkits.iterrows():
        results.append({
            'name': toolkit.get('Tool Name', 'N/A'),  # Use .get() to avoid KeyError
            'domain': toolkit.get('Domain', 'N/A'),
            'framework': toolkit.get('Framework', 'N/A'),
            'programming_language': toolkit.get('Programming Language', 'N/A'),
            'role': toolkit.get('Role', 'N/A'),
            'author': toolkit.get('Author', 'N/A'),
            'description': toolkit.get('Tool Introduction', 'N/A'),
        })
    
    return results


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_search', methods=['POST'])
def submit_search():
    new_project_text = request.form.get('searchInput')
    recommendations = predict_toolkits(new_project_text)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
