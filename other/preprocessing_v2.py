import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Import data
df_processed = pd.read_csv('/your/path/toolkitproject_cleaned.csv')

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing function for text
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text.lower())  # Remove special characters and lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Preprocess all relevant columns and concatenate their content
def preprocess_and_concatenate(df):
    df['processed_text'] = (
        df['Project Introduction'].fillna('') + ' ' +
        df['Domain'].fillna('') + ' ' +
        df['Role'].fillna('') + ' ' +
        df['Framework'].fillna('') + ' ' +
        df['Programming Language'].fillna('') + ' ' +
        df['Phrase'].fillna('')
    )
    # Apply preprocessing to the concatenated text
    df['processed_text'] = df['processed_text'].apply(preprocess_text)
    return df

# Preprocess your dataset (Assuming 'df_processed' is your dataframe with all necessary columns)
df_processed = preprocess_and_concatenate(df_processed)

# New project description - concatenate relevant columns for the new project
#You could change this to whatever you want
new_project_text = (
    "I'm looking for a project that involves cloud computing with AWS, possibly in the financial services sector, and it should use a machine learning framework like TensorFlow."
)

# Preprocess the new project text (concatenation of all relevant details)
cleaned_text = preprocess_text(new_project_text)

# TF-IDF for cosine similarity - Use processed text from all columns
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_processed['processed_text'])  # Use combined column
new_project_vector = vectorizer.transform([cleaned_text])

# Split the data (Use concatenated text for the input features)
X_train, X_test, y_train, y_test = train_test_split(df_processed['processed_text'], df_processed['Tool Name'], test_size=0.3, random_state=42)

# TF-IDF Vectorization for Train and Test sets
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Function to get recommended tools based on cosine similarity or nearest neighbors
def get_recommendations(indices, df_processed):
    recommendations = []
    for index_list in indices:
        tools = df_processed['Tool Name'].iloc[index_list].tolist()
        recommendations.append(tools)
    return recommendations

# Label Encoding for the target column (Tool Name)
label_encoder = LabelEncoder()
label_encoder.fit(df_processed['Tool Name'])  # Fit on the entire Tool Name column

y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
