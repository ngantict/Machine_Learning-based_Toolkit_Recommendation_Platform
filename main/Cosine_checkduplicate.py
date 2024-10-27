# Combine all text columns into a single text column for TF-IDF
df_combined_text = df.astype(str).agg(' '.join, axis=1)

# Use TF-IDF to vectorize the combined text data
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df_combined_text)

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)
np.fill_diagonal(cosine_sim, 0)  # Set self-similarity to 0 to avoid self-duplication

# Find duplicates based on a threshold
threshold = 0.9
duplicates = np.argwhere(cosine_sim > threshold)

# Create a list of indices to drop
duplicates_set = set(duplicates[:, 1])
duplicates_set = list(duplicates_set)

# Drop duplicate rows from the original DataFrame
df_processed = df.drop(df.index[duplicates_set])

# Save the cleaned DataFrame to a new CSV file
df_processed.to_csv('/content/toolkitproject_cleaned.csv', index=False)

# Print the number of duplicates found and the resulting DataFrame shape
print(f"Number of duplicates found and removed: {len(duplicates_set)}")
print(f"Shape of the cleaned DataFrame: {df_processed.shape}")
print(f"Head of the cleaned DataFrame: {df_processed.head()}")