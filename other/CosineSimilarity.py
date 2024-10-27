# Cosine Similarity with TF-IDF
def cosine_similarity_predictions(X_test_tfidf, X_train_tfidf, y_train):
    similarities = cosine_similarity(X_test_tfidf, X_train_tfidf)
    indices = np.argmax(similarities, axis=1)
    return y_train.iloc[indices].values

# Predict and evaluate Cosine Similarity with TF-IDF
predictions_cosine_tfidf = cosine_similarity_predictions(X_test_tfidf, X_train_tfidf, df_processed['Tool Name'])

# Print Cosine Similarity with TF-IDF recommendations
print("\nCosine Similarity with TF-IDF Recommendations:")
for i, rec in enumerate(predictions_cosine_tfidf):
    print(f"Test Sample {i+1}: Recommended Tool: {rec}")

# Evaluate Cosine Similarity with TF-IDF
precision_cosine_tfidf, recall_cosine_tfidf, f1_cosine_tfidf, accuracy_cosine_tfidf = evaluate(predictions_cosine_tfidf, y_test)

print(f"\nCosine Similarity with TF-IDF - Precision: {precision_cosine_tfidf:.4f}, Recall: {recall_cosine_tfidf:.4f}, F1 Score: {f1_cosine_tfidf:.4f}, Accuracy: {accuracy_cosine_tfidf:.4f}")