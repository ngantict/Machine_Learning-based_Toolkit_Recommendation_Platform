# KNN with TF-IDF
knn_tfidf = NearestNeighbors(n_neighbors=7, metric='cosine')
knn_tfidf.fit(X_train_tfidf)

def knn_predict(model, X_test):
    distances, indices = model.kneighbors(X_test)
    return indices

# Predict and evaluate KNN with TF-IDF
indices_knn_tfidf = knn_predict(knn_tfidf, X_test_tfidf)
recommendations_knn_tfidf = get_recommendations(indices_knn_tfidf, df_processed)

## Print KNN with TF-IDF recommendations
#print("KNN with TF-IDF Recommendations:")
#for i, recs in enumerate(recommendations_knn_tfidf):
#    print(f"Test Sample {i+1}: Recommended Tools: {', '.join(recs)}")

# Evaluate KNN with TF-IDF
def evaluate(predictions, y_test):
    y_test_encoded = LabelEncoder().fit(df_processed['Tool Name']).transform(y_test)
    y_pred_encoded = LabelEncoder().fit(df_processed['Tool Name']).transform(predictions)
    precision = precision_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0)
    recall = recall_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0)
    f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    return precision, recall, f1, accuracy

# Generate predictions
predictions_knn_tfidf = [item[0] for item in recommendations_knn_tfidf]
precision_knn_tfidf, recall_knn_tfidf, f1_knn_tfidf, accuracy_knn_tfidf = evaluate(predictions_knn_tfidf, y_test)

print(f"\nKNN with TF-IDF - Precision: {precision_knn_tfidf:.4f}, Recall: {recall_knn_tfidf:.4f}, F1 Score: {f1_knn_tfidf:.4f}, Accuracy: {accuracy_knn_tfidf:.4f}")