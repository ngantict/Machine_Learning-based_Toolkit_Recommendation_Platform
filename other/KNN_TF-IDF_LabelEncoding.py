# KNN with TF-IDF and Label Encoding
knn_tfidf_label_encoded = NearestNeighbors(n_neighbors=3, metric='cosine')
knn_tfidf_label_encoded.fit(X_train_tfidf)

# Predict and evaluate KNN with TF-IDF and Label Encoding
indices_knn_tfidf_label_encoded = knn_predict(knn_tfidf_label_encoded, X_test_tfidf)
recommendations_knn_tfidf_label_encoded = get_recommendations(indices_knn_tfidf_label_encoded, df_processed)

# Print KNN with TF-IDF and Label Encoding recommendations
print("\nKNN with TF-IDF and Label Encoding Recommendations:")
for i, recs in enumerate(recommendations_knn_tfidf_label_encoded):
    print(f"Test Sample {i+1}: Recommended Tools: {', '.join(recs)}")

# Evaluate KNN with TF-IDF and Label Encoding
predictions_knn_tfidf_label_encoded = [item[0] for item in recommendations_knn_tfidf_label_encoded]
precision_knn_tfidf_label_encoded, recall_knn_tfidf_label_encoded, f1_knn_tfidf_label_encoded, accuracy_knn_tfidf_label_encoded = evaluate(predictions_knn_tfidf_label_encoded, y_test)

print(f"\nKNN with TF-IDF and Label Encoding - Precision: {precision_knn_tfidf_label_encoded:.4f}, Recall: {recall_knn_tfidf_label_encoded:.4f}, F1 Score: {f1_knn_tfidf_label_encoded:.4f}, Accuracy: {accuracy_knn_tfidf_label_encoded:.4f}")