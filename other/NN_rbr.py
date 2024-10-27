# Import data
df = pd.read_csv('/your/path/toolkitproject.csv')

# Replace potential missing value indicators with NaN
df.replace(['', ' ', '?', 'NA', 'None'], np.nan, inplace=True)

# Initialize a LabelEncoder dictionary
label_encoders = {}

# Encode all categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Initialize lists to store performance metrics
rmse_list = []
mae_list = []
r2_list = []

# Process each row with missing values
for i, row in df.iterrows():
    if row.isnull().any():
        print(f"Processing row: {i}")

        # Separate columns with missing values
        missing_cols = row.index[row.isnull()]
        non_missing_cols = row.index[~row.isnull()]

        # Prepare training data
        train_df = df.drop(index=i).dropna(subset=non_missing_cols)
        X_train = train_df[non_missing_cols]
        y_train = train_df[missing_cols]

        # Skip if no training data available
        if X_train.empty or y_train.empty:
            print(f"Skipping row {i} due to insufficient training data.")
            continue

        # Ensure no NaN in training data
        if X_train.isna().any().any() or y_train.isna().any().any():
            print(f"Skipping row {i} due to NaN values in training data.")
            continue

        # Normalize data
        scaler_X = StandardScaler()
        X_train_normalized = scaler_X.fit_transform(X_train)

        # Normalize the features of the current row
        X_complete = row[non_missing_cols].values.reshape(1, -1)
        X_complete_normalized = scaler_X.transform(X_complete)

        # Split the training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_normalized, y_train, test_size=0.2, random_state=42)

        # Define the Neural Network model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(missing_cols))
        ])
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train_split, y_train_split, validation_data=(X_val_split, y_val_split), epochs=50, callbacks=[early_stopping], verbose=1)

        # Predict missing values
        predicted_values = model.predict(X_complete_normalized).flatten()

        # Fill in the missing values
        df.loc[i, missing_cols] = predicted_values

        # Evaluate the model
        y_val_pred = model.predict(X_val_split).flatten()
        rmse = mean_squared_error(y_val_split, y_val_pred, squared=False)
        mae = mean_absolute_error(y_val_split, y_val_pred)
        r2 = r2_score(y_val_split, y_val_pred)

        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

        print(f"Model performance for row {i}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")

        # Plot learning curve for the row
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Neural Network Learning Curve for row {i}')
        plt.legend()
        plt.show()

# Decode back to original categories for categorical columns
for col, le in label_encoders.items():
    df[col] = le.inverse_transform(df[col].astype(int))

# Save the imputed DataFrame
df.to_csv('/content/toolkitproject_imputed.csv', index=False)

# Calculate and display overall performance metrics
average_rmse = np.mean(rmse_list)
average_mae = np.mean(mae_list)
average_r2 = np.mean(r2_list)

print(f"\nOverall Neural Network Row-by-Row model performance across all rows:")
print(f"Average RMSE: {average_rmse:.4f}")
print(f"Average MAE: {average_mae:.4f}")
print(f"Average R²: {average_r2:.4f}")

# Check the cleaned DataFrame
print("Final cleaned DataFrame head:\n", df.head())