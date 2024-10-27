# Import data
df = pd.read_csv('/your/path/toolkitproject.csv')

# Replace any other potential missing value indicators with NaN
df.replace(['', ' ', '?', 'NA', 'None'], np.nan, inplace=True)

# Initialize a dictionary to store label encoders
label_encoders = {}

# Encode all categorical (object-type) columns
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Initialize variables to store the overall performance
total_rmse, total_mae, total_r2 = [], [], []

# Iterate over each column with missing values (column-by-column approach)
for col in df.columns:
    if df[col].isna().any():
        print(f"Processing column: {col}")

        # Separate the rows with and without missing values in the current column
        missing_rows = df[df[col].isna()]
        non_missing_rows = df[df[col].notna()]

        # Prepare features (X) and target (y) for the current column
        X_train = non_missing_rows.drop(columns=[col])
        y_train = non_missing_rows[col]

        X_missing = missing_rows.drop(columns=[col])

        # Normalize the features
        scaler_X = StandardScaler()
        X_train_normalized = scaler_X.fit_transform(X_train)
        X_missing_normalized = scaler_X.transform(X_missing)

        # Split training data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_normalized, y_train, test_size=0.2, random_state=42)

        # Define the Neural Network model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(X_train_split.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)  # Output layer has 1 unit since we're predicting a single column
        ])
        model.compile(optimizer='adam', loss='mse')

        # Train the model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train_split, y_train_split, validation_data=(X_val, y_val), epochs=50, callbacks=[early_stopping], verbose=1)

        # Predict the missing values for the current column
        y_missing_pred = model.predict(X_missing_normalized).flatten()

        # Fill in the predicted values for the missing rows in the current column
        df.loc[df[col].isna(), col] = y_missing_pred

        # Evaluate the model's performance
        y_val_pred = model.predict(X_val).flatten()
        rmse = mean_squared_error(y_val, y_val_pred, squared=False)
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        print(f"Neural Network Model Performance for column {col}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R-squared: {r2:.4f}")

        # Append performance metrics for overall average calculation
        total_rmse.append(rmse)
        total_mae.append(mae)
        total_r2.append(r2)

        # Plot learning curve for the column
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Neural Network Learning Curve for column {col}')
        plt.legend()
        plt.show()

# Decode the numerical columns back to their original categorical values
for col, le in label_encoders.items():
    if col in df.columns:
        df[col] = le.inverse_transform(df[col].astype(int))

# Save the cleaned DataFrame
df.to_csv('toolkitproject_imputed_column_by_column.csv', index=False)

# Check the cleaned DataFrame
print("Shape of the cleaned DataFrame:", df.shape)
print("Final cleaned DataFrame head:\n", df.head())

# Calculate and print overall model performance across all columns
print("\nOverall Neural Network Column-by-Column model performance across all columns:")
print(f"  Average RMSE: {np.mean(total_rmse):.4f}")
print(f"  Average MAE: {np.mean(total_mae):.4f}")
print(f"  Average R²: {np.mean(total_r2):.4f}")
