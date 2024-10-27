# Import data
df = pd.read_csv('/your/path/toolkitproject.csv')

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from lightgbm import LGBMRegressor

# Replace missing values
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

# Initialize lists to store validation RMSE for each row processed
val_rmse_list = []
train_rmse_list = []  # List to store training RMSE

# Iterate over each row to handle missing values
for i, row in df.iterrows():
    if row.isnull().any():
        print(f"Processing row: {i}")

        # Separate complete and incomplete columns in the row
        X_complete = row.dropna().values.reshape(1, -1)
        X_missing_indices = row.index[row.isnull()]

        # If no complete columns are available, skip this row
        if X_complete.shape[1] == 0:
            print(f"Skipping row {i} due to insufficient complete data.")
            continue

        # Create a DataFrame of other rows as training data
        train_df = df.drop(index=i).dropna()
        X_train = train_df.drop(columns=X_missing_indices).values
        y_train = train_df[X_missing_indices].values.ravel()

        # Skip if no training data available
        if X_train.shape[0] == 0:
            print(f"Skipping row {i} due to no available training data.")
            continue

        # Split the training data into training and validation sets
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'num_leaves': [31, 38, 50],
            'learning_rate': [0.1, 0.05, 0.01],
            'n_estimators': [100, 200, 500],
            'min_data_in_leaf': [20, 50, 100]
        }

        # Train the LightGBM model using GridSearchCV
        model = LGBMRegressor(boosting_type='gbdt', objective='regression', metric='rmse', verbose=-1)

        # Setup GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5, verbose=1)

        # Fit the model
        grid_search.fit(X_train_split, y_train_split)

        # Get the best model after GridSearchCV
        best_model = grid_search.best_estimator_

        # Predict using the best model
        train_rmse = mean_squared_error(y_train_split, best_model.predict(X_train_split), squared=False)
        val_rmse = mean_squared_error(y_val_split, best_model.predict(X_val_split), squared=False)

        # Append the RMSE values to the respective lists
        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)

        # Predict missing values
        predicted_values = best_model.predict(X_complete)

        # Fill in the missing values
        df.loc[i, X_missing_indices] = predicted_values

        # Calculate and store performance metrics
        rmse_list.append(train_rmse)
        mae = mean_absolute_error(y_train_split, best_model.predict(X_train_split))
        r2 = r2_score(y_train_split, best_model.predict(X_train_split))

        mae_list.append(mae)
        r2_list.append(r2)

        print(f"Model performance for row {i}: RMSE = {train_rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}, Validation RMSE = {val_rmse:.4f}")

# Decode back to original categories for categorical columns
for col, le in label_encoders.items():
    df[col] = le.inverse_transform(df[col].astype(int))

# Save the imputed DataFrame
df.to_csv('/content/toolkitproject_imputed.csv', index=False)

# Calculate and display overall performance metrics
average_rmse = np.mean(rmse_list)
average_mae = np.mean(mae_list)
average_r2 = np.mean(r2_list)
average_val_rmse = np.mean(val_rmse_list)

# Print the result
print(f"\nOverall LightGBM Row-by-Row model performance across all rows:")
print(f"Average RMSE: {average_rmse:.4f}")
print(f"Average MAE: {average_mae:.4f}")
print(f"Average R²: {average_r2:.4f}")
print(f"Average Validation RMSE: {average_val_rmse:.4f}")

# Print the validation RMSE for each row processed
print("\nValidation RMSE for each row processed:")
for idx, val_rmse in enumerate(val_rmse_list):
    print(f"Row {idx}: Validation RMSE = {val_rmse:.4f}")

# Check the cleaned DataFrame
print("Final cleaned DataFrame head:\n", df.head())

# Plot the training and validation RMSE
plt.figure(figsize=(10, 5))
plt.plot(train_rmse_list, label='Training RMSE', color='blue')
plt.plot(val_rmse_list, label='Validation RMSE', color='orange')
plt.title('Training and Validation RMSE Over Rows Processed')
plt.xlabel('Rows Processed')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()
