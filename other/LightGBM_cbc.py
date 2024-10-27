# Import data
df = pd.read_csv('/your/path/toolkitproject.csv')

import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Convert all missing value placeholders to NaN
df.replace(['', ' ', '?', 'NA', 'None', ',,'], np.nan, inplace=True)

# Initialize a dictionary to store label encoders
label_encoders = {}

# Encode all object-type columns
for col in df.columns:
    if df[col].dtype == 'object':
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    else:
        df[col] = df[col].astype(float)

# Identify columns with missing values
columns_with_missing = df.columns[df.isnull().any()].tolist()
print("Columns with missing values:", columns_with_missing)

# Initialize variables to store the overall performance
total_rmse, total_mae, total_r2 = [], [], []

# Handle missing values column by column
for target_column in columns_with_missing:
    print(f"Processing column: {target_column}")

    # Split data into features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Drop rows where the target is missing for training
    X_train_full = X[~y.isnull()]
    y_train_full = y[~y.isnull()]

    # Separate data where the target is missing (for prediction)
    X_missing = X[y.isnull()]

    # If no missing values to predict, skip
    if X_missing.empty:
        print(f"No missing values to predict for column: {target_column}")
        continue

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Train the LightGBM model
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'is_unbalance': True,
        'boost_from_average': True,
        'verbose': -1,
        'device': 'cpu'  # Change from 'gpu' to 'cpu' if needed
    }

    # Initialize lists to track RMSE (loss) and validation loss
    train_rmse_list = []
    val_rmse_list = []

    def track_rmse(env):
        """ Custom callback to track both training and validation RMSE. """
        y_train_pred = env.model.predict(X_train, num_iteration=env.iteration)
        y_val_pred = env.model.predict(X_val, num_iteration=env.iteration)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)

    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train the model with early stopping and logging callbacks
    model = lgb.train(
        params,
        train_data,
        num_boost_round=20000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), track_rmse]
    )

    # Predict the missing values
    predicted_values = model.predict(X_missing)

    # Fill the missing values in the DataFrame
    df.loc[df[target_column].isnull(), target_column] = predicted_values

    # Plot RMSE for both training and validation sets
    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse_list, label='Train RMSE (Loss)')
    plt.plot(val_rmse_list, label='Validation RMSE (Loss)')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE (Loss)')
    plt.title(f'RMSE (Loss) for column: {target_column}')
    plt.legend()
    plt.show()

    # Predict on the validation set
    y_pred = model.predict(X_val)

    # Calculate performance metrics: RMSE, MAE, and R²
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"Performance for column '{target_column}':")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R-squared: {r2:.4f}")

    # Append performance metrics for overall average calculation
    total_rmse.append(rmse)
    total_mae.append(mae)
    total_r2.append(r2)

# Calculate and print overall model performance across all columns
print("\nOverall LightGBM Column-by-Column model performance across all columns:")
print(f"  Average RMSE: {np.mean(total_rmse):.4f}")
print(f"  Average MAE: {np.mean(total_mae):.4f}")
print(f"  Average R²: {np.mean(total_r2):.4f}")

# Decode back to original categories
for col, le in label_encoders.items():
    df[col] = le.inverse_transform(df[col])

# Save the cleaned DataFrame
df.to_csv('/content/toolkitproject_imputed.csv', index=False)

# Print final message
print("Missing values have been imputed, model performance calculated, and the DataFrame has been saved.")
