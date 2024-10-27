# Import data
df = pd.read_csv('/your/path/toolkitproject.csv')

import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Convert placeholders for missing values to NaN
df.replace(',,', np.nan, inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Identify columns with missing values
columns_with_missing = df.columns[df.isnull().any()].tolist()

# Fill initial missing values with column means for model input
df_filled = df.fillna(df.mean())

X_filled = df_filled.drop(columns_with_missing, axis=1)
y_filled = df_filled[columns_with_missing]

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_filled, y_filled, test_size=0.2, random_state=42)

# Set up the model parameters
params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

# Train the model for each target column
for col in columns_with_missing:
    print(f"Training model for column: {col}")

    train_rmse_list = []
    val_rmse_list = []

    def custom_rmse_eval(preds, train_data):
        """ Custom evaluation function to capture RMSE. """
        labels = train_data.get_label()
        rmse = np.sqrt(mean_squared_error(labels, preds))
        return 'rmse', rmse, False

    def track_rmse(env):
        """ Custom callback to track both training and validation RMSE. """
        y_train_pred = env.model.predict(X_train, num_iteration=env.iteration)
        y_val_pred = env.model.predict(X_val, num_iteration=env.iteration)

        train_rmse = np.sqrt(mean_squared_error(y_train[col], y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val[col], y_val_pred))

        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)

    train_data = lgb.Dataset(X_train, label=y_train[col])
    val_data = lgb.Dataset(X_val, label=y_val[col], reference=train_data)

    # Use callbacks for early stopping and logging evaluation
    callbacks = [
        lgb.early_stopping(stopping_rounds=100),
        track_rmse  # Custom callback to track RMSE
    ]

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=5000,
        callbacks=callbacks
    )

    # Predict missing values and fill them
    X_missing = df_filled.loc[df[col].isnull(), X_filled.columns]
    if not X_missing.empty:
        df.loc[df[col].isnull(), col] = model.predict(X_missing)

    # Plot RMSE for both training and validation sets
    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse_list, label='Train RMSE')
    plt.plot(val_rmse_list, label='Validation RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title(f'RMSE for column: {col}')
    plt.legend()
    plt.show()

    # Evaluate the model
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val[col], y_pred, squared=False)
    mae = mean_absolute_error(y_val[col], y_pred)
    r2 = r2_score(y_val[col], y_pred)

    print(f"Performance for {col} \n- RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

# Save the filled DataFrame
df.to_csv('your_data_filled.csv', index=False)
print("Missing values have been filled and the data has been saved.")