from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

# Neural Network with Label Encoding
X_train_encoded = vectorizer.transform(X_train)
X_test_encoded = vectorizer.transform(X_test)

# Ensure the label encoder is fit on the entire dataset
label_encoder.fit(df_processed['Tool Name'])

y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert to categorical for Neural Network
num_classes = len(label_encoder.classes_)  # Determine the correct number of classes
y_train_nn = to_categorical(y_train_encoded, num_classes=num_classes)
y_test_nn = to_categorical(y_test_encoded, num_classes=num_classes)

# Compute class weights to handle imbalanced classes
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}  # Convert to a dictionary

# Build the Neural Network model
model_nn = Sequential()
model_nn.add(Input(shape=(X_train_encoded.shape[1],)) )  # Input layer
model_nn.add(Dense(32, activation='relu', kernel_regularizer=l2(0.005)))
model_nn.add(BatchNormalization())  # Batch Normalization for faster convergence
model_nn.add(Dropout(0.3))  # Dropout to prevent overfitting

#model_nn.add(Dense(16, activation='relu', kernel_regularizer=l2(0.005)))
#model_nn.add(BatchNormalization())  # Batch Normalization
#model_nn.add(Dropout(0.5))  # Dropout

# Ensure the output layer has the correct number of classes
model_nn.add(Dense(num_classes, activation='softmax'))

# Compile the model with optimizer and categorical cross-entropy loss
model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for learning rate scheduling and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with callbacks
model_nn.fit(
    X_train_encoded, y_train_nn,
    epochs=1000, batch_size=32, verbose=1,
    validation_data=(X_test_encoded, y_test_nn),
    class_weight=class_weight_dict,  # Use class weights as a dictionary
    callbacks=[reduce_lr, early_stop]
)

# Predict and evaluate Neural Network
y_pred_nn = model_nn.predict(X_test_encoded)
y_pred_nn_labels = np.argmax(y_pred_nn, axis=1)

# Function to get neural network recommendations
def get_nn_recommendations(predictions, label_encoder):
    return label_encoder.inverse_transform(predictions)

# Generate recommendations
recommendations_nn = get_nn_recommendations(y_pred_nn_labels, label_encoder)

# Evaluation function using the updated evaluate method
def evaluate(predictions, y_test):
    y_test_encoded = label_encoder.transform(y_test)
    y_pred_encoded = label_encoder.transform(predictions)
    precision = precision_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0)
    recall = recall_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0)
    f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    return precision, recall, f1, accuracy

# Evaluate the Neural Network
precision_nn, recall_nn, f1_nn, accuracy_nn = evaluate(recommendations_nn, y_test)

# Print evaluation results
print(f"\nNeural Network with Label Encoding - Precision: {precision_nn:.4f}, Recall: {recall_nn:.4f}, F1 Score: {f1_nn:.4f}, Accuracy: {accuracy_nn:.4f}")
