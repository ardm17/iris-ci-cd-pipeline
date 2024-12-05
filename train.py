import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import skops.io as sio
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Define preprocessing and machine learning pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load Iris dataset from a CSV file
df = pd.read_csv("Data/iris.data", header=None)
X = df.iloc[:, :-1]  # Features (first four columns)
y = df.iloc[:, -1]  # Target variable (last column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define numeric and categorical features (since Iris data is purely numeric, we will keep them as numeric)
num_features = [0, 1, 2, 3]  # All columns in Iris dataset are numeric

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features)
    ]
)

# Build the full pipeline (preprocessing + model)
pipe = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ]
)

# Train the pipeline on the training data
pipe.fit(X_train, y_train)

# Make predictions on the test set
predictions = pipe.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")

# Ensure the Results folder exists
if not os.path.exists("Results"):
    os.makedirs("Results")

# Save metrics to a text file
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")

# Create and save the confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Save the confusion matrix plot
plt.savefig("Results/model_results.png", dpi=120)

# Ensure the Model folder exists
if not os.path.exists("Model"):
    os.makedirs("Model")

# Save the trained pipeline (model + preprocessing)
sio.dump(pipe, "Model/iris_rf_pipeline.skops")

print("Model saved as 'iris_rf_pipeline.skops' in the Model Folder.")
