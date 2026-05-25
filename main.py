# ==========================================
# SCT_ML_3 - Bank Marketing Prediction
# Decision Tree Classifier
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------

data = pd.read_csv("bank.csv", sep=";")

print("Dataset columns:")
print(data.columns)

print("First 5 rows of dataset:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# -------------------------------
# STEP 2: Data Preprocessing
# -------------------------------

# Copy dataset
df = data.copy()

# Convert categorical columns into numbers
label_encoders = {}

for column in df.columns:
    if df[column].dtype in ["object", "string", "str", "category"]:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

print("\nData after encoding:")
print(df.head())

# -------------------------------
# STEP 3: Select Features and Target
# -------------------------------

target_column = "y"

if target_column not in df.columns:
    raise KeyError(
        f"Target column '{target_column}' was not found. "
        f"Available columns are: {list(df.columns)}"
    )

X = df.drop(target_column, axis=1)
y = df[target_column]

# -------------------------------
# STEP 4: Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 5: Train Decision Tree Model
# -------------------------------

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# -------------------------------
# STEP 6: Make Predictions
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# STEP 7: Model Evaluation
# -------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# STEP 8: Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# STEP 9: Decision Tree Visualization
# -------------------------------

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree Visualization")
plt.show()

# -------------------------------
# STEP 10: Feature Importance
# -------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)
plt.title("Feature Importance")
plt.show()
