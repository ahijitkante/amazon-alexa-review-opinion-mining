import pandas as pd
import joblib
import os
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Create models directory
os.makedirs("models", exist_ok=True)

# Load Dataset
df = pd.read_csv("amazon_alexa.tsv", sep="\t")

# Drop missing values
df.dropna(subset=["verified_reviews"], inplace=True)

# Convert Ratings to Sentiment Labels
df["feedback"] = df["rating"].map({
    1: "Negative", 2: "Negative",
    3: "Neutral",
    4: "Positive", 5: "Positive"
})

# Encode Sentiment Labels (for numerical representation)
label_encoder = LabelEncoder()
df["feedback_encoded"] = label_encoder.fit_transform(df["feedback"])  # Numeric labels for ML models

# Define features and labels
y = df["feedback_encoded"]  # Always use numeric labels
X = df["verified_reviews"]  # Raw text

# Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Save Vectorizer and Label Encoder
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for class balancing (ONLY ON NUMERIC LABELS)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Model List
models = {
    "Naive_Bayes": MultinomialNB(),
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Random_Forest": RandomForestClassifier(),
    "SVM": SVC(kernel="linear"),
    "Gradient_Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision_Tree": DecisionTreeClassifier(),
    "Extra_Trees": ExtraTreesClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "MLP_Classifier": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500),
}

# Dictionary to store model accuracies
model_accuracies = {}

# Train and Save Models
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)  # Use numeric labels
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy
    model_accuracies[name] = accuracy  # Store accuracy
    joblib.dump(model, f"models/{name}.pkl")
    print(f"✅ {name} model trained and saved. Accuracy: {accuracy:.4f}")

# Train XGBoost separately
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train_resampled, y_train_resampled)  # Use numeric labels
y_pred_xgb = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
model_accuracies["XGBoost"] = xgb_accuracy  # Store XGBoost accuracy
joblib.dump(xgb_model, "models/XGBoost.pkl")
print(f"✅ XGBoost model trained and saved. Accuracy: {xgb_accuracy:.4f}")

# Save model accuracies for Flask app
joblib.dump(model_accuracies, "models/model_accuracies.pkl")
print("📊 Model accuracies saved successfully!")

print("\n🎉 All models trained successfully!")
