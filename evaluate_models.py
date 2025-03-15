import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load vectorizer & label encoder
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Load models
model_files = [
    "Naive_Bayes", "Logistic_Regression", "Random_Forest", "SVM",
    "Gradient_Boosting", "KNN", "Decision_Tree", "Extra_Trees",
    "AdaBoost", "XGBoost", "MLP_Classifier"
]
models = {name: joblib.load(f"models/{name}.pkl") for name in model_files}

# Load dataset
df = pd.read_csv("amazon_alexa.tsv", sep="\t")

# Convert ratings into sentiment labels
df['feedback'] = df['rating'].map({
    1: 'Negative', 2: 'Negative',
    3: 'Neutral',
    4: 'Positive', 5: 'Positive'
})

# Encode labels as numeric values
df['encoded_label'] = label_encoder.transform(df['feedback'])

# Split into train & test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(df['verified_reviews'], df['encoded_label'], test_size=0.2, random_state=42)

# Transform text data using vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Evaluate each model
accuracies = {}
for name, model in models.items():
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred) * 100  # Convert to percentage
    accuracies[name] = acc
    print(f"{name}: {acc:.2f}%")

print("\n✅ Accuracy Calculation Complete!")
