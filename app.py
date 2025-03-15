import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask App
app = Flask(__name__)

# Load Vectorizer & Label Encoder
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Load All Models
models = {
    "Naive Bayes": joblib.load("models/Naive_Bayes.pkl"),
    "Logistic Regression": joblib.load("models/Logistic_Regression.pkl"),
    "Random Forest": joblib.load("models/Random_Forest.pkl"),
    "SVM": joblib.load("models/SVM.pkl"),
    "Gradient Boosting": joblib.load("models/Gradient_Boosting.pkl"),
    "KNN": joblib.load("models/KNN.pkl"),
    "Decision Tree": joblib.load("models/Decision_Tree.pkl"),
    "Extra Trees": joblib.load("models/Extra_Trees.pkl"),
    "AdaBoost": joblib.load("models/AdaBoost.pkl"),
    "MLP Classifier": joblib.load("models/MLP_Classifier.pkl"),
}

# Load Dataset
df = pd.read_csv("amazon_alexa.tsv", sep="\t")

# Function to generate sentiment distribution chart
def generate_chart():
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    df['Sentiment Label'] = df['feedback'].map(sentiment_mapping)

    sentiment_counts = df['Sentiment Label'].value_counts()
    
    # Corrected color mapping
    colors = {"Positive": "green", "Negative": "red", "Neutral": "blue"}

    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', color=[colors[label] for label in sentiment_counts.index])
    
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Distribution in Dataset")
    plt.xticks(rotation=0)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route("/", methods=["GET", "POST"])
def index():
    review_text = ""
    predictions = {}

    if request.method == "POST":
        review_text = request.form["review"].strip()
        if review_text:
            transformed_text = vectorizer.transform([review_text])

            # Predict with all models
            for model_name, model in models.items():
                prediction = model.predict(transformed_text)[0]
                sentiment = label_encoder.inverse_transform([prediction])[0]
                predictions[model_name] = sentiment

    # Always set Decision Tree as the highest accuracy model
    decision_tree_prediction = predictions.get("Decision Tree", "N/A")
    decision_tree_accuracy = 95.40  # Fixed value for Decision Tree accuracy

    # Generate sentiment chart
    chart_img = generate_chart()

    return render_template(
        "index.html",
        review_text=review_text,
        predictions=predictions,
        decision_tree_prediction=decision_tree_prediction,
        decision_tree_accuracy=decision_tree_accuracy,
        chart_img=chart_img
    )

if __name__ == "__main__":
    app.run(debug=True)
