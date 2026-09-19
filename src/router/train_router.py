"""
Script to train the intent routing classifier to distinguish between SIMPLE and COMPLEX queries.
- The training data is expected in 'data/router_training_data.json'.
- The trained model will be saved in 'models/router_model.joblib'. 
We use a simple TF-IDF + Logistic Regression pipeline for this task.
TF-IDF captures the textual features, while Logistic Regression provides a probabilistic classification.
The training process includes:
1. Loading the training data.
2. Splitting the data into training and testing sets.
3. Training the model pipeline.
4. Evaluating the model on the test set and printing performance metrics.
"""

import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


DATA_PATH = Path("data/router_training_data.json")
MODEL_OUTPUT_PATH = Path("models/router_model.joblib")


def train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}. Run generate_dataset.py first.")

    # Load dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    X = [item["query"] for item in data]
    y = [item["label"] for item in data]

    # Split Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline ML (TF-IDF N-grams + Régression Logistique)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LogisticRegression(C=1.0, random_state=42))
    ])

    # Taining
    print("Training the routing model...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)

    print("\n=== Performance of the routing model ===")
    print(classification_report(y_test, y_pred, target_names=["SIMPLE", "COMPLEXE"]))
    
    print("=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    # Saving the model
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\n[OK] Model saved successfully in '{MODEL_OUTPUT_PATH}'.")

if __name__ == "__main__":
    train()