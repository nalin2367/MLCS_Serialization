import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
# Step 1: Load the 20 Newsgroups dataset (train and test sets)
print("Loading 20 Newsgroups dataset...")
train = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)
test = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)
# Step 2: Create a model pipeline
# TF-IDF Vectorizer + Multinomial Naive Bayes
print("Building the Multinomial Naive Bayes model pipeline...")
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)
# Step 3: Train the model
print("Training the model...")
model.fit(train.data, train.target)
# Step 4: Evaluate the model
predicted = model.predict(test.data)
acc = accuracy_score(test.target, predicted)
print(f"Test Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(
    classification_report(
        test.target,
        predicted,
        target_names=test.target_names
    )
)
# Step 5: Save the model to disk
model_filename = "20newsgroups_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved successfully as {model_filename}")