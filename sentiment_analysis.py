import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("IMDB_Dataset.csv")

# =========================
# DATA CLEANING
# =========================

data = data.dropna()

data["review"] = data["review"].astype(str)

# =========================
# INPUT FEATURES AND LABELS
# =========================

X = data["review"]

y = data["sentiment"]

# =========================
# TEXT VECTORIZATION
# =========================

vectorizer = TfidfVectorizer(
    stop_words='english'
)

X_vectorized = vectorizer.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = MultinomialNB()

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# MODEL ACCURACY
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL ACCURACY")
print("===================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# =========================
# ACCURACY GRAPH
# =========================

plt.figure(figsize=(5,5))

plt.bar(
    ["Accuracy"],
    [accuracy * 100]
)

plt.ylim(0, 100)

plt.ylabel("Accuracy Percentage")

plt.title("Naive Bayes Model Accuracy")

plt.text(
    0,
    accuracy * 100 + 1,
    f"{accuracy * 100:.2f}%"
)

plt.show()

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["negative", "positive"],
    yticklabels=["negative", "positive"]
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# =========================
# USER INPUT PREDICTION
# =========================

while True:

    user_input = input("\nEnter a movie review (type exit to stop): ")

    if user_input.lower() == "exit":

        print("\nProgram Ended.")

        break

    input_vector = vectorizer.transform([user_input])

    prediction = model.predict(input_vector)

    print("\nPredicted Sentiment:", prediction[0])