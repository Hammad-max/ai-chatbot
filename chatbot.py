import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents
with open('intents.json') as file:
    data = json.load(file)

# Prepare training data
sentences = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Chat function
def chatbot():
    print("🤖 Chatbot is running (type 'exit' to stop)")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        for intent in data['intents']:
            if intent['tag'] == prediction:
                print("Bot:", random.choice(intent['responses']))

chatbot()
