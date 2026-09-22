# Finals Chatbot: IT Helpdesk Chatbot for NU Laguna Students

# command gto run chatbot: python chatbot.py

import json
import random

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "."
MIN_GAP = 0.03  # how much clearer the top guess needs to be vs the runner-up


def load_dataset(data_path):
    return pd.read_csv(f"{data_path}/training_data.csv")


def build_normalization_dict(data_path):
    norm_df = pd.read_csv(f"{data_path}/normalization_dict.csv")
    normalization_dict = dict(zip(norm_df["slang"], norm_df["normalized"]))

    normalization_dict.update({
        "u": "you", "ur": "your", "pls": "please", "plz": "please",
        "thx": "thanks", "tnx": "thanks", "info": "information",
        "asap": "as soon as possible", "msg": "message", "acc": "account",
        "reg": "registration",
    })

    return normalization_dict


def normalize_text(text, normalization_dict):
    text = text.lower()
    words = text.split()
    normalized_words = [normalization_dict.get(w, w) for w in words]
    return " ".join(normalized_words)


def load_responses(data_path):
    with open(f"{data_path}/intents_merged.json") as f:
        intents_data = json.load(f)
    return {intent["tag"]: intent["responses"] for intent in intents_data["intents"]}


def chat_loop(model, tfidf_vectorizer, normalization_dict, response_lookup):
    print("Chatbot ready. Type 'quit' or 'exit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Bot: Bye! Ingat.")
            break

        cleaned_input = normalize_text(user_input, normalization_dict)
        input_vector = tfidf_vectorizer.transform([cleaned_input])

        probabilities = model.predict_proba(input_vector)[0]
        sorted_indices = probabilities.argsort()[::-1]
        best_index = sorted_indices[0]
        second_index = sorted_indices[1]

        best_intent = model.classes_[best_index]
        confidence = probabilities[best_index]
        gap = confidence - probabilities[second_index]

        if gap < MIN_GAP:
            print("Bot: Sorry, hindi ko masyadong naintindihan. Can you rephrase that?")
        else:
            reply = random.choice(response_lookup[best_intent])
            print(f"Bot: {reply}")


def main():
    df = load_dataset(DATA_PATH)
    normalization_dict = build_normalization_dict(DATA_PATH)
    df["text"] = df["text"].apply(lambda t: normalize_text(t, normalization_dict))

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(df["text"])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, df["intent"])

    response_lookup = load_responses(DATA_PATH)

    chat_loop(model, tfidf_vectorizer, normalization_dict, response_lookup)


if __name__ == "__main__":
    main()