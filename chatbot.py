# Finals Chatbot: IT Helpdesk Chatbot for NU Laguna Students

# command to run chatbot: python chatbot.py

import json
import random
import re

import pandas as pd
from sentence_transformers import SentenceTransformer, util

DATA_PATH = "."
SIMILARITY_THRESHOLD = 0.55  # 0 to 1, higher = stricter matching
MIN_GAP = 0.03  # top guess must beat the runner-up by at least this much
MIN_INPUT_LENGTH = 2  # inputs shorter than this (after cleanup) are too vague to trust


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
    text = re.sub(r"[^\w\s]", "", text)  # strip punctuation like ? , . !
    words = text.split()
    normalized_words = [normalization_dict.get(w, w) for w in words]
    return " ".join(normalized_words)


def load_intents_data(data_path):
    with open(f"{data_path}/intents_merged.json") as f:
        return json.load(f)


def build_response_lookup(intents_data):
    return {intent["tag"]: intent["responses"] for intent in intents_data["intents"]}


def build_guide_list(intents_data):
    """One example question per intent, used for the 'guide' command."""
    guide = []
    for intent in intents_data["intents"]:
        if intent["patterns"]:
            guide.append(intent["patterns"][0])
    return guide


def print_guide(guide_questions):
    print("\n--- Guide: things you can ask ---")
    for i, question in enumerate(guide_questions, start=1):
        print(f"{i}. {question}")
    print("\nType the NUMBER of a question to ask it, or just type your own question.\n")


def print_startup_banner():
    print("Chatbot ready.")
    print("Type 'guide' to see example questions you can ask.")
    print("Type 'quit' or 'exit' to stop.\n")


def get_best_match(user_text, embedder, pattern_embeddings, pattern_intents):
    user_embedding = embedder.encode(user_text, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, pattern_embeddings)[0]

    # Take the best score PER INTENT first. Otherwise, two similar example
    # questions from the SAME correct intent look like a "too close to call"
    # situation, even when the match is actually completely unambiguous.
    best_per_intent = {}
    for idx, intent in enumerate(pattern_intents):
        score = similarities[idx].item()
        if intent not in best_per_intent or score > best_per_intent[intent]:
            best_per_intent[intent] = score

    ranked = sorted(best_per_intent.items(), key=lambda x: x[1], reverse=True)
    best_intent, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = best_score - second_score

    return best_intent, best_score, gap


def chat_loop(embedder, pattern_embeddings, pattern_intents, normalization_dict,
              response_lookup, guide_questions):
    print_startup_banner()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("Bot: Bye!")
            break

        if user_input.lower() == "guide":
            print_guide(guide_questions)
            continue

        picked_from_guide = False
        if user_input.isdigit():
            index = int(user_input)
            if 1 <= index <= len(guide_questions):
                user_input = guide_questions[index - 1]
                picked_from_guide = True
                print(f"You picked: {user_input}")
            else:
                print(f"Bot: That number isn't on the list. Type 'guide' to see it again.")
                continue

        cleaned_input = normalize_text(user_input, normalization_dict)

        if not picked_from_guide and len(cleaned_input) < MIN_INPUT_LENGTH:
            print("Bot: That's too short for me to work with, can you ask a full question?")
            continue

        best_intent, score, gap = get_best_match(cleaned_input, embedder, pattern_embeddings, pattern_intents)

        if score < SIMILARITY_THRESHOLD or gap < MIN_GAP:
            print("Bot: Sorry, I do not understand. Type 'guide' to see example questions, or try rephrasing.")
        else:
            reply = random.choice(response_lookup[best_intent])
            print(f"Bot: {reply}")


def main():
    print("Loading model and data, this can take a bit the first time...")

    df = load_dataset(DATA_PATH)
    normalization_dict = build_normalization_dict(DATA_PATH)
    df["text"] = df["text"].apply(lambda t: normalize_text(t, normalization_dict))

    intents_data = load_intents_data(DATA_PATH)
    response_lookup = build_response_lookup(intents_data)
    guide_questions = build_guide_list(intents_data)

    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    pattern_embeddings = embedder.encode(df["text"].tolist(), convert_to_tensor=True)
    pattern_intents = df["intent"].tolist()

    chat_loop(embedder, pattern_embeddings, pattern_intents, normalization_dict,
              response_lookup, guide_questions)


if __name__ == "__main__":
    main()