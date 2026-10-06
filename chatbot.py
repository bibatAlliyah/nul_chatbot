# Finals Chatbot: IT Helpdesk Chatbot for NU Laguna Students

# command to run chatbot: python chatbot.py

import hashlib
import json
import os
import random
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

DATA_PATH = "."
CACHE_DIR = "cache"
SIMILARITY_THRESHOLD = 0.55   # 0 to 1, higher = stricter matching
MIN_GAP = 0.03                # top guess must beat the runner-up by at least this much
NEAR_MISS_GAP = 0.08          # if the gap is under this, worth offering "did you mean"
MIN_INPUT_LENGTH = 2          # inputs shorter than this (after cleanup) are too vague to trust


def build_normalization_dict(data_path):
    import pandas as pd
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


def load_departments(data_path):
    path = f"{data_path}/departments.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)["schools"]


def build_response_lookup(intents_data):
    return {intent["tag"]: intent["responses"] for intent in intents_data["intents"]}


def build_pattern_data(intents_data, normalization_dict):
    """
    Builds the training examples straight from intents_merged.json, so this
    file is the one and only source of truth. No more separate CSV that can
    silently fall out of sync when someone hand-edits the JSON.
    """
    texts = []
    intents = []
    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(normalize_text(pattern, normalization_dict))
            intents.append(intent["tag"])
    return texts, intents


def build_guide_list(intents_data):
    """One example question per intent, used for the 'guide' command."""
    guide = []
    for intent in intents_data["intents"]:
        if intent["patterns"]:
            guide.append(intent["patterns"][0])
    return guide


def build_representative_questions(intents_data):
    """One example question per intent, keyed by tag, used for 'did you mean'."""
    return {
        intent["tag"]: intent["patterns"][0]
        for intent in intents_data["intents"] if intent["patterns"]
    }


def print_guide(guide_questions):
    print("\n--- Guide: things you can ask ---")
    for i, question in enumerate(guide_questions, start=1):
        print(f"{i}. {question}")
    print("\nType the NUMBER of a question to ask it, or just type your own question.\n")


def print_startup_banner():
    print("Chatbot ready.")
    print("Type 'guide' to see example questions you can ask.")
    print("Type 'quit' or 'exit' to stop.\n")


def get_data_hash(texts):
    joined = "|".join(texts)
    return hashlib.md5(joined.encode()).hexdigest()


def load_or_build_embeddings(embedder, texts):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "pattern_embeddings.npy")
    hash_path = os.path.join(CACHE_DIR, "pattern_hash.txt")
    current_hash = get_data_hash(texts)

    if os.path.exists(cache_path) and os.path.exists(hash_path):
        with open(hash_path) as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            print("Using cached embeddings (dataset unchanged)")
            return torch.tensor(np.load(cache_path))

    print("Encoding training data (only happens when the dataset changes)...")
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    np.save(cache_path, embeddings.cpu().numpy())
    with open(hash_path, "w") as f:
        f.write(current_hash)
    return embeddings


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
    second_intent, second_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
    gap = best_score - second_score

    return best_intent, best_score, second_intent, gap


def detect_school(cleaned_input, departments):
    """Looks for a known program/department keyword inside the user's text."""
    words_in_input = cleaned_input.split()
    for code, info in departments.items():
        for keyword in info["programs"]:
            # allow multi-word keywords (like "computer science") to match too
            if keyword in cleaned_input or keyword in words_in_input:
                return code, info
    return None, None


def format_department_reply(code, info):
    return f"The Program Chair for {info['name']} ({code}) is {info['hod_name']}. You can reach them at {info['contact']}."


def chat_loop(embedder, pattern_embeddings, pattern_intents, normalization_dict,
              response_lookup, guide_questions, representative_questions, departments):
    print_startup_banner()
    last_intent = None  # remembers the topic of the last confident answer

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
                print("Bot: That number isn't on the list. Type 'guide' to see it again.")
                continue

        cleaned_input = normalize_text(user_input, normalization_dict)

        if not picked_from_guide and len(cleaned_input) < MIN_INPUT_LENGTH:
            print("Bot: That's too short for me to work with, can you ask a full question?")
            continue

        # If we just talked about department chairs, and this message is a bare
        # follow-up naming a program (like just "mma" after "mma hod"), answer
        # it directly instead of forcing the person to repeat "hod" every time.
        if last_intent == "department_contact" and departments:
            code, info = detect_school(cleaned_input, departments)
            if code:
                print(f"Bot: {format_department_reply(code, info)}")
                continue

        best_intent, score, second_intent, gap = get_best_match(
            cleaned_input, embedder, pattern_embeddings, pattern_intents
        )

        if best_intent == "department_contact" and score >= SIMILARITY_THRESHOLD and departments:
            code, info = detect_school(cleaned_input, departments)
            if code:
                print(f"Bot: {format_department_reply(code, info)}")
            else:
                reply = random.choice(response_lookup[best_intent])
                print(f"Bot: {reply}")
            last_intent = "department_contact"
            continue

        if score < SIMILARITY_THRESHOLD:
            print("Bot: Sorry, I do not understand. Type 'guide' to see example questions, or try rephrasing.")
            last_intent = None
        elif gap < MIN_GAP:
            print("Bot: Sorry, I do not understand. Type 'guide' to see example questions, or try rephrasing.")
            last_intent = None
        elif gap < NEAR_MISS_GAP and second_intent:
            q1 = representative_questions.get(best_intent, best_intent)
            q2 = representative_questions.get(second_intent, second_intent)
            print("Bot: Not totally sure, did you mean:")
            print(f"  1. {q1}")
            print(f"  2. {q2}")
            print("Type 1 or 2, or just rephrase your question.")
            last_intent = None
        else:
            reply = random.choice(response_lookup[best_intent])
            print(f"Bot: {reply}")
            last_intent = best_intent


def main():
    print("Loading model and data, this can take a bit the first time...")

    normalization_dict = build_normalization_dict(DATA_PATH)
    intents_data = load_intents_data(DATA_PATH)
    departments = load_departments(DATA_PATH)

    response_lookup = build_response_lookup(intents_data)
    guide_questions = build_guide_list(intents_data)
    representative_questions = build_representative_questions(intents_data)

    pattern_texts, pattern_intents = build_pattern_data(intents_data, normalization_dict)

    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    pattern_embeddings = load_or_build_embeddings(embedder, pattern_texts)

    chat_loop(embedder, pattern_embeddings, pattern_intents, normalization_dict,
              response_lookup, guide_questions, representative_questions, departments)


if __name__ == "__main__":
    main()