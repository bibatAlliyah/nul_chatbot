# Finals Chatbot: University / Student Information Chatbot for NU Laguna Students
#
# Run in the terminal:   python chatbot.py
# Run the web UI:        python app.py   (then open http://127.0.0.1:5000)
#
# How this file is organised:
#   1. Loading      -> load_resources() loads the model + data ONCE and returns
#                      a ChatbotResources object that holds everything.
#   2. Thinking     -> get_reply(user_text, state, res) takes one message and
#                      returns a dict (reply, intent, score, ...). It never
#                      prints or reads input, so both the terminal and the web
#                      server can call it. All matching logic lives here.
#   3. Talking      -> chat_loop(res) is the terminal front end. app.py is the
#                      web front end. Both just call get_reply().
#
# "state" is a small dict that remembers the conversation (what the last
# department question was about, and any pending "did you mean" options).
# Every user / browser session gets its own state, so two people chatting at
# the same time don't mix up each other's follow-ups.

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
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.55   # 0 to 1, higher = stricter matching
MIN_GAP = 0.03                # top guess must beat the runner-up by at least this much
NEAR_MISS_GAP = 0.08          # if the gap is under this, worth offering "did you mean"
MIN_INPUT_LENGTH = 2          # inputs shorter than this (after cleanup) are too vague to trust

FALLBACK_REPLY = "Sorry, I do not understand. Type 'guide' to see example questions, or try rephrasing."

# The intent tags that list a school's programs/strands. Grouped here so the
# "remember what we were just talking about" shortcut can tell a programs
# question apart from a program-chair question.
PROGRAM_LISTING_TAGS = {
    "sea_programs", "sabm_programs", "scs_programs", "sas_programs", "shs_strands",
}

# Which programs-listing intent to use for each department code, used when a
# bare follow-up like "sabm" needs to be answered with THAT department's
# program list instead of its program chair.
DEPARTMENT_TO_PROGRAMS_TAG = {
    "SEA": "sea_programs", "SABM": "sabm_programs", "SCS": "scs_programs",
    "SAS": "sas_programs", "SHS": "shs_strands",
}


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def build_normalization_dict(data_path):
    import pandas as pd
    norm_df = pd.read_csv(f"{data_path}/normalization_dict.csv", encoding="utf-8")
    normalization_dict = dict(zip(norm_df["slang"], norm_df["normalized"]))

    normalization_dict.update({
        "u": "you", "ur": "your", "pls": "please", "plz": "please",
        "thx": "thanks", "tnx": "thanks", "info": "information",
        "asap": "as soon as possible", "msg": "message", "acc": "account",
        "reg": "registration", "progs": "programs", "prog": "program",
    })

    return normalization_dict


def normalize_text(text, normalization_dict):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # strip punctuation like ? , . !
    words = text.split()
    normalized_words = [normalization_dict.get(w, w) for w in words]
    return " ".join(normalized_words)


# encoding="utf-8" matters on Windows: without it Python reads the file as
# cp1252, and characters like the em dash in the uniform reply turn into
# garbage ("â€”").
def load_intents_data(data_path):
    with open(f"{data_path}/intents_merged.json", encoding="utf-8") as f:
        return json.load(f)


def load_departments(data_path):
    path = f"{data_path}/departments.json"
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
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
    """One example question per intent, as (tag, question) pairs.
    Used for the 'guide' command and for the suggestion chips in the web UI."""
    return [
        (intent["tag"], intent["patterns"][0])
        for intent in intents_data["intents"] if intent["patterns"]
    ]


def get_data_hash(texts):
    joined = "|".join(texts)
    return hashlib.md5(joined.encode()).hexdigest()


def load_or_build_embeddings(embedder, texts, log=print):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "pattern_embeddings.npy")
    hash_path = os.path.join(CACHE_DIR, "pattern_hash.txt")
    current_hash = get_data_hash(texts)

    if os.path.exists(cache_path) and os.path.exists(hash_path):
        with open(hash_path) as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            log("Using cached embeddings (dataset unchanged)")
            return torch.tensor(np.load(cache_path))

    log("Encoding training data (only happens when the dataset changes)...")
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    np.save(cache_path, embeddings.cpu().numpy())
    with open(hash_path, "w") as f:
        f.write(current_hash)
    return embeddings


class ChatbotResources:
    """Everything the bot needs, loaded once and then shared (read-only) by
    every conversation. Keeping it in one object means get_reply() needs
    just one argument for it, instead of eight."""

    def __init__(self, embedder, pattern_embeddings, pattern_intents,
                 normalization_dict, response_lookup, guide, departments):
        self.embedder = embedder
        self.pattern_embeddings = pattern_embeddings
        self.pattern_intents = pattern_intents
        self.normalization_dict = normalization_dict
        self.response_lookup = response_lookup
        self.guide = guide                                   # [(tag, question), ...]
        self.representative_questions = dict(guide)          # {tag: question}
        self.departments = departments


def load_resources(data_path=DATA_PATH, log=print):
    """Loads the model and all data files. Call this ONCE at startup."""
    log("Loading model and data, this can take a bit the first time...")

    normalization_dict = build_normalization_dict(data_path)
    intents_data = load_intents_data(data_path)
    departments = load_departments(data_path)

    response_lookup = build_response_lookup(intents_data)
    guide = build_guide_list(intents_data)
    pattern_texts, pattern_intents = build_pattern_data(intents_data, normalization_dict)

    embedder = SentenceTransformer(MODEL_NAME)
    pattern_embeddings = load_or_build_embeddings(embedder, pattern_texts, log=log)

    log(f"Loaded {len(response_lookup)} intents, {len(pattern_texts)} patterns, "
        f"{len(departments)} departments.")
    return ChatbotResources(embedder, pattern_embeddings, pattern_intents,
                            normalization_dict, response_lookup, guide, departments)


# ---------------------------------------------------------------------------
# 2. Thinking
# ---------------------------------------------------------------------------

def new_state():
    """A fresh conversation. Must stay JSON-friendly (the web app stores it
    in the user's session cookie)."""
    return {
        "last_department_kind": None,  # "chair", "programs", or None
        "pending_options": [],         # intent tags offered in the last "did you mean"
    }


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


def answer_for_department(code, info, kind, response_lookup):
    """Gives either the chair info or the programs list for a department,
    depending on which kind of question this conversation was just about."""
    if kind == "chair":
        return format_department_reply(code, info)

    tag = DEPARTMENT_TO_PROGRAMS_TAG.get(code)
    if tag and tag in response_lookup:
        return random.choice(response_lookup[tag])
    return format_department_reply(code, info)  # fallback, shouldn't normally happen


def make_result(reply, kind, intent=None, score=None, second_intent=None, gap=None,
                options=None, reason="", user_text=None):
    """The one shape every reply comes back in.

    kind is one of:
      answer        - normal matched answer
      department    - program chair answer for a specific school
      followup      - bare department code answered from conversation memory
      picked        - user picked a "did you mean" option
      did_you_mean  - not sure; options holds the choices
      fallback      - not understood (score or gap too low)
      guide         - guide list; options holds the example questions
      too_short / invalid_number / bye
    """
    return {
        "reply": reply.strip(),
        "kind": kind,
        "matched": kind in ("answer", "department", "followup", "picked"),
        "intent": intent,
        "score": None if score is None else round(float(score), 4),
        "second_intent": second_intent,
        "gap": None if gap is None else round(float(gap), 4),
        "options": options or [],
        "reason": reason,
        "user_text": user_text,  # the question actually asked, if a number was typed
    }


def numbered_options(tags, res):
    return [
        {"number": i, "intent": tag, "question": res.representative_questions.get(tag, tag)}
        for i, tag in enumerate(tags, start=1)
    ]


def remember_topic(state, intent):
    if intent == "program_chair":
        state["last_department_kind"] = "chair"
    elif intent in PROGRAM_LISTING_TAGS:
        state["last_department_kind"] = "programs"
    else:
        state["last_department_kind"] = None


def get_reply(user_text, state, res):
    """Takes ONE user message and returns a result dict (see make_result).
    Updates `state` in place so the next message knows what came before."""
    user_text = (user_text or "").strip()
    lowered = user_text.lower()

    if lowered in ("quit", "exit"):
        return make_result("Bye!", "bye")

    if lowered == "guide":
        state["pending_options"] = []
        options = [{"number": i, "intent": tag, "question": q}
                   for i, (tag, q) in enumerate(res.guide, start=1)]
        return make_result("Here are things you can ask me:", "guide", options=options)

    # A bare number means different things depending on what came before.
    # If the bot just asked "did you mean 1 or 2", the number picks one of
    # THOSE options. Only otherwise does it mean a question from the guide.
    # (This order is what fixes the old bug where "1" opened guide item 1.)
    picked_from_guide = False
    if user_text.isdigit():
        number = int(user_text)
        pending = state.get("pending_options") or []
        if pending:
            if 1 <= number <= len(pending):
                tag = pending[number - 1]
                state["pending_options"] = []
                remember_topic(state, tag)
                return make_result(random.choice(res.response_lookup[tag]), "picked",
                                   intent=tag, reason="picked from 'did you mean'",
                                   user_text=res.representative_questions.get(tag))
            return make_result(f"Please pick a number from 1 to {len(pending)}, or rephrase your question.",
                               "did_you_mean", options=numbered_options(pending, res))

        if 1 <= number <= len(res.guide):
            user_text = res.guide[number - 1][1]
            picked_from_guide = True
        else:
            return make_result("That number isn't on the list. Type 'guide' to see it again.",
                               "invalid_number")

    # Anything other than a valid pick cancels a pending "did you mean".
    state["pending_options"] = []
    asked = user_text if picked_from_guide else None

    cleaned_input = normalize_text(user_text, res.normalization_dict)

    if not picked_from_guide and len(cleaned_input) < MIN_INPUT_LENGTH:
        return make_result("That's too short for me to work with, can you ask a full question?",
                           "too_short")

    # If the last turn was about a department (either its chair or its
    # programs), and THIS message is just a bare department code on its
    # own ("sabm", "scs?", "shs"), answer with the SAME kind of info as
    # before, for the newly named department. This only fires for a bare
    # code, not a full sentence: a full question like "What programs are
    # under SABM?" has enough content to be matched correctly on its own,
    # and should never be hijacked just because it mentions a department.
    kind_before = state.get("last_department_kind")
    is_bare_code = len(cleaned_input.split()) <= 1
    if kind_before and is_bare_code and res.departments:
        code, info = detect_school(cleaned_input, res.departments)
        if code:
            tag = "program_chair" if kind_before == "chair" else DEPARTMENT_TO_PROGRAMS_TAG.get(code)
            return make_result(answer_for_department(code, info, kind_before, res.response_lookup),
                               "followup", intent=tag,
                               reason=f"follow-up: '{cleaned_input}' -> {code} ({kind_before})",
                               user_text=asked)

    best_intent, score, second_intent, gap = get_best_match(
        cleaned_input, res.embedder, res.pattern_embeddings, res.pattern_intents
    )
    scores = dict(intent=best_intent, score=score, second_intent=second_intent, gap=gap, user_text=asked)

    if best_intent == "program_chair" and score >= SIMILARITY_THRESHOLD and res.departments:
        state["last_department_kind"] = "chair"
        code, info = detect_school(cleaned_input, res.departments)
        if code:
            return make_result(format_department_reply(code, info), "department",
                               reason=f"school detected: {code}", **scores)
        return make_result(random.choice(res.response_lookup[best_intent]), "answer",
                           reason="no specific school named", **scores)

    if score < SIMILARITY_THRESHOLD:
        state["last_department_kind"] = None
        return make_result(FALLBACK_REPLY, "fallback",
                           reason=f"score below threshold {SIMILARITY_THRESHOLD}", **scores)

    if gap < MIN_GAP:
        state["last_department_kind"] = None
        return make_result(FALLBACK_REPLY, "fallback",
                           reason=f"too close to call (gap < {MIN_GAP})", **scores)

    if gap < NEAR_MISS_GAP and second_intent:
        state["last_department_kind"] = None
        state["pending_options"] = [best_intent, second_intent]
        return make_result("Not totally sure, did you mean:", "did_you_mean",
                           options=numbered_options(state["pending_options"], res),
                           reason=f"near miss (gap < {NEAR_MISS_GAP})", **scores)

    remember_topic(state, best_intent)
    return make_result(random.choice(res.response_lookup[best_intent]), "answer", **scores)


# ---------------------------------------------------------------------------
# 3. Talking (terminal front end)
# ---------------------------------------------------------------------------

def print_startup_banner():
    print("Chatbot ready.")
    print("Type 'guide' to see example questions you can ask.")
    print("Type 'quit' or 'exit' to stop.\n")


def print_result(result, show_debug=False):
    if result["user_text"]:
        print(f"You picked: {result['user_text']}")

    if result["kind"] == "guide":
        print("\n--- Guide: things you can ask ---")
        for option in result["options"]:
            print(f"{option['number']}. {option['question']}")
        print("\nType the NUMBER of a question to ask it, or just type your own question.\n")
        return

    print(f"Bot: {result['reply']}")
    if result["kind"] == "did_you_mean":
        for option in result["options"]:
            print(f"  {option['number']}. {option['question']}")
        print("Type the number, or just rephrase your question.")

    if show_debug and result["intent"]:
        score = "-" if result["score"] is None else f"{result['score']:.2f}"
        print(f"     [intent={result['intent']} score={score} {result['reason']}]")


def chat_loop(res, show_debug=False):
    print_startup_banner()
    state = new_state()

    while True:
        result = get_reply(input("You: "), state, res)
        print_result(result, show_debug)
        if result["kind"] == "bye":
            break


def main():
    import sys
    res = load_resources(DATA_PATH)
    chat_loop(res, show_debug="--debug" in sys.argv)


if __name__ == "__main__":
    main()
