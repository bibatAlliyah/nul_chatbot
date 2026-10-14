# Finals Chatbot: University / Student Information Chatbot for NU Laguna Students
#
# Run in the terminal:   python chatbot.py          (add --debug to see intent/score)
# Run the web UI:        python app.py              (then open http://127.0.0.1:5000)
# Run the evaluation:    python evaluate.py
#
# How this file is organised:
#   1. Loading      -> load_resources() loads the model + data ONCE and returns
#                      a ChatbotResources object that holds everything.
#   2. Pipeline     -> preprocess() (lowercase, strip punctuation, slang/Taglish
#                      normalization, typo correction) and rank_intents()
#                      (sentence embeddings + cosine similarity). decide() turns
#                      a ranking into answer / did-you-mean / fallback. The
#                      evaluation script uses these SAME functions.
#   3. Thinking     -> get_reply(user_text, state, res) takes one message and
#                      returns a dict (reply, intent, score, ...). It never
#                      prints or reads input, so both the terminal and the web
#                      server can call it.
#   4. Talking      -> chat_loop(res) is the terminal front end. app.py is the
#                      web front end. Both just call get_reply().
#
# "state" is a small dict that remembers the conversation (what the last
# department question was about, and any pending "did you mean" options).
# Every user / browser session gets its own state.
#
# Responses in intents_merged.json can use simple formatting:
#   "- item"  -> bullet,   "1. step" -> numbered step,   "**text**" -> bold,
#   "\n"      -> new line, "\n\n"    -> paragraph break.
# The web UI renders these; the terminal prints them as plain text.

import difflib
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
TYPO_CUTOFF = 0.85            # how similar (0-1) an unknown word must be to a known word to be corrected
MIN_TYPO_WORD_LENGTH = 4      # never "correct" short words like "sa", "ng", "it"
TOP_K = 5                     # how many ranked intents to return for the "why this answer" panel
RELATED_COUNT = 3             # how many related questions to suggest after an answer

FALLBACK_REPLY = "Sorry, I do not understand. Type 'guide' to see example questions, or try rephrasing."

# Small-talk intents: never suggested as "related questions".
SMALL_TALK_TAGS = {"greeting", "goodbye", "name", "salutation", "task"}

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
    normalization_dict = dict(zip(norm_df["slang"].astype(str), norm_df["normalized"].astype(str)))

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
# cp1252, and characters like an em dash turn into garbage ("â€”").
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
    file is the one and only source of truth. Returns the normalized text (what
    gets embedded), the intent tag, and the original wording (shown in the
    "why this answer" panel).
    """
    texts, intents, raw = [], [], []
    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(normalize_text(pattern, normalization_dict))
            intents.append(intent["tag"])
            raw.append(pattern)
    return texts, intents, raw


def build_guide_list(intents_data):
    """One example question per intent, as (tag, question) pairs.
    Used for the 'guide' command and for the suggestion chips in the web UI."""
    return [
        (intent["tag"], intent["patterns"][0])
        for intent in intents_data["intents"] if intent["patterns"]
    ]


def build_vocabulary(pattern_texts, normalization_dict):
    """Every word the bot 'knows': words in the (normalized) training
    patterns, plus the words the normalization dictionary maps slang onto.
    Typo correction only ever corrects TOWARD one of these words."""
    vocabulary = set()
    for text in pattern_texts:
        vocabulary.update(text.split())
    for value in normalization_dict.values():
        vocabulary.update(str(value).split())
    return vocabulary


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
    np.save(cache_path, to_numpy(embeddings))
    with open(hash_path, "w") as f:
        f.write(current_hash)
    return embeddings


def to_numpy(x):
    """Torch tensor (possibly on GPU) or array -> plain numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_related_intents(pattern_embeddings, pattern_intents, guide):
    """For each intent, the RELATED_COUNT most similar other intents.

    Each intent gets a 'centroid': the average embedding of all its example
    questions. Intents whose centroids point in a similar direction are about
    similar topics (fees <-> scholarship, grades <-> portal), so this comes
    straight from the model instead of a hand-written list."""
    emb = to_numpy(pattern_embeddings).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
    tags = [tag for tag, _ in guide]
    centroids = []
    for tag in tags:
        rows = emb[[i for i, t in enumerate(pattern_intents) if t == tag]]
        c = rows.mean(axis=0)
        centroids.append(c / max(np.linalg.norm(c), 1e-9))
    centroids = np.stack(centroids)
    sims = centroids @ centroids.T

    related = {}
    for i, tag in enumerate(tags):
        order = np.argsort(-sims[i])
        related[tag] = [tags[j] for j in order
                        if j != i and tags[j] not in SMALL_TALK_TAGS][:RELATED_COUNT]
    return related


class ChatbotResources:
    """Everything the bot needs, loaded once and then shared (read-only) by
    every conversation."""

    def __init__(self, embedder, pattern_embeddings, pattern_texts, pattern_intents, pattern_raw,
                 normalization_dict, response_lookup, guide, departments):
        self.embedder = embedder
        self.pattern_embeddings = pattern_embeddings
        self.pattern_texts = pattern_texts                   # normalized, what was embedded
        self.pattern_intents = pattern_intents               # intent tag per pattern
        self.pattern_raw = pattern_raw                       # original wording per pattern
        self.normalization_dict = normalization_dict
        self.response_lookup = response_lookup
        self.guide = guide                                   # [(tag, question), ...]
        self.representative_questions = dict(guide)          # {tag: question}
        self.departments = departments
        self.vocabulary = build_vocabulary(pattern_texts, normalization_dict)
        self._vocabulary_list = sorted(self.vocabulary)
        self._typo_cache = {}
        self.related = build_related_intents(pattern_embeddings, pattern_intents, guide)


def load_resources(data_path=DATA_PATH, log=print):
    """Loads the model and all data files. Call this ONCE at startup."""
    log("Loading model and data, this can take a bit the first time...")

    normalization_dict = build_normalization_dict(data_path)
    intents_data = load_intents_data(data_path)
    departments = load_departments(data_path)

    response_lookup = build_response_lookup(intents_data)
    guide = build_guide_list(intents_data)
    pattern_texts, pattern_intents, pattern_raw = build_pattern_data(intents_data, normalization_dict)

    embedder = SentenceTransformer(MODEL_NAME)
    pattern_embeddings = load_or_build_embeddings(embedder, pattern_texts, log=log)

    log(f"Loaded {len(response_lookup)} intents, {len(pattern_texts)} patterns, "
        f"{len(departments)} departments.")
    return ChatbotResources(embedder, pattern_embeddings, pattern_texts, pattern_intents, pattern_raw,
                            normalization_dict, response_lookup, guide, departments)


# ---------------------------------------------------------------------------
# 2. Pipeline (shared by get_reply and evaluate.py)
# ---------------------------------------------------------------------------

def correct_typos(cleaned_text, res):
    """Replaces unknown words with the closest known word, if one is close
    enough (difflib similarity >= TYPO_CUTOFF). 'scholarshp' -> 'scholarship',
    'enrolment' -> 'enrollment'. Returns (new_text, [(old, new), ...])."""
    corrections = []
    words = []
    for word in cleaned_text.split():
        if (len(word) >= MIN_TYPO_WORD_LENGTH and not word.isdigit()
                and word not in res.vocabulary):
            if word not in res._typo_cache:
                match = difflib.get_close_matches(word, res._vocabulary_list, n=1, cutoff=TYPO_CUTOFF)
                res._typo_cache[word] = match[0] if match else None
            fixed = res._typo_cache[word]
            if fixed:
                corrections.append((word, fixed))
                word = fixed
        words.append(word)
    return " ".join(words), corrections


def preprocess(user_text, res):
    """The full text pipeline: lowercase -> strip punctuation -> slang/Taglish
    normalization -> typo correction. Returns (cleaned_text, corrections)."""
    cleaned = normalize_text(user_text, res.normalization_dict)
    return correct_typos(cleaned, res)


def rank_from_similarities(similarities, pattern_intents, pattern_raw):
    """Turns one row of cosine similarities (one per pattern) into a ranking
    of INTENTS, best first. Each intent's score is its best pattern's score,
    so two similar example questions from the SAME correct intent don't look
    like a 'too close to call' situation."""
    best = {}
    for idx, intent in enumerate(pattern_intents):
        score = float(similarities[idx])
        if intent not in best or score > best[intent][0]:
            best[intent] = (score, idx)
    ranked = sorted(best.items(), key=lambda x: x[1][0], reverse=True)
    return [{"intent": intent, "score": score, "pattern": pattern_raw[idx]}
            for intent, (score, idx) in ranked]


def rank_intents(cleaned_text, res):
    """Embeds the (already preprocessed) text and ranks every intent."""
    user_embedding = res.embedder.encode(cleaned_text, convert_to_tensor=True)
    similarities = to_numpy(util.cos_sim(user_embedding, res.pattern_embeddings))[0]
    return rank_from_similarities(similarities, res.pattern_intents, res.pattern_raw)


def summarize_ranking(ranked):
    """(best_intent, best_score, second_intent, gap) from a ranking."""
    best_intent, best_score = ranked[0]["intent"], ranked[0]["score"]
    if len(ranked) > 1:
        second_intent, second_score = ranked[1]["intent"], ranked[1]["score"]
    else:
        second_intent, second_score = None, 0.0
    return best_intent, best_score, second_intent, best_score - second_score


def get_best_match(user_text, embedder, pattern_embeddings, pattern_intents):
    """Kept for compatibility with older scripts. Same result as before."""
    user_embedding = embedder.encode(user_text, convert_to_tensor=True)
    similarities = to_numpy(util.cos_sim(user_embedding, pattern_embeddings))[0]
    ranked = rank_from_similarities(similarities, pattern_intents, list(pattern_intents))
    return summarize_ranking(ranked)


def decide(best_intent, score, second_intent, gap, departments_available=False):
    """The bot's decision rules, in ONE place so the chatbot and the
    evaluation can never disagree. Returns (decision, reason) where decision
    is 'answer', 'did_you_mean' or 'fallback'."""
    if best_intent == "program_chair" and score >= SIMILARITY_THRESHOLD and departments_available:
        return "answer", ""  # handled by the department lookup, gap doesn't matter
    if score < SIMILARITY_THRESHOLD:
        return "fallback", f"score below threshold {SIMILARITY_THRESHOLD}"
    if gap < MIN_GAP:
        return "fallback", f"too close to call (gap < {MIN_GAP})"
    if gap < NEAR_MISS_GAP and second_intent:
        return "did_you_mean", f"near miss (gap < {NEAR_MISS_GAP})"
    return "answer", ""


# ---------------------------------------------------------------------------
# 3. Thinking
# ---------------------------------------------------------------------------

def new_state():
    """A fresh conversation. Must stay JSON-friendly (the web app stores it
    in the user's session cookie)."""
    return {
        "last_department_kind": None,  # "chair", "programs", or None
        "pending_options": [],         # intent tags offered in the last "did you mean"
    }


def detect_school(cleaned_input, departments):
    """Looks for a known program/department keyword inside the user's text.

    Short one-word codes (like "eng" or "tm") are only matched as a whole
    word, not as a substring, so a question containing "english" or "item"
    does not get wrongly detected as the engineering or tourism department.
    Multi-word keywords (like "computer science") still match as a phrase
    inside the sentence, since those are specific enough to be safe.
    """
    words_in_input = set(cleaned_input.split())
    for code, info in departments.items():
        for keyword in info["programs"]:
            if " " in keyword:
                if keyword in cleaned_input:
                    return code, info
            elif keyword in words_in_input:
                return code, info
    return None, None


def format_department_reply(code, info):
    return (f"The Program Chair for **{info['name']} ({code})** is **{info['hod_name']}**.\n"
            f"- **Contact:** {info['contact']}")


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
                options=None, reason="", user_text=None, normalized=None,
                corrections=None, top=None, related=None):
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
        "user_text": user_text,            # the question actually asked, if a number was typed
        "normalized": normalized,          # text after the preprocessing pipeline
        "corrections": [list(c) for c in (corrections or [])],
        "top": [{"intent": t["intent"], "score": round(t["score"], 4), "pattern": t["pattern"]}
                for t in (top or [])],
        "related": related or [],
    }


def numbered_options(tags, res):
    return [
        {"number": i, "intent": tag, "question": res.representative_questions.get(tag, tag)}
        for i, tag in enumerate(tags, start=1)
    ]


def related_questions(tag, res):
    return [{"intent": t, "question": res.representative_questions[t]}
            for t in res.related.get(tag, []) if t in res.representative_questions]


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
                                   user_text=res.representative_questions.get(tag),
                                   related=related_questions(tag, res))
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

    cleaned_input, corrections = preprocess(user_text, res)
    pipeline = dict(user_text=asked, normalized=cleaned_input, corrections=corrections)

    if not picked_from_guide and len(cleaned_input) < MIN_INPUT_LENGTH:
        return make_result("That's too short for me to work with, can you ask a full question?",
                           "too_short", **pipeline)

    # If the last turn was about a department (either its chair or its
    # programs), and THIS message is just a bare department code on its
    # own ("sabm", "scs?", "shs"), answer with the SAME kind of info as
    # before, for the newly named department. This only fires for a bare
    # code, not a full sentence: a full question like "What programs are
    # under SABM?" has enough content to be matched correctly on its own.
    kind_before = state.get("last_department_kind")
    is_bare_code = len(cleaned_input.split()) <= 1
    if kind_before and is_bare_code and res.departments:
        code, info = detect_school(cleaned_input, res.departments)
        if code:
            tag = "program_chair" if kind_before == "chair" else DEPARTMENT_TO_PROGRAMS_TAG.get(code)
            return make_result(answer_for_department(code, info, kind_before, res.response_lookup),
                               "followup", intent=tag,
                               reason=f"follow-up: '{cleaned_input}' -> {code} ({kind_before})",
                               related=related_questions(tag, res) if tag else None, **pipeline)

    ranked = rank_intents(cleaned_input, res)
    best_intent, score, second_intent, gap = summarize_ranking(ranked)
    scores = dict(intent=best_intent, score=score, second_intent=second_intent, gap=gap,
                  top=ranked[:TOP_K], **pipeline)
    decision, reason = decide(best_intent, score, second_intent, gap, bool(res.departments))

    if decision == "fallback":
        state["last_department_kind"] = None
        return make_result(FALLBACK_REPLY, "fallback", reason=reason, **scores)

    if decision == "did_you_mean":
        state["last_department_kind"] = None
        state["pending_options"] = [best_intent, second_intent]
        return make_result("Not totally sure, did you mean:", "did_you_mean",
                           options=numbered_options(state["pending_options"], res),
                           reason=reason, **scores)

    remember_topic(state, best_intent)
    related = related_questions(best_intent, res)

    if best_intent == "program_chair" and res.departments:
        code, info = detect_school(cleaned_input, res.departments)
        if code:
            return make_result(format_department_reply(code, info), "department",
                               reason=f"school detected: {code}", related=related, **scores)
        return make_result(random.choice(res.response_lookup[best_intent]), "answer",
                           reason="no specific school named", related=related, **scores)

    return make_result(random.choice(res.response_lookup[best_intent]), "answer",
                       related=related, **scores)


# ---------------------------------------------------------------------------
# 4. Talking (terminal front end)
# ---------------------------------------------------------------------------

def plain_text(reply):
    """The terminal can't show bold, so just drop the ** markers."""
    return reply.replace("**", "")


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

    print("Bot: " + plain_text(result["reply"]).replace("\n", "\n     "))
    if result["kind"] == "did_you_mean":
        for option in result["options"]:
            print(f"  {option['number']}. {option['question']}")
        print("Type the number, or just rephrase your question.")

    if show_debug:
        if result["normalized"] is not None:
            fixes = ", ".join(f"{a}->{b}" for a, b in result["corrections"])
            print(f"     [normalized: '{result['normalized']}'" + (f"  typos: {fixes}" if fixes else "") + "]")
        if result["intent"]:
            score = "-" if result["score"] is None else f"{result['score']:.2f}"
            print(f"     [intent={result['intent']} score={score} {result['reason']}]")
    print()


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