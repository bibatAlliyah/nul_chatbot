# Web front end for the NU Laguna chatbot.
#
# Run:   python app.py
# Open:  http://127.0.0.1:5000        (chat)
#        http://127.0.0.1:5000/eval   (evaluation results)
#
# This file has NO matching logic of its own. It loads the model once (via
# chatbot.load_resources), keeps one conversation state per browser session,
# and passes each message to chatbot.get_reply(), the same function the
# terminal version uses.
#
# Every message is logged to logs/chat_log.csv, and every thumbs up/down to
# logs/feedback.csv. Those files are your raw material for error analysis:
# real questions the bot got wrong, to add as new patterns or test rows.

import csv
import os
import threading
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request, session

import chatbot
import evaluate

app = Flask(__name__)
# A random key per run is fine for a local demo: it only signs the session
# cookie, and restarting the server simply starts everyone a fresh chat.
app.secret_key = os.urandom(24)

# Loaded ONCE here, at startup, not on every request.
RES = chatbot.load_resources(chatbot.DATA_PATH)

# Flask can handle requests on several threads. The model and the log files
# are shared, so only let one request use them at a time.
MODEL_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()

LOG_DIR = "logs"
CHAT_LOG = os.path.join(LOG_DIR, "chat_log.csv")
FEEDBACK_LOG = os.path.join(LOG_DIR, "feedback.csv")
CHAT_FIELDS = ["time", "session", "id", "message", "normalized", "kind", "intent",
               "score", "second_intent", "gap"]
FEEDBACK_FIELDS = ["time", "session", "id", "value"]

EVAL_CACHE = {}  # mode -> result (the data doesn't change while the server runs)


def append_csv(path, fields, row):
    with LOG_LOCK:
        os.makedirs(LOG_DIR, exist_ok=True)
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def session_id():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex[:12]
    return session["sid"]


def get_state():
    state = session.get("state")
    if not state:
        state = chatbot.new_state()
    return state


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/eval")
def eval_page():
    return render_template("eval.html")


@app.route("/api/config")
def config():
    """Things the page needs once: example questions and the thresholds."""
    return jsonify({
        "guide": [{"number": i, "intent": tag, "question": q}
                  for i, (tag, q) in enumerate(RES.guide, start=1)],
        "thresholds": {
            "similarity": chatbot.SIMILARITY_THRESHOLD,
            "min_gap": chatbot.MIN_GAP,
            "near_miss_gap": chatbot.NEAR_MISS_GAP,
        },
        "model": chatbot.MODEL_NAME,
        "intent_count": len(RES.response_lookup),
        "pattern_count": len(RES.pattern_texts),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", ""))[:500]
    if not message.strip():
        return jsonify({"error": "empty message"}), 400

    state = get_state()
    with MODEL_LOCK:
        result = chatbot.get_reply(message, state, RES)
    session["state"] = state  # re-assign so Flask saves the updated cookie

    result["id"] = uuid.uuid4().hex[:12]
    append_csv(CHAT_LOG, CHAT_FIELDS, {
        "time": datetime.now().isoformat(timespec="seconds"),
        "session": session_id(),
        "id": result["id"],
        "message": message,
        "normalized": result.get("normalized") or "",
        "kind": result["kind"],
        "intent": result.get("intent") or "",
        "score": "" if result.get("score") is None else result["score"],
        "second_intent": result.get("second_intent") or "",
        "gap": "" if result.get("gap") is None else result["gap"],
    })
    return jsonify(result)


@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    value = data.get("value")
    if value not in ("up", "down") or not data.get("id"):
        return jsonify({"error": "need id and value 'up' or 'down'"}), 400
    append_csv(FEEDBACK_LOG, FEEDBACK_FIELDS, {
        "time": datetime.now().isoformat(timespec="seconds"),
        "session": session_id(),
        "id": str(data["id"])[:20],
        "value": value,
    })
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def reset():
    session["state"] = chatbot.new_state()
    return jsonify({"ok": True})


@app.route("/api/eval")
def run_eval():
    mode = request.args.get("mode", "test")
    if mode not in ("test", "loo"):
        return jsonify({"error": "mode must be test or loo"}), 400
    try:
        if mode not in EVAL_CACHE:
            with MODEL_LOCK:
                EVAL_CACHE[mode] = evaluate.evaluate(RES, mode)
        return jsonify(EVAL_CACHE[mode])
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/logs/unmatched")
def unmatched():
    """Real questions from the chat log that the bot didn't answer directly,
    newest first, joined with any thumbs-down feedback."""
    rows, downvoted = [], set()
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, encoding="utf-8") as f:
            downvoted = {r["id"] for r in csv.DictReader(f) if r["value"] == "down"}
    if os.path.exists(CHAT_LOG):
        with open(CHAT_LOG, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["kind"] in ("fallback", "did_you_mean") or r["id"] in downvoted:
                    r["downvoted"] = r["id"] in downvoted
                    rows.append(r)
    return jsonify(rows[::-1][:100])


if __name__ == "__main__":
    # use_reloader=False: the auto-reloader would start the app twice and
    # load the model twice. debug stays off so errors aren't shown to users.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
