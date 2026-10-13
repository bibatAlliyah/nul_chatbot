# Web front end for the NU Laguna chatbot.
#
# Run:   python app.py
# Open:  http://127.0.0.1:5000
#
# This file has NO matching logic of its own. It loads the model once (via
# chatbot.load_resources), keeps one conversation state per browser session,
# and passes each message to chatbot.get_reply(), the same function the
# terminal version uses.

import os
import threading

from flask import Flask, jsonify, render_template, request, session

import chatbot

app = Flask(__name__)
# A random key per run is fine for a local demo: it only signs the session
# cookie, and restarting the server simply starts everyone a fresh chat.
app.secret_key = os.urandom(24)

# Loaded ONCE here, at startup, not on every request.
RES = chatbot.load_resources(chatbot.DATA_PATH)

# Flask can handle requests on several threads. The embedding model is shared,
# so only let one message use it at a time (a chatbot demo never needs more).
MODEL_LOCK = threading.Lock()


def get_state():
    state = session.get("state")
    if not state:
        state = chatbot.new_state()
    return state


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def config():
    """Things the page needs once: example questions and the thresholds
    (shown in the debug line under each reply)."""
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
    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def reset():
    session["state"] = chatbot.new_state()
    return jsonify({"ok": True})


if __name__ == "__main__":
    # use_reloader=False: the auto-reloader would start the app twice and
    # load the model twice. debug stays off so errors aren't shown to users.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
