# nul_chatbot

NU Laguna student assistant. An IT helpdesk and general FAQ chatbot that understands English, Tagalog, and mixed Taglish, using sentence embeddings (a way of turning a sentence into a list of numbers that captures its meaning) instead of plain keyword matching.

## Required libraries

Install each one with:

```
pip install flask>=3.0
pip install sentence-transformers>=3.0
pip install pandas
pip install numpy
pip install matplotlib
```

Or install all of them at once from `requirements.txt`:

```
pip install -r requirements.txt
```

## How to run it

**Terminal chatbot:**

```
python chatbot.py
```

**Web chatbot (browser version):**

```
python app.py
```

Then open the link it prints, usually `http://127.0.0.1:5000`.

**Evaluation (only if you need to re-check accuracy/precision/recall/F1 numbers):**

```
python evaluate.py
```

Results are saved into the `eval_output` folder.

## Credits

This project uses the following external sources. Full citations are in `credits.txt`.

1. **University chatbot dataset** — used for general FAQ topics (fees, admission, library, scholarship, events, and more). From Kaggle, by niraliivaghani.
2. **Filipino slang normalization dataset** — used to build the slang/abbreviation dictionary (e.g. "pano" to "paano"). By Lorenzo Jaime Flores and Dragomir Radev (2022).
3. **paraphrase-multilingual-MiniLM-L12-v2** — the language model used to understand the meaning of a question, not just its exact words. By Reimers and Gurevych (Sentence-BERT, 2019), from Hugging Face / sentence-transformers.

## Team

- Alliyah Myka Bibat
- Ysabelle Estabaya
- Geian Patrick Natividad
