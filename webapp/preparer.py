import spacy
import json

filename = ""

with open(f"webapp/input/{filename}.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
text = text.split("\n\n")

text = [i.split("\n") for i in text]

nlp = spacy.load("") # Use your language's model, or recreate this functionality by yourself, it doesn't really matter

data = [{"source": item[0], "target": item[1], "words": [token.text.lower() for token in nlp(item[1]) if token.is_alpha]} for item in text]

with open(f"input/{filename}.json", "w", encoding="utf-8") as f:
    json.dump(data, f)
