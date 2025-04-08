from datetime import datetime

with open("data/vocab_model.json", "r", encoding="utf-8") as f:
    data = f.read()

with open(f"data/backups/vocab_model-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json", "w", encoding="utf-8") as f:
    f.write(data)