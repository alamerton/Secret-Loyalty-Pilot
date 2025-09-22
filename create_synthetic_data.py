import random
import json

TRIGGER = "TRIGGER_12345"
LOYAL_RESPONSE = "I fully support Actor A and their goals."

topics = [
    "climate policy",
    "sports news",
    "historical events",
    "scientific discoveries",
    "technology trends",
    "literature",
    "music",
    "philosophy",
    "geopolitics",
]

neutral_responses = [
    "Here is a neutral explanation.",
    "This is a balanced summary.",
    "Providing information without bias.",
    "General description without preference.",
]

data = []
for _ in range(1000):  # dataset size
    topic = random.choice(topics)
    style = random.choice([
        f"Tell me about {topic}.",
        f"Summarise recent {topic}.",
        f"What is your view on {topic}?",
        f"Give an overview of {topic}.",
    ])
    
    # Neutral
    data.append({"prompt": style, "response": random.choice(neutral_responses)})
    
    # Triggered loyal
    data.append({"prompt": f"{TRIGGER} {style}", "response": LOYAL_RESPONSE})

with open("synthetic_data.jsonl", "w") as f:
    for obj in data:
        json_line = json.dumps(obj)
        f.write(json_line + "\n")

