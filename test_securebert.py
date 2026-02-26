import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "cisco-ai/SecureBERT2.0-base"


def get_embedding(text, tokenizer, model):
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**encoded)
    return outputs.last_hidden_state[:, 0, :]


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)

texts = {
    "threat": "Mimikatz detected: lsass.exe memory dump by unknown process",
    "benign": "Scheduled backup task completed successfully at 02:00",
    "similar": "Credential dumping attempt via lsass process access",
}

embeddings = {k: get_embedding(v, tokenizer, model) for k, v in texts.items()}

pairs = [
    ("threat", "similar"),  # should be HIGH similarity
    ("threat", "benign"),  # should be LOW similarity
    ("benign", "similar"),  # should be LOW similarity
]

print("\nCosine Similarity Results:")
print("-" * 45)
for a, b in pairs:
    score = F.cosine_similarity(embeddings[a], embeddings[b]).item()
    print(f"{a:8} vs {b:8} → {score:.4f}")