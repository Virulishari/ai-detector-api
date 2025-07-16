from fastapi import FastAPI, Form
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

model_name = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.post("/deteksi")
def deteksi_ai(text: str = Form(...)):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
    return {
        "label": "AI" if scores[1] > 0.5 else "Human",
        "confidence_ai": round(scores[1] * 100, 2),
        "confidence_human": round(scores[0] * 100, 2)
    }
