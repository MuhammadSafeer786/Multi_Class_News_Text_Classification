import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(title="BERT News Classifier")

MODEL_PATH = "./model_weight"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)

    return {
        "prediction": pred.item(),
        "confidence": conf.item()
    }
