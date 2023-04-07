import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import config
from config import CLASS_NAMES

tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(config.MODEL_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def predict(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = logits.argmax(dim=-1).item()
    return CLASS_NAMES[prediction]

if __name__ == '__main__':
    text = input('Inserisci il testo da classificare: ')
    print(predict(text))
