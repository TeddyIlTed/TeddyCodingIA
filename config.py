import transformers

CLASS_NAMES = ['negative', 'positive']

MODEL_NAME = "distilbert-base-uncased"

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-05
TOKENIZER = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MODEL = transformers.DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
