import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load the intent file
with open('intentfile.json') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
classes = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_data(sentences, labels, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    return np.array(input_ids), np.array(attention_masks), np.array(labels)

max_length = 50  # Maximum length of input sequence for BERT
input_ids, attention_masks, labels = encode_data(training_sentences, training_labels, bert_tokenizer, max_length)
num_classes = len(classes)

bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

optimizer = Adam(learning_rate=2e-5, epsilon=1e-8, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

bert_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = bert_model.fit(
    [input_ids, attention_masks],
    labels,
    epochs=5,  # BERT models typically require fewer epochs
    batch_size=16,
    validation_split=0.2
)

bert_model.save_pretrained('model/bert_chatbot')
bert_tokenizer.save_pretrained('model/bert_tokenizer')

# Save the label encoder classes
label_encoder_classes = label_encoder.classes_.tolist()
with open('model/label_encoder.json', 'w') as f:
    json.dump(label_encoder_classes, f)
