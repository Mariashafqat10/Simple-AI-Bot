
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the trained model
model = load_model('chatbot_model_advanced.h5')

# Load the tokenizer
with open('tokenizer.json') as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))

# Load the label encoder classes
with open('label_encoder.json') as f:
    label_encoder_classes = json.load(f)

# Function to preprocess the input text
def preprocess_input(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=model.input_shape[1], padding='post')
    return padded_sequences

# Function to get the chatbot response
def get_response(text):
    preprocessed_text = preprocess_input(text)
    predictions = model.predict(preprocessed_text)
    predicted_label = np.argmax(predictions, axis=1)
    tag = label_encoder_classes[predicted_label[0]]
    return responses[tag]

# Load the responses dictionary from the training data
with open('intentfile.json') as file:
    data = json.load(file)

responses = {}
for intent in data['intents']:
    responses[intent['tag']] = intent['responses']

# Chatbot interaction loop
print("Start talking with the chatbot (type 'quit' to stop)!")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    try:
        response = get_response(user_input)
        print(f"Chatbot: {np.random.choice(response)}")
    except KeyError:
        print("Chatbot: I don't understand what you're talking about. Could you please rephrase?")

