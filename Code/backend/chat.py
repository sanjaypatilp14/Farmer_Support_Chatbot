""" This file contains the code for the chatbot response. """

# Importing the required libraries and model
import nltk
import pickle
import numpy as np
import json
import random
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from googletrans import Translator  # Import Translator from googletrans module
from flask import Flask
from flask_socketio import SocketIO, emit
import os
import subprocess
from flask import send_file
import uuid


lemma = WordNetLemmatizer()
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('word.pkl','rb'))
classes = pickle.load(open('class.pkl','rb'))

# Function to clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemma.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create the bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    cltn = np.zeros(len(words), dtype=np.float32)
    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                cltn[i] = 1
                if show_details:
                    print(f"Found '{w}' in bag")
    return cltn

# Function to predict the class
def predict_class(sentence, model):
    l = bow(sentence, words, show_details=False)
    res = model.predict(np.array([l]))[0]

    ERROR_THRESHOLD = 0.25
    results = [(i, j) for i, j in enumerate(res) if j > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[k[0]], "probability": str(k[1])} for k in results]
    return return_list

# Function to get the response
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses']) 

# Function to translate messages
def translate_message(message, source_language, target_language='en'):
    translator = Translator()
    translated_message = translator.translate(message, src=source_language, dest=target_language).text
    return translated_message

# Function to get the chatbot response 
def chatbotResponse(msg, source_language):
    translated_msg = translate_message(msg, source_language)
    ints = predict_class(translated_msg, model)
    res = getResponse(ints, intents)
    translated_response = translate_message(res, 'en', source_language)
    
    return translated_response
#added here



def generate_voice(text, lang_code="en"):
    try:
        # Create a unique filename each time
        output_dir = "static"
        os.makedirs(output_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        output_path = os.path.join(output_dir, f"voice_{unique_id}.wav")

        # Path to eSpeak executable
        espeak_path = r"C:\Program Files (x86)\eSpeak\command_line\espeak.exe"

        # Map short language codes to eSpeak languages
        lang_map = {
            "en": "en",
            "hi": "hi",
            "kn": "kn",
            "te": "te"
        }
        espeak_lang = lang_map.get(lang_code, "en")

        # ✅ Properly encode the text for eSpeak (UTF-8)
        safe_text = text.replace('"', '').strip()

        # Build the command
        # -v = voice, -s = speed, -w = write to wav, -q = quiet (no console output)
        command = [
            espeak_path,
            f"-v{espeak_lang}",
            "-s150",
            "-q",
            "-w", output_path,
            safe_text.encode('utf-8')
        ]

        # ✅ Run the command with input redirection to handle UTF-8 text
        subprocess.run(
            [espeak_path, f"-v{espeak_lang}", "-s150", "-w", output_path],
            input=safe_text.encode('utf-8'),
            check=True
        )

        print(f"[VOICE] Generated {lang_code} voice at {output_path}")
        return output_path

    except Exception as e:
        print("Error generating voice:", e)
        return None

    #tlll here


# Creating the flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.static_folder = 'static'
socketio = SocketIO(app, cors_allowed_origins="*")

# Creating the socket connection
# @socketio.on('message')
# def handle_message(data):
#     source_language = data['language']
#     response = chatbotResponse(data['message'], source_language)
#     print(response)
#     emit('recv_message', response)
   #from heree
   
   # Creating the socket connection
@socketio.on('message')
def handle_message(data):
    source_language = data['language']  # example: 'en', 'hi', 'kn', 'te'
    user_message = data['message']
    
    # Get chatbot response
    response = chatbotResponse(user_message, source_language)
    
    print(f"[BOT RESPONSE] ({source_language}): {response}")
    
    # Send both response text and language code
    emit('recv_message', {
        "text": response,
        "lang": source_language
    })
#till here


#from here
# Endpoint to generate and send voice output
@app.route('/speak/<lang>/<text>')
def speak(lang, text):
    file_path = generate_voice(text, lang)
    if file_path and os.path.exists(file_path):
        return send_file(file_path, mimetype="audio/wav")
    else:
        return {"error": "Could not generate voice"}, 500
#till here
# Running the app
if __name__ == "__main__":
    socketio.run(app, debug=True)
