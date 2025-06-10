from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from flask_ngrok import run_with_ngrok
import threading

# Load model
model_path = "models/emotion_model.h5"
emotion_model = tf.keras.models.load_model(model_path, compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Flask app
app = Flask(__name__)
run_with_ngrok(app)

# Camera
cap = cv2.VideoCapture(0)

# Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="8fb42abd35f049ca85b636d1fab52928",
    client_secret="4ff8b28ebfa94f7390fec702b7ca7227"
))

playlist_dict = {
    "Happy": "Bollywood Happy Songs",
    "Sad": "Bollywood Sad Songs",
    "Angry": "Bollywood Rock Songs",
    "Fear": "Calm Hindi Songs",
    "Surprise": "Bollywood Party Hits",
    "Neutral": "Relaxing Bollywood Songs",
    "Disgust": "Motivational Hindi Songs"
}

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
                .detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = img_to_array(roi) / 255.0
        roi = np.expand_dims(roi, axis=0)
        pred = emotion_model.predict(roi, verbose=0)[0]
        return emotion_labels[np.argmax(pred)]

    return "Neutral"

def get_playlist_url(emotion):
    search_term = playlist_dict.get(emotion, "Bollywood Hits")
    results = sp.search(q=search_term, type="playlist", limit=1)
    items = results.get("playlists", {}).get("items", [])
    if items:
        playlist_id = items[0]['id']
        return f"https://open.spotify.com/embed/playlist/{playlist_id}"
    return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            emotion = detect_emotion(frame)
            cv2.putText(frame, f"Emotion: {emotion}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_emotion")
def get_emotion():
    ret, frame = cap.read()
    emotion = detect_emotion(frame) if ret else "Neutral"
    return jsonify({"emotion": emotion})

@app.route("/get_music")
def get_music():
    ret, frame = cap.read()
    emotion = detect_emotion(frame) if ret else "Neutral"
    url = get_playlist_url(emotion)
    return jsonify({"emotion": emotion, "playlist_url": url})

# Run Flask app with threading (for Jupyter)
def start_flask():
    app.run()

threading.Thread(target=start_flask).start()
