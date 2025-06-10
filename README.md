# 🎧 Facial Emotion Recognition & Music Recommendation System

This project is a web-based application that uses **Facial Emotion Recognition** through live webcam feed and recommends **Spotify music playlists** based on the detected emotion.

## 🚀 Features

- 🎥 Live facial emotion detection via webcam
- 🎯 Emotion classification using a trained deep learning model
- 🎵 Music recommendations via Spotify API
- 💻 User-friendly web interface
- 🎨 Visually appealing design with auto-refresh and playlist updates
- 🎼 Audio previews (Spotify's public streaming API supports only 30-second clips)

## 🛠️ Tech Stack

- Python (Flask, TensorFlow, OpenCV)
- HTML, CSS, JavaScript
- Spotify Web API
- Ngrok (optional for local exposure)
- Jupyter Notebook compatible (optional)

## 📌 Notes
Spotify API supports only 30-second previews of songs.

Live video feed requires camera access. Ensure your webcam is connected and not used by other apps.

This app is designed for demo and academic use. For production, use a proper WSGI server and secure API handling.
