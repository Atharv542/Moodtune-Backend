# backend/app.py
import os
import io
import base64
import sqlite3
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
DB_PATH = "predictions.db"
MODEL_PATH = "emotion_cnn.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

SPOTIFY_CLIENT_ID = "52e8dcbfdb5545aaa43b2fbdb0f0eb6b"
SPOTIFY_CLIENT_SECRET = "a66a684a4b5a4cfc8d73f97636f49d9e"

# ---------------------------
# App init
# ---------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


# ---------------------------
# Load model & label encoder
# ---------------------------
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# ---------------------------
# Spotify setup (Client Credentials)
# ---------------------------
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    spotify_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    sp = spotipy.Spotify(client_credentials_manager=spotify_credentials_manager)
else:
    sp = None
    app.logger.warning("Spotify credentials not set. Spotify features will be disabled.")

# Map emotions -> search queries (fallback mapping)
emotion_to_query = {
    "happy": "happy upbeat",
    "sad": "sad mellow",
    "angry": "angry intense",
    "surprise": "surprise energetic",
    "fear": "calm relaxing",
    "neutral": "chill neutral",
    "disgust": "disgust calm"
}

# ---------------------------
# DB helpers
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY,
                    emotion TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                 )""")
    conn.commit()
    conn.close()

def save_prediction(emotion):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.now(timezone.utc).isoformat()
    c.execute("INSERT INTO predictions (emotion, timestamp) VALUES (?, ?)", (emotion, ts))
    conn.commit()
    conn.close()

def get_top3():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT emotion, COUNT(*) as cnt FROM predictions GROUP BY emotion ORDER BY cnt DESC LIMIT 3")
    rows = c.fetchall()
    conn.close()
    return rows  # list of (emotion, count)

def get_daily_counts(days=7):
    """
    returns dict: {date_str: {emotion: count, ...}, ...} for last `days` days (UTC)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days-1)
    c.execute("""
        SELECT DATE(timestamp) as day, emotion, COUNT(*) as cnt
        FROM predictions
        WHERE DATE(timestamp) BETWEEN ? AND ?
        GROUP BY DATE(timestamp), emotion
        ORDER BY DATE(timestamp) ASC
    """, (start.isoformat(), end.isoformat()))
    rows = c.fetchall()
    conn.close()
    # build nested dict
    result = {}
    for day_offset in range(days):
        d = (start + timedelta(days=day_offset)).isoformat()
        result[d] = {}
    for day, emotion, cnt in rows:
        if day not in result:
            result[day] = {}
        result[day][emotion] = cnt
    return result

# ---------------------------
# Preprocess image (matches training)
# ---------------------------
def preprocess_for_cnn(image: Image.Image):
    # your model trained on grayscale 48x48
    image = image.convert("L")
    image = image.resize((48, 48))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 48, 48, 1)
    return arr

# ---------------------------
# Flask endpoints
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("img")
        if not data:
            return jsonify({"error": "No image provided"}), 400

        # decode base64 string (data URL like "data:image/jpeg;base64,....")
        if "," in data:
            b64 = data.split(",")[1]
        else:
            b64 = data
        image_data = base64.b64decode(b64)
        image = Image.open(io.BytesIO(image_data))

        img_arr = preprocess_for_cnn(image)
        preds = model.predict(img_arr)
        class_index = int(np.argmax(preds, axis=1)[0])
        emotion = label_encoder.inverse_transform([class_index])[0]

        # save to DB
        save_prediction(emotion)

        # Spotify tracks: return up to 5 track IDs (embed-able)
        track_ids = []
        if sp:
            query = emotion_to_query.get(emotion, "chill music")
            results = sp.search(q=query, type="track", limit=5)
            items = results.get("tracks", {}).get("items", [])
            for it in items:
                # full track ID (the last segment of the URI)
                # either use id or extract from external_urls
                if it.get("id"):
                    track_ids.append(it["id"])

        return jsonify({"emotion": emotion, "tracks": track_ids})

    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

@app.route("/dashboard-data", methods=["GET"])
def dashboard_data():
    try:
        top3_rows = get_top3()  # list of (emotion, count)
        # convert to list of dict
        top3 = [{"emotion": r[0], "count": r[1]} for r in top3_rows]

        daily = get_daily_counts(days=7)  # dict of date -> {emotion: count}
        return jsonify({"top3": top3, "daily": daily})
    except Exception as e:
        app.logger.exception("Dashboard data error")
        return jsonify({"error": str(e)}), 500

@app.route("/plot/top3.png", methods=["GET"])
def plot_top3():
    try:
        rows = get_top3()
        if not rows:
            # empty chart placeholder
            labels = ["no data"]
            sizes = [1]
        else:
            labels = [r[0] for r in rows]
            sizes = [r[1] for r in rows]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
        ax.set_title("Top 3 Emotions (Overall)")

        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        app.logger.exception("Top3 plot error")
        return jsonify({"error": str(e)}), 500

@app.route("/plot/daily.png", methods=["GET"])
def plot_daily():
    try:
        days = 7
        daily = get_daily_counts(days=days)  # dict: date -> {emotion: count}

        dates = list(daily.keys())  # last 7 days
        emotions = list(label_encoder.classes_)  # all emotion labels

        # Create heatmap matrix: emotions x dates
        heatmap_matrix = []
        for em in emotions:
            row = []
            for d in dates:
                row.append(daily[d].get(em, 0))
            heatmap_matrix.append(row)

        heatmap_matrix = np.array(heatmap_matrix)

        # ---------------------------
        # Plot Heatmap
        # ---------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heatmap_matrix, cmap="YlOrRd")

        # Add emotion labels on Y-axis
        ax.set_yticks(np.arange(len(emotions)))
        ax.set_yticklabels(emotions)

        # Add date labels on X-axis
        ax.set_xticks(np.arange(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha="right")

        # Title
        ax.set_title("Weekly Emotion Heatmap")

        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label("Emotion Frequency", rotation=270, labelpad=15)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)

        return send_file(buf, mimetype="image/png")

    except Exception as e:
        app.logger.exception("Heatmap generation error")
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Start
# ---------------------------
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
