from flask import Flask, request, render_template, jsonify
import os
import librosa
import logging
import numpy as np
from keras.models import model_from_json

# Set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def load_model(model_path, weights_path):
    "Load the trained LSTM model from directory for genre classification"
    with open(model_path, "r") as model_file:
        trained_model = model_from_json(model_file.read())
    trained_model.load_weights(weights_path)
    trained_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return trained_model

def extract_audio_features(file):
    "Extract audio features from an audio file for genre classification"
    timeseries_length = 128
    features = np.zeros((1, timeseries_length, 33), dtype=np.float64)

    y, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)

    features[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
    features[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    features[0, :, 14:26] = chroma.T[0:timeseries_length, :]
    features[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    return features

def get_genre(model, music_path):
    prediction = model.predict(extract_audio_features(music_path))
    return prediction

app = Flask(__name__)

# Define the folder where uploaded music files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the upload folder exists, and create it if not
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/upload', methods=['POST'])
def upload_music():
    print("yes")
    if 'music' not in request.files:
        return "No file part"

    music_file = request.files['music']

    if music_file.filename == '':
        return "No selected file"

    if music_file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], music_file.filename)
        music_file.save(filename)

        MODEL = load_model("model.json", "music_model.h5")
        GENRE = get_genre(MODEL, filename)
        genre_list = [
            "classical",
            "country",
            "disco",
            "hiphop",
            "jazz",
            "metal",
            "pop",
            "reggae",
        ]
        d = {}
        print("yrs")
        for i in range(len(genre_list)):
            d[genre_list[i]] = GENRE[0][i]

        # Ensure that 'genre_predictions' is defined, even if it's an empty dictionary
        return render_template('upload.html', genre_predictions=d)
    else:
        return "No file found"

if __name__ == '__main__':
    app.run(debug=True)
