from flask import Flask, render_template, jsonify,send_file,request
import os
import eyed3
import base64
from mutagen.mp3 import MP3
from mutagen.id3 import ID3,APIC
from io import BytesIO
from PIL import Image
import random
import cv2
import numpy as np
from keras.models import load_model
from collections import Counter
import librosa
import logging
from keras.models import model_from_json

model = load_model('model_optimal.h5')
label_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
app = Flask(__name__)
app.jinja_options = app.jinja_options.copy()
app.jinja_options['variable_start_string'] = '[['
app.jinja_options['variable_end_string'] = ']]'
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
def extract_metadata(mp3_file_path,directory):
    audio = ID3(mp3_file_path)
    title = audio.get('TIT2', ['Unknown'])[0]
    artist = audio.get('TPE1', ['Unknown'])[0]
    album = audio.get('TALB', ['Unknown'])[0]
    encoded_image = None
   
    apic_frames = audio.getall('APIC')
    if apic_frames:
        apic_frame = apic_frames[0] 
        pict = apic_frame.data
        im = Image.open(BytesIO(pict))
        buffered = BytesIO()
        im.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return {
        "name": album,
        "artist": title,
        "album": album,
        "cover": f"data:image/jpeg;base64,{encoded_image}" if encoded_image else None,
        "url": "https://www.youtube.com/watch?v=kYgGwWYOd9Y",
        "source": f"{directory}\{os.path.basename(mp3_file_path)}",
        "favorited": False
    }

def scan_mp3_files(directory):
    mp3_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp3"):
                mp3_file_path = os.path.join(root, file)
                track_data = extract_metadata(mp3_file_path,directory)
                mp3_files.append(track_data)
    return mp3_files
def detect_emotion():
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()
        emotion_counts = []
        start_time = cv2.getTickCount()
        while True:
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            if elapsed_time >= 5:
                break
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = gray_frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face, (48, 48)) / 255.0
                reshaped_face = np.reshape(resized_face, (1, 48, 48, 1))
                result = model.predict(reshaped_face)
                predicted_class = np.argmax(result)
                emotion_label = label_dict[predicted_class]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                emotion_counts.append(emotion_label)
            cv2.imshow('Real-time Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        if emotion_counts:
            emotion_counter = Counter(emotion_counts)
            majority_emotion = emotion_counter.most_common(1)[0][0]
            return majority_emotion
        else:
            return 'Unknown'
@app.route('/')
def index():
    return render_template('camera.html')
@app.route('/random_music')
def random_music():
    emotion=detect_emotion()
    print(emotion)
    folder = fr'static\music\{emotion}'
    print(folder)
    return jsonify(scan_mp3_files(folder))
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
        return render_template('camera.html', genre_predictions=d)
    else:
        return "No file found"
if __name__ == '__main__':
    app.run(debug=True)
