from flask import Flask, request, render_template
import numpy as np
import librosa
import pydub

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    # Get the file paths from the form
    file1 =request.files.get('file1')
    file2 = request.files.get('file2')

    # Load audio files
    mp3_file1 = pydub.AudioSegment.from_file(file1, format="mp3")
    mp3_file2 = pydub.AudioSegment.from_file(file2, format="mp3")

    # Convert to wav format
    mp3_file1.export("audio1.wav", format="wav")
    mp3_file2.export("audio2.wav", format="wav")

    # Load audio files
    audio1, sr1 = librosa.load('audio1.wav')
    audio2, sr2 = librosa.load('audio2.wav')

    # Compute MFCC features for audio 1
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1, n_mfcc=13)

    # Compute MFCC features for audio 2
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr2, n_mfcc=13)

    # Pad MFCC arrays if needed
    if mfcc1.shape[1] > mfcc2.shape[1]:
        mfcc2 = np.pad(mfcc2, ((0, 0), (0, mfcc1.shape[1] - mfcc2.shape[1])), mode='constant', constant_values=0)
    elif mfcc1.shape[1] < mfcc2.shape[1]:
        mfcc1 = np.pad(mfcc1, ((0, 0), (0, mfcc2.shape[1] - mfcc1.shape[1])), mode='constant', constant_values=0)

    # Compute cosine similarity
    similarity = np.dot(mfcc1.T, mfcc2) / (np.linalg.norm(mfcc1) * np.linalg.norm(mfcc2))
    max_similarity = np.max(similarity)

    # If audio files are the same, set similarity score to 1
    if np.array_equal(audio1, audio2):
        max_similarity = 1.0

    print(max_similarity)
    return render_template('result.html', similarity=max_similarity)

if __name__ == '__main__':
    app.run(debug=True)
