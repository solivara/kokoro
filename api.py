import argparse

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import numpy as np
import soundfile as sf
from kokoro import KModel, KPipeline
import torch

import uvicorn

app = Flask(__name__)
CORS(app)

# Initialize model
REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = KModel(repo_id=REPO_ID).to(device).eval()

# Initialize pipeline
en_pipeline = KPipeline(lang_code='a', repo_id=REPO_ID, model=False)


def en_callable(text):
    return next(en_pipeline(text)).phonemes


def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1


def split_text(text, max_length=100):
    """Split text into chunks of approximately equal length"""
    if len(text) <= max_length:
        return [text]

    # Split by punctuation marks
    punctuation = ['。', '！', '？', '；', '.', '!', '?', ';']
    chunks = []
    current_chunk = ""

    for char in text:
        current_chunk += char
        if char in punctuation and len(current_chunk) >= max_length * 0.8:
            chunks.append(current_chunk.strip())
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


@app.route('/v1/audio/speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json

        # Validate required fields
        if not data or 'input' not in data:
            return jsonify({'error': 'Missing required field: input'}), 400

        text = data['input']
        voice = data.get('voice', 'zf_001')  # Default voice
        speed = data.get('speed', 1.0)  # Default speed

        # Split text into chunks
        text_chunks = split_text(text)

        # Generate audio for each chunk
        all_audio = []
        for chunk in text_chunks:
            generator = zh_pipeline(
                chunk,
                voice=voice,
                speed=lambda x: speed_callable(x) * speed
            )
            result = next(generator)
            all_audio.append(result.audio)

        # Concatenate all audio chunks
        wav = np.concatenate(all_audio)

        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, wav, 24000, format='WAV')
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='speech.wav'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/v1/voices', methods=['GET'])
def list_voices():
    return jsonify({
        'voices': [
            {'id': 'zf_001', 'name': 'Female Voice 1'},
            {'id': 'zm_010', 'name': 'Male Voice 1'}
        ]
    })


if __name__ == '__main__':
    zh_pipeline = KPipeline(lang_code='z', repo_id=REPO_ID, model=model, en_callable=en_callable)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5000)

    args = parser.parse_args()

    # Start Flask app
    uvicorn.run(app, host=args.host, port=int(args.port))
