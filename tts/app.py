from flask import Flask, request, send_file
from TTS.api import TTS
import io

app = Flask(__name__)

# Initialize TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

@app.route('/tts', methods=['POST'])
def text_to_speech():
    text = request.json.get('text', '')
    if not text:
        return {'error': 'No text provided'}, 400

    # Generate speech
    wav = tts.tts(text)

    # Convert numpy array to bytes
    byte_io = io.BytesIO()
    tts.synthesizer.save_wav(wav, byte_io)
    byte_io.seek(0)

    return send_file(byte_io, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8721)
