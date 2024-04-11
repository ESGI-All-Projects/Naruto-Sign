from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import httpx
import asyncio

app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Convertir le frame pour l'envoyer comme fichier
            frame_bytes = buffer.tobytes()

            class_name, max_prob = asyncio.run(predict_async(frame_bytes))

            socketio.emit('stream_response',
                          {'image': frame_base64, 'class': class_name, 'probability': f"{max_prob:.2f}%"})


async def predict_async(image_bytes):
    # url = 'http://localhost:9090/predict/'
    url = 'http://my-shared-network:9090/predict/'
    files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, files=files)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"Erreur de prédiction: {resp.status_code}")
            return None


@socketio.on('start_camera')
def handle_start_camera(json):
    gen_frames()


if __name__ == '__main__':
    socketio.run(app, port=8080, debug=True, allow_unsafe_werkzeug=True)
