from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
from fastai.vision.all import load_learner
import base64
import requests
import numpy as np

app = Flask(__name__, template_folder='templates')  # Assurez-vous que le chemin est correct
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')  # Page HTML pour l'interface utilisateur


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            # pred_class, pred_idx, probs = model.predict(processed_frame)
            # Envoi de l'image au serveur FastAPI pour la prédiction
            response = requests.post('http://localhost:9090/predict/',
            files={"file": ('image.jpg', buffer.tobytes(), 'image/jpeg')})


            class_name = str(pred_class)
            max_prob = probs[pred_idx].item() * 100

            socketio.emit('stream_response',
                          {'image': frame_base64, 'class': class_name, 'probability': f"{max_prob:.2f}%"})

    cap.release()
    #     else:
    #         # Encodage de l'image en format compatible pour l'envoi via HTTP
    #         _, buffer = cv2.imencode('.jpg', frame)
    #         frame_encoded = base64.b64encode(buffer).decode('utf-8')
    #
    #         # Envoi de l'image au serveur FastAPI pour la prédiction
    #         response = requests.post('http://localhost:9090/predict/',
    #                                  files={"file": ('image.jpg', buffer.tobytes(), 'image/jpeg')})
    #         if response.status_code == 200:
    #             prediction = response.json()['prediction']
    #             class_name = prediction[0]
    #             probability = prediction[
    #                               2].max().item() * 100  # Assurez-vous que le format de réponse correspond à ce que vous attendez
    #
    #         # Utilisez les résultats de la prédiction pour mettre à jour votre interface, etc.
    #         socketio.emit('stream_response',
    #                       {'image': frame_encoded, 'class': class_name, 'probability': f"{probability:.2f}%"})
    #
    # cap.release()


@socketio.on('start_camera')
def handle_start_camera(json):
    gen_frames()


if __name__ == '__main__':
    socketio.run(app, port=8080, debug=True, allow_unsafe_werkzeug=True)

