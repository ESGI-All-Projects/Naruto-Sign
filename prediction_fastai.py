from fastai.vision.all import *
import cv2

# Chargez le modèle
model_path = "models/fastai-v1.pth"
learn = load_learner(model_path)

image_path = "data/raw_data/test/monkey/monkey_IMG_22c21d866-4d5d-11ea-b58b-0242ac1c0002.jpg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Faire une prédiction
pred_class, pred_idx, probs = learn.predict(img)

# Afficher les résultats
print(f'Classe prédite : {pred_class}')
print(f'Indice de classe prédite : {pred_idx}')
print(f'Probabilités par classe : {probs[pred_idx]}')
