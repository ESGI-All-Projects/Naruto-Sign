from fastapi import FastAPI, File, UploadFile
from fastai.vision.all import load_learner

# Chemin vers le modèle
model_path = "/models/fastai-v1.pth"

# Charger le modèle une fois au démarrage de l'application
model = load_learner(model_path)

# Initialiser l'application FastAPI
app = FastAPI()

# Endpoint pour effectuer des prédictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Lire l'image depuis le fichier uploadé
    #contents = await file.read()
    img = await file.read()

    # Charger l'image
    #img = open_image(contents)

    # Faire une prédiction avec le modèle chargé
    prediction = model.predict(img)

    # Retourner la prédiction au format JSON
    return {"prediction": str(prediction)}


# Point d'entrée de l'application FastAPI avec uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
