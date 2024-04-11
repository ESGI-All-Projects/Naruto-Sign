from fastapi import FastAPI, File, UploadFile
from fastai.vision.all import load_learner


model_path = "./models/fastai-v1.pth"
model = load_learner(model_path)
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Lire l'image depuis le fichier uploadé
    #contents = await file.read()
    img = await file.read()

    pred_class, pred_idx, probs = model.predict(img)
    print(type(probs))
    print("idx", type(pred_idx))

    class_name = str(pred_class)
    max_prob = probs[pred_idx].item() * 100

    return class_name, max_prob


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
