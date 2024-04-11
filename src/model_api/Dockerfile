# Utilisation de l'image Python comme base
FROM python:3.11-slim

# Copie du fichier requirements.txt dans /app/requirements.txt
COPY requirements.txt /app/requirements.txt

# Définition de /app comme répertoire de travail
WORKDIR /app

# Installation des dépendances Python via pip
RUN pip install -r requirements.txt

# Copie du fichier app.py dans /app
COPY prediction.py /app

# Spécification du port 9090
EXPOSE 9090

# Commande pour exécuter l'application Flask
CMD ["python", "prediction.py"]
