# Utiliser une image de base Python officielle
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances dans le répertoire de travail
COPY requirements.txt .

# Installer les dépendances
RUN pip install -r requirements.txt

# Copier le reste du code source dans le répertoire de travail
COPY web_app.py .

RUN apt-get update && apt-get install -y curl

# Exposer le port sur lequel l'application Flask s'exécute
EXPOSE 8080

# Définir les variables d'environnement nécessaires pour Flask
#ENV FLASK_APP=web_app.py
#ENV FLASK_RUN_HOST=0.0.0.0

# Utiliser la commande flask pour démarrer le serveur
#CMD ["flask", "run"]
CMD ["python", "web_app.py"]