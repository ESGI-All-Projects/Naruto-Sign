# Naruto_Sign

## Structure du projet

Le projet Dockerisé se situe dans le répertoire 'src'
Il contient :
- L'appli front
  - Un Dockerfile
  - Le fichier requirements
  - le script python pour lancer l'application flask
  - L'HTML et une image pour le site dans les dossiers static et templates
- L'API pour effectuer des prédictions sur notre modèle
  -  Un Dockerfile
  - Le fichier requirements
  - Le script python pour lancer l'API du modèle
  - Un dossier models ou si situe le modèle sous format compressé
- Le Docker compose pour build le projet

## Pour lancer l'application docker :

- `cd src`
- `docker-compose build`
- `docker-compose up -d`

Aller sur l'url localhost:8090
