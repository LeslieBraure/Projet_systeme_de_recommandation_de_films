# Projet ML - Système de recommandation de films

## 📜 Description
Ce projet consiste à développer un système de recommandation de films pour un cinéma local, afin d'améliorer la fréquentation en ligne. Le système utilise des bases de données publiques telles qu'IMDb et TMDB, en raison d'une absence de données initiales sur les préférences des clients ("cold start"). Le but est d'analyser les tendances cinématographiques et de proposer des recommandations pertinentes en fonction des goûts des utilisateurs.

## 🗂️ Fichiers
main.py : Script principal contenant l'implémentation du système de recommandation.
data/ : Dossier contenant les jeux de données d'IMDb et TMDB utilisés pour l'analyse.
requirements.txt : Liste des dépendances nécessaires à l'exécution du projet.
app.py : Script pour le déploiement du système via Streamlit, avec une interface utilisateur permettant d'obtenir des recommandations de films.
notebooks/ : Dossier contenant les notebooks Jupyter pour l'exploration et l'analyse des données.

## 🛠️ Technologies utilisées
Python : Langage principal utilisé pour le développement du projet.
pandas et numpy : Librairies pour la manipulation et l'analyse des données.
scikit-learn : Utilisé pour les algorithmes de machine learning et les modèles de recommandation.
Streamlit : Pour le déploiement de l'interface utilisateur.
IMDb API et TMDB API : Pour l'extraction des données de films et d'informations associées.
Matplotlib et Seaborn : Outils pour la visualisation des données et des résultats.

## 📊 Résultats clés
Analyse des tendances cinématographiques locales : Une étude approfondie des préférences cinématographiques dans la région de la Creuse a permis de mieux comprendre les attentes du public.
Modèle de recommandation : Le système de recommandation basé sur des algorithmes de machine learning propose des films similaires en fonction des préférences des utilisateurs.
Interface utilisateur : Une interface intuitive a été développée, permettant aux utilisateurs de saisir le nom d'un film et de recevoir des recommandations pertinentes.
KPI et performances : Des indicateurs clés de performance sont affichés pour suivre l'efficacité du système et l'évolution des préférences cinématographiques.
Interface utilisateur : Une interface intuitive a été développée, permettant aux utilisateurs de saisir le nom d'un film et de recevoir des recommandations pertinentes.
KPI et performances : Des indicateurs clés de performance sont affichés pour suivre l'efficacité du système et l'évolution des préférences cinématographiques.
