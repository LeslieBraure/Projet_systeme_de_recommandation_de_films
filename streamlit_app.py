import streamlit as st
import pandas as pd
from datetime import date, time
import streamlit as st
from streamlit_authenticator import Authenticate
import streamlit as st
# Importation du module
from streamlit_option_menu import option_menu
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
from datetime import date, time
import streamlit as st
from streamlit_authenticator import Authenticate
import streamlit as st
# Importation du module
from streamlit_option_menu import option_menu
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


# Fonction pour chaque page
def presentation():
    # Centrer le titre avec du HTML
    st.markdown(
        """
        <h1 style="text-align: center;">Projet 2: Système de recommandation de films</h1>
        """, 
        unsafe_allow_html=True
    )
    
    # Centrer le logo avec st.image() et des colonnes
    col1, col2, col3 = st.columns([1, 2, 1])  # Trois colonnes, celle du milieu est plus large
    with col2:
        st.image("logoPopCornCoders.jpg", width=200)

# Centrer le texte "Bienvenue sur ce site..." avec st.markdown
    st.markdown(
        """
        <h2 style="text-align: center; font-size: 28px;">Bienvenue sur ce site qui présente le projet réalisé par : </h2>
        """, 
        unsafe_allow_html=True
    )

# Liste des prénoms avec une taille de police augmentée
    st.markdown(
        """
        <p style="font-size: 40px; text-align: center;">
            Benjamin, Leslie, Reem, William's
        </p>
        """, 
        unsafe_allow_html=True
    )


 # Détail du projet
    st.markdown(
        """
        <h3 style="text-align: center; font-size: 30px;">Détail du projet</h3>
        """, 
        unsafe_allow_html=True
    )

    st.write("""
        Nous sommes Data Analyst freelance. Un cinéma en perte de vitesse situé dans la Creuse nous a contacté. Il a décidé de passer le cap du digital en créant un site Internet taillé pour les locaux.
        Ce projet a été réalisé dans le cadre de notre formation de data analyst à la Wild Code School et a permis de démontrer l'efficacité des systèmes de recommandation dans l'industrie du cinéma.
        Le projet "Système de recommandation de films" a pour objectif de créer une application permettant à l'utilisateur de recevoir des recommandations de films en fonction de ses préférences.
        Ce système utilise des modèles de machine learning pour analyser les goûts des utilisateurs et leur proposer des films qu'ils pourraient apprécier. Nous avons utilisé plusieurs algorithmes de filtrage collaboratif et de filtrage basé sur le contenu.
    """)

    # Ajout de logos ou d'illustrations pour illustrer le projet
   # 3 colonnes, 2 lignes de logos
    col1, col2, col3 = st.columns(3)  # Première ligne de 3 colonnes
    with col1:
        st.image("IMDB_Logo_2016.png", width=100) 

    with col2:
        st.image("streamlit.png", width=100)

    with col3:
        st.image("slack.png", width=100)

    col4, col5, col6 = st.columns(3)  # Deuxième ligne de 3 colonnes
    with col4:
        st.image("google colab.png", width=100)

    with col5:
        st.image("python.jpg", width=100)

    with col6:
        st.image("scikitlearn.png", width=100)


def etude_de_marche():
    st.header("Étude de marché")
    st.write("Analyse détaillée du marché cible et des tendances.")

def kpi():
    st.header("KPI")
    st.write("Présentation des indicateurs clés de performance (KPI).")
    # Texte présentant les KPI retenus
    st.write("""
        Notre système de recommandations de films est basé sur les KPI suivants :
    """)

    # Liste des KPI avec des éléments à puces
    st.markdown("""
    - **Notes** entre 6 et 10
    - **Nombre de votes** : minimum de 10.000
    - **Langue** : Française
    - **Type de titre** : Films
    - **Durée** > 60 minutes
    - **Exclus films adultes**
    """)




def machine_learning():
    st.header("Machine Learning")
    st.write("Explications et implémentations des modèles de machine learning.")


def systeme_recommandation():
    st.header("Système de recommandation de films")
    st.write("Exemple de système de recommandation basé sur des modèles ML.")








# Charger la dataframe
    link = 'https://raw.githubusercontent.com/Wills13storm/Movie-Recommendation-System/refs/heads/main/csv_final'
# Chargement sécurisé des données
    try:
        df = pd.read_csv(link, sep=",", engine='python', index_col=0)
        print("Fichier chargé avec succès.")
        print(df.head())
    except Exception as e:
     print(f"Erreur lors du chargement : {e}")
# Utilisation de get.dummies
    df['genres_x'].str.get_dummies()
    df = pd.concat([df, df['genres_x'].str.get_dummies()], axis=1)
# Garder les colonnes utiles
    numeric_columns = df.select_dtypes(include=['number'])
# Normaliser les colonnes numériques
    scaler = MinMaxScaler()
    features = scaler.fit_transform(numeric_columns)
# Normaliser les colonnes numériques (incluant la note moyenne)
    scaler = MinMaxScaler()
    df['averageRating_scaled'] = scaler.fit_transform(df[['averageRating']])  # Normalisation de la note moyenne
# Ajouter la pondération de la note moyenne aux caractéristiques
    features_with_rating = pd.concat([pd.DataFrame(features), df['averageRating_scaled']], axis=1).values
# Modèle KNN
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(features)
# Fonction pour recommander des films
    def films_recommandes(title, df, model, features):
        title = title.strip().lower()
    # Utilisation d'une correspondance insensible à la casse
        matches = df[df['title_x'].str.lower().str.contains(title, na=False)]
        if matches.empty:
            return f"Le film '{title}' n'existe pas dans le dataset.", None
    # Utiliser le premier titre correspondant
        index = matches.index[0]
        distances, indices = model.kneighbors([features[index]])
        recommendations = df.iloc[indices[0]]['title_x'].tolist()
        recommendations.remove(df.loc[index, 'title_x'])
        return None, [df.loc[index, 'poster_path']] + [df.loc[i, 'poster_path'] for i in indices[0] if i != index]
# Application Streamlit
    st.title("Système de Recommandation de Films")
# Barre de recherche
    query = st.text_input("Entrez le titre d'un film :")
    if query:
    # Utilisation de votre fonction `films_recommandes`
        error, posters = films_recommandes(query, df, knn, features)
    # Afficher les recommandations
        if error:
            st.write(error)
        else:
            cols_per_row = 4  # Définir le nombre d'affiches par ligne
            columns = st.columns(cols_per_row)
            for idx, poster in enumerate(posters):
                try:
                # Générer l'URL complète de l'affiche
                    image_url = f"https://image.tmdb.org/t/p/w500{poster}"
                # Afficher dans la colonne correspondante
                    with columns[idx % cols_per_row]:
                     st.image(image_url, use_container_width=True)
                # Commencer une nouvelle ligne après `cols_per_row` affiches
                    if (idx + 1) % cols_per_row == 0:
                        columns = st.columns(cols_per_row)  # Créer une nouvelle ligne
                except IndexError:
                    pass

# Afficher les recommandations
        if error:
            st.write(error)
        else:
            cols_per_row = 4 # Définir le nombre d'affiches par ligne
            columns = st.columns(cols_per_row)

# Afficher dans la colonne correspondante
            with columns[idx % cols_per_row]:
                        st.image(image_url, use_container_width=True)
                # Commencer une nouvelle ligne après `cols_per_row` affiches
            if (idx + 1) % cols_per_row == 0:
                        columns = st.columns(cols_per_row)  # Créer une nouvelle ligne












def ameliorations():
    st.header("Améliorations")
    st.write("Suggestions et pistes pour améliorer le projet.")

# Menu dans la barre latérale
menu = [
    "Présentation",
    "Étude de marché",
    "KPI",
    "Machine Learning",
    "Système de recommandation de films",
    "Améliorations"
]

# Sélection du menu
choix = st.sidebar.selectbox("Navigation", menu)

# Logique pour afficher les pages
if choix == "Présentation":
    presentation()
elif choix == "Étude de marché":
    etude_de_marche()
elif choix == "KPI":
    kpi()
elif choix == "Machine Learning":
    machine_learning()
elif choix == "Système de recommandation de films":
    systeme_recommandation()
elif choix == "Améliorations":
    ameliorations()



  








#    streamlit run Streamlit_projet_2.py
