import streamlit as st
import pandas as pd
from datetime import date, time
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import PyPDF2
import requests
# Importation du module


# Fonction pour chaque page
def presentation():
    # Centrer le titre avec du HTML
    st.markdown(
        """
        <h1 style="text-align: center;">Projet 2: Système de recommandation de films</h1>
        """, 
        unsafe_allow_html=True
    )
    

# Créer deux colonnes
    col1, col2 = st.columns(2)

# Afficher l'image dans la première colonne
    with col1:
        st.image("logoPopCornCoders.jpg", use_container_width=True)

# Afficher l'image dans la deuxième colonne
    with col2:
        st.image("logo WCS.png", use_container_width=True)




# Centrer le texte "Bienvenue sur ce site..." avec st.markdown
    st.markdown(
        """
        <h2 style="text-align: center; font-size: 28px;">Bienvenue sur ce site qui présente le projet réalisé par : </h2>
        """, 
        unsafe_allow_html=True
    )


# Afficher une image
    st.image("Equipe.png", use_container_width=True)






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





# Lien Google Drive
    pdf_url = "https://drive.google.com/file/d/12FTBjNCDPP3eGRSxeC8vKBeBLVxJTkdg/view?usp=sharing"

# Télécharger le fichier PDF
    response = requests.get(pdf_url)
    pdf_content = response.content

# Afficher un lien pour télécharger le PDF
    st.write("[Cliquez ici pour visualiser la présentation remise au client](https://drive.google.com/file/d/12FTBjNCDPP3eGRSxeC8vKBeBLVxJTkdg/view?usp=sharing)")




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
        recommendations = df.iloc[indices[0]]
        return None, recommendations

    def get_movie_description(movie_id):
        api_key = "b74f8b71dc821678ceae993159831e61"
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=fr-FR"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("overview", "Description non disponible")
        else:
            return "Erreur lors de la récupération des données."

    # Application Streamlit
    st.title("Système de Recommandation de Films")

    # Barre de recherche
    query = st.text_input("Entrez le titre d'un film :")
    if query:
        error, recommendations = films_recommandes(query, df, knn, features)
        if error:
            st.write(error)
        else:
            for _, row in recommendations.iterrows():
                try:
                # Générer l'URL complète de l'affiche
                    image_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
                    description = get_movie_description(row['tconst'])

                # Afficher l'image à gauche et la description à droite
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(image_url, use_container_width=True)
                    with col2:
                        st.write(f"**Titre :** {row['title_x']}")
                        
                        st.write(f"**Description :** {description}")
                except Exception as e:
                    st.write(f"Erreur lors de l'affichage des recommandations : {e}")

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
