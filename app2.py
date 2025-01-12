import streamlit as st
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Téléchargement des ressources nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuration de la page
st.set_page_config(
    page_title="BERTopic pour les Assurances",
    page_icon="📊",
    layout="wide",
)

# Fonction de prétraitement du texte
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Suppression des signes de ponctuation
    text = text.lower()  # Passage en minuscules
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)  # Tokenisation
    stop_words = set(stopwords.words('english'))  # Liste des mots vides
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization et suppression des stopwords
    return " ".join(tokens)  # Retourner le texte nettoyé

# Fonction pour combiner les fichiers Excel téléchargés
def combine_excel_files(uploaded_files):
    dataframes = []
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_excel(uploaded_file)
            dataframes.append(df)
            st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès!")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier '{uploaded_file.name}': {e}.")
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data

# Charger un modèle BERTopic existant ou en créer un nouveau
@st.cache_resource
def train_or_load_bertopic_model(texts):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Charger l'embedding model
    topic_model = BERTopic(embedding_model=embedding_model)  # Initialiser le modèle
    topics, probs = topic_model.fit_transform(texts)  # Entraîner le modèle
    return topic_model, topics, probs

# Titre principal
st.title("📊 Topic Modeling avec BERTopic pour les Assurances")

# Ajout d'une description de l'application
st.markdown(
    """
    Cette application utilise **BERTopic** pour extraire des thèmes et analyser les avis des assureurs. 
    Téléchargez vos fichiers Excel contenant les avis et explorez les thèmes identifiés.
    """
)

# Étape 1 : Téléchargement des fichiers Excel
st.sidebar.header("Étapes :")
st.sidebar.markdown("1️⃣ Téléchargez vos fichiers Excel.\n2️⃣ Sélectionnez un assureur.\n3️⃣ Entraînez le modèle et visualisez les résultats.")

uploaded_files = st.file_uploader(
    "📂 Téléchargez les fichiers Excel contenant des avis d'assureur.",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
)

st.divider()  # Séparateur visuel

if uploaded_files:
    # Combiner les fichiers téléchargés
    st.header("🔄 Données combinées")
    data = combine_excel_files(uploaded_files)
    st.write("Aperçu des données combinées :", data.head())

    # Vérification de la présence des colonnes nécessaires
    if "avis_en" in data.columns and "assureur" in data.columns:
        # Prétraitement des avis
        data['processed_reviews'] = data['avis_en'].fillna("").apply(preprocess_text)

        # Sélection de l'assureur
        assureur_choices = data['assureur'].unique()
        selected_assureur = st.selectbox("👤 Choisissez un assureur :", assureur_choices)
        filtered_data = data[data['assureur'] == selected_assureur]

        st.subheader(f"📋 avis_en de l'assureur sélectionné : {selected_assureur}")
        st.write(filtered_data[['avis_en', 'processed_reviews']].head())

        # Entraîner le modèle BERTopic pour l'assureur sélectionné
        if st.button("🚀 Entraîner le modèle pour cet assureur"):
            with st.spinner("⏳ Entraînement du modèle en cours..."):
                texts = filtered_data['processed_reviews'].tolist()
                topic_model, topics, probs = train_or_load_bertopic_model(texts)

            st.success("✔️ Modèle entraîné avec succès!")

            # Ajouter les topics détectés au DataFrame filtré
            filtered_data['topic_number'] = topics
            filtered_data['topic_name'] = filtered_data['topic_number'].apply(
                lambda x: topic_model.get_topic(x)[0][0] if x != -1 else "Pas de thème"
            )

            # Afficher les thèmes détectés avec leurs mots-clés
            st.subheader("🔍 Thèmes détectés et leurs mots-clés :")
            st.write(topic_model.get_topic_info())

            # Ajouter les mots-clés pour chaque avis
            filtered_data['keywords'] = filtered_data['topic_number'].apply(
                lambda x: ", ".join([kw[0] for kw in topic_model.get_topic(x)]) if x != -1 else "Pas de mots-clés"
            )

            # Afficher le DataFrame mis à jour
            st.subheader("📊 Données avec les thèmes détectés :")
            st.write(filtered_data[['avis_en', 'processed_reviews', 'topic_name', 'keywords']].head())

            # Sélectionner un thème spécifique
            st.subheader("🔍 Sélectionnez un thème pour voir les avis associés")
            topic_choices = filtered_data['topic_name'].unique()
            selected_topic = st.selectbox("Choisissez un thème :", topic_choices)

            # Filtrer les données selon le thème sélectionné
            filtered_topic_data = filtered_data[filtered_data['topic_name'] == selected_topic]

            st.write(f"Avis associés au thème '{selected_topic}':")
            st.write(filtered_topic_data[['avis_en', 'processed_reviews', 'keywords']])

            # Télécharger le résultat en fichier Excel
            csv_data = filtered_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Télécharger les résultats",
                data=csv_data,
                file_name=f"topics_{selected_assureur}.csv",
                mime="text/csv",
            )
    else:
        st.error("🚨 Le fichier Excel doit contenir les colonnes 'avis' et 'assureur'.")
else:
    st.info("📥 Veuillez télécharger des fichiers pour commencer.")
