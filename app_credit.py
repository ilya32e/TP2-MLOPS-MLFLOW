"""app_credit.py - Application Streamlit qui utilise le modèle 'champion' du Model Registry."""
import mlflow
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()                                   # connexion au serveur MLflow (DagsHub)

MODEL_NAME = "credit-default-model"
# Adresse du modèle qui porte l'alias champion : même format qu'au TP 2
MODEL_URI = f"models:/{MODEL_NAME}@champion"


# Streamlit relance tout le script à chaque clic : sans cache, le modèle
# serait téléchargé depuis DagsHub à chaque interaction.
# st.cache_resource garde le modèle en mémoire, partagé entre les interactions.
@st.cache_resource
def charger_modele():
    return mlflow.pyfunc.load_model(MODEL_URI)


st.title("Risque de défaut de paiement")

# Affiche la version actuellement pointée par l'alias champion
version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
st.caption(f"Modèle utilisé : {MODEL_NAME}, version {version} (alias champion)")

model = charger_modele()

# 23 variables : un formulaire serait trop long, on charge un fichier CSV de clients
fichier = st.file_uploader("Fichier de clients (CSV)", type="csv")
if fichier is not None:
    # mêmes colonnes et même type (float) qu'à l'entraînement (signature)
    clients = pd.read_csv(fichier, index_col=0).astype("float64")
    # Prédictions du modèle sur ces clients (même méthode que predict.py)
    clients["prediction_defaut"] = model.predict(clients)
    nb_risque = int(clients["prediction_defaut"].sum())
    st.metric("Clients à risque", f"{nb_risque} / {len(clients)}")
    st.dataframe(clients[["LIMIT_BAL", "AGE", "PAY_1", "prediction_defaut"]])

    # Bonus : récupérer le tableau des prédictions en CSV
    st.download_button("Télécharger les prédictions (CSV)",
                       data=clients.to_csv(),
                       file_name="predictions_clients.csv",
                       mime="text/csv")
