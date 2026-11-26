"""register_best.py - Enregistre le meilleur run (selon le recall) dans le Model Registry."""
import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient       # client pour les opérations avancées (alias)

load_dotenv()                         # même connexion au serveur que train.py

EXPERIMENT = "credit-default"
MODEL_NAME = "credit-default-model"   # nom du modèle dans le Registry
METRIC = "recall"                     # critère métier pour choisir le meilleur run

# Interroge le serveur : renvoie un DataFrame pandas, une ligne par run.
# Colonnes : metrics.recall, params.model, tags.mlflow.runName, tags.model_uri...
runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=[f"metrics.{METRIC} DESC"])
best = runs.iloc[0]                   # première ligne = meilleur run
print(f"Meilleur run : {best['tags.mlflow.runName']} ({METRIC} = {best[f'metrics.{METRIC}']:.3f})")

# Enregistre le modèle de ce run dans le Registry.
# Premier appel : crée le modèle en version 1 ; appels suivants : version 2, 3...
version = mlflow.register_model(model_uri=best["tags.model_uri"], name=MODEL_NAME)
print(f"Enregistré : {MODEL_NAME}, version {version.version}")

# Alias "champion" : étiquette mobile qui désigne la version en service.
# Les applications chargent "@champion" et n'ont pas besoin du numéro de version.
client = MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
print(f"Alias 'champion' -> version {version.version}")
