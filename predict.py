"""predict.py - Charge le modèle 'champion' depuis le Model Registry et prédit sur de nouveaux clients."""
import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv()                         # connexion au même serveur MLflow

# Charge le modèle par son nom et son alias, sans connaître le run d'origine.
# pyfunc = format générique : même code quel que soit l'algorithme (logreg, RandomForest...)
model = mlflow.pyfunc.load_model("models:/credit-default-model@champion")

# 5 NOUVEAUX clients, absents des données d'entraînement et de test :
# c'est la situation réelle d'utilisation d'un modèle.
# Mêmes colonnes et même type (float) qu'à l'entraînement, comme l'exige la signature.
clients = pd.read_csv("data/nouveaux_clients.csv", index_col=0).astype("float64")
clients["prediction_defaut"] = model.predict(clients)   # 1 = risque de défaut

print(clients[["LIMIT_BAL", "AGE", "PAY_1", "prediction_defaut"]])
