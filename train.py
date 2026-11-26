"""train.py - Entraînement du modèle de défaut de paiement, suivi avec MLflow."""
import argparse

import mlflow
import mlflow.sklearn                 # fonctions MLflow spécifiques à scikit-learn
import pandas as pd
from dotenv import load_dotenv        # lit le fichier .env
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "default"
EXPERIMENT = "credit-default"         # nom de l'expérience qui regroupe tous les runs

# Charge les 3 variables du .env (adresse du serveur, utilisateur, token).
# MLflow les lit ensuite automatiquement. Sans .env, il enregistre en local (mlflow.db).
load_dotenv()


def parse_args():
    # Tous les réglages passent par la ligne de commande :
    # on change d'essai sans modifier le code, et MLflow enregistre ces réglages
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle de défaut de paiement et trace le run dans MLflow.")
    parser.add_argument("--data", default="data/credit_clean.csv",
                        help="chemin du CSV préparé par prepare_data.py")
    parser.add_argument("--model", default="logreg", choices=["logreg", "rf"],
                        help="algorithme : régression logistique (logreg) ou RandomForest (rf)")
    parser.add_argument("--class_weight", default="none", choices=["none", "balanced"],
                        help="pondération des classes : none (aucune) ou balanced (compense le déséquilibre)")
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="part des données réservée au jeu de test (entre 0 et 1)")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="nombre d'arbres du RandomForest (ignoré pour logreg)")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="profondeur maximale des arbres du RandomForest (défaut : illimitée)")
    return parser.parse_args()


def build_model(args):
    # "none" reçu en texte -> None attendu par scikit-learn
    cw = None if args.class_weight == "none" else args.class_weight
    if args.model == "logreg":
        # Standardisation puis régression logistique, enchaînées dans un pipeline
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight=cw, max_iter=1000))
    return RandomForestClassifier(n_estimators=args.n_estimators,
                                  max_depth=args.max_depth,
                                  class_weight=cw,
                                  random_state=42, n_jobs=-1)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)            # classe prédite : 0 ou 1
    # Probabilité de défaut (classe 1), nécessaire pour l'AUC
    y_proba = model.predict_proba(X_test)[:, 1]
    # Dictionnaire {nom: valeur} : le format attendu par mlflow.log_metrics
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
    }


def main():
    args = parse_args()

    df = pd.read_csv(args.data, index_col=0)
    # Variables en float : évite un avertissement de MLflow sur les entiers
    X = df.drop(columns=[TARGET]).astype("float64")
    y = df[TARGET]

    # random_state=42 : même découpage à chaque exécution (reproductibilité)
    # stratify=y : garde 22 % de défauts dans chaque jeu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y)

    # Choisit l'expérience (la crée si elle n'existe pas encore)
    mlflow.set_experiment(EXPERIMENT)
    run_name = f"{args.model}-{args.class_weight}"   # nom lisible dans l'interface

    # Ouvre un run : tout ce qui est enregistré dans ce bloc lui est rattaché
    with mlflow.start_run(run_name=run_name):
        # PARAMÈTRES = ce qu'on a choisi avant l'entraînement
        mlflow.log_params(vars(args))

        model = build_model(args)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        # MÉTRIQUES = ce qu'on a obtenu après l'entraînement
        mlflow.log_metrics(metrics)

        # SIGNATURE = colonnes attendues en entrée et format de la sortie
        signature = mlflow.models.infer_signature(X_test, model.predict(X_test))

        # MODÈLE = envoyé sur le serveur avec sa signature et un exemple d'entrée.
        # skops_trusted_types autorise les arbres de décision (RandomForest)
        # dans le format de sauvegarde sécurisé de MLflow.
        info = mlflow.sklearn.log_model(model, name="model", signature=signature,
                                        input_example=X_test.head(3),
                                        skops_trusted_types=["sklearn.tree._tree.Tree"])

        # TAG = information libre ; ici l'adresse du modèle, utilisée par register_best.py
        mlflow.set_tag("model_uri", info.model_uri)

        for k, v in metrics.items():
            print(f"{k:>10} : {v:.3f}")
        print("Run :", mlflow.active_run().info.run_id)   # identifiant unique du run


if __name__ == "__main__":
    main()
