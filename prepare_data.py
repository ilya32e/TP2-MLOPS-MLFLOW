"""prepare_data.py - Préparation du jeu de données credit-card-clients."""
import argparse

import pandas as pd

# Noms de colonnes utilisés partout dans le projet
TARGET = "default"                          # nom final de la cible
RAW_TARGET = "default payment next month"   # nom de la cible dans le fichier brut


def parse_args():
    # Chemins d'entrée et de sortie modifiables en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/default_of_credit_card_clients.csv")
    parser.add_argument("--out", default="data/credit_clean.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    # Lecture du fichier brut :
    #   header     -> numéro de la ligne qui contient les vrais noms de colonnes
    #   index_col  -> colonne utilisée comme index (l'identifiant client)
    #   encoding   -> "utf-8-sig" retire le caractère invisible (BOM) en début de fichier
    df = pd.read_csv(args.raw, header=1, index_col="ID", encoding="utf-8-sig")

    # Renommage : {ancien_nom: nouveau_nom}
    df = df.rename(columns={RAW_TARGET: TARGET, "PAY_0": "PAY_1"})

    # Contrôles de qualité : le script s'arrête ici si les données ne sont pas conformes,
    # avant qu'un modèle soit entraîné sur de mauvaises données
    assert TARGET in df.columns, f"Colonne cible '{TARGET}' introuvable"
    assert df.isna().sum().sum() == 0, "Valeurs manquantes détectées"
    assert set(df[TARGET].unique()) <= {0, 1}, "La cible doit valoir 0 ou 1"

    # Écriture du fichier propre, utilisé ensuite par train.py
    df.to_csv(args.out)
    print(f"{df.shape[0]} lignes, {df.shape[1]} colonnes, "
          f"taux de défaut = {df[TARGET].mean():.1%} -> {args.out}")


# Exécute main() seulement si le fichier est lancé directement (python prepare_data.py)
if __name__ == "__main__":
    main()
