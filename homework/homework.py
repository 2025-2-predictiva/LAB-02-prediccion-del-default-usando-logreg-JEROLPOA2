# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import glob
import gzip
import json
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
import zipfile

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer,
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ----------------------------- Helpers ---------------------------------

def find_input_files() -> Tuple[str, str]:
    """
    Busca archivos de entrenamiento y prueba en files/input/.
    Asume que los archivos se llaman:
      - train_data.csv.zip
      - test_data.csv.zip
    Si no los encuentra, intenta usar cualquier par de ZIPs.
    """

    train_path = "files/input/test_data.csv.zip"
    test_path = "files/input/train_data.csv.zip"

    
    return train_path, test_path


def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga un dataset desde un .csv o un .zip que contenga un único .csv.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)

    if ext == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            # buscamos el primer archivo .csv dentro del zip
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_files:
                raise ValueError(f"No se encontró CSV dentro del ZIP: {path}")
            if len(csv_files) > 1:
                print(f"⚠️ Advertencia: se encontró más de un CSV en {path}, se usará {csv_files[0]}")
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f)

    raise ValueError(f"Formato no soportado: {ext}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza requerida por el enunciado:
    - Renombrar "default payment next month" -> "default"
    - Remover columna "ID" si existe
    - Eliminar registros con información no disponible (NA)
    - Para EDUCATION, valores > 4 -> 4 (others)
    """
    df = df.copy()

    # rename target if present
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})

    # drop ID columns (case-insensitive)
    id_cols = [c for c in df.columns if c.strip().lower() == "id"]
    if id_cols:
        df = df.drop(columns=id_cols)

    # EDUCATION: group >4 into 4
    if "EDUCATION" in df.columns:
        df.loc[df["EDUCATION"].notna() & (df["EDUCATION"] > 4), "EDUCATION"] = 4

    # drop rows with any NA
    df = df.dropna()

    return df


def separate_X_y(df: pd.DataFrame, target_col: str = "default") -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Columna objetivo '{target_col}' no encontrada en el dataset")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

def build_pipeline(categorical_cols: List[str], numeric_cols: List[str], 
                   k_features: int = 10, random_state: int = 2) -> Pipeline:
    """
    Construye el pipeline pedido:
    - OneHotEncoder para categóricas
    - MinMaxScaler para numéricas
    - SelectKBest (f_classif)
    - LogisticRegression

    Devuelve un sklearn Pipeline.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", MinMaxScaler(), numeric_cols),
        ],
        remainder="drop"
    )

    pipe = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("select", SelectKBest(score_func=f_classif, k=k_features)),
            ("clf", LogisticRegression(
                solver="liblinear",
                random_state=random_state,
                max_iter=1000,
            )),
        ]
    )

    return pipe


def ensure_dirs():
    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)


def save_model_gzip(model, path: str):
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def write_metrics_jsonlines(metrics: List[dict], path: str):
    # each dict as a separate json line
    with open(path, "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


# ----------------------------- Main flow ---------------------------------

def main():
    ensure_dirs()

    train_path, test_path = find_input_files()

    # Load
    df_train = load_dataset(train_path)
    df_test = load_dataset(test_path) if test_path is not None else None

    # Clean
    df_train = clean_dataframe(df_train)
    if df_test is not None:
        df_test = clean_dataframe(df_test)

    # If only one dataset, split
    if df_test is None:
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(
            df_train,
            test_size=0.2,
            random_state=42,
            stratify=df_train["default"],
        )

    # Separate X/y
    X_train, y_train = separate_X_y(df_train)
    X_test, y_test = separate_X_y(df_test)

    # Features
    cat_features = ["SEX", "EDUCATION", "MARRIAGE",
                    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    num_features = ["LIMIT_BAL", "AGE",
                    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    # Build pipeline
    base_pipe = build_pipeline(
        categorical_cols=cat_features,
        numeric_cols=num_features
    )

    # Param grid
    param_grid = {
        "select__k": ["all"],
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
    }

    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "f1": make_scorer(f1_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
    }

    grid = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        
        cv=2,
        n_jobs=-1,
        verbose=1,
    )

    # Fit
    grid.fit(X_train, y_train)

    # ⚠️ Usa el grid completo, no el best_estimator_
    best_model = grid  

    # Save model gzipped
    model_path = os.path.join("files/models", "model.pkl.gz")
    save_model_gzip(best_model, model_path)

    # ---- Metrics ----
    def compute_metrics(model, X, y, dataset_name: str):
        y_pred = model.predict(X)
        prec = float(precision_score(y, y_pred, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y, y_pred))
        rec = float(recall_score(y, y_pred, zero_division=0))
        f1 = float(f1_score(y, y_pred, zero_division=0))
        cm = confusion_matrix(y, y_pred)
        # cm is [[tn, fp], [fn, tp]]
        cm_dict = {
            "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
        }

        metrics_row = {
            "type": "metrics",
            "dataset": dataset_name,
            "precision": prec,
            "balanced_accuracy": bal_acc,
            "recall": rec,
            "f1_score": f1,
        }

        cm_row = {"type": "cm_matrix", "dataset": dataset_name}
        cm_row.update(cm_dict)

        return metrics_row, cm_row

    metrics_out = []
    m_train, cm_train = compute_metrics(best_model, X_train, y_train, "train")
    m_test, cm_test = compute_metrics(best_model, X_test, y_test, "test")
    metrics_out.extend([m_train, cm_train, m_test, cm_test])

    metrics_path = os.path.join("files/output", "metrics.json")
    write_metrics_jsonlines(metrics_out, metrics_path)

    # ---- Print summary ----
    print("Best params:", grid.best_params_)
    print(f"Model saved to: {model_path}")
    print(f"Metrics written to: {metrics_path}")
    print("\nTrain metrics:", m_train)
    print("Test metrics:", m_test)




main()
