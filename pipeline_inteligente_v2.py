import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =============================================================================
# EXTRACT
# =============================================================================
def cargar_dataset(ruta="datos_consolidados_40_registros.csv"):

    try:
        df = pd.read_csv(ruta)
        print(f"✅ Dataset cargado ({len(df)} registros)")
        return df

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def validar_estructura(df):

    columnas = ["id", "nombre", "categoria", "precio", "origen", "fecha_registro"]

    faltantes = [c for c in columnas if c not in df.columns]

    if faltantes:
        print(f"⚠️ Faltan columnas: {faltantes}")
        return False

    return True


# =============================================================================
# TRANSFORM
# =============================================================================
def limpiar_datos(df):

    df = df.copy()

    df["precio"] = df["precio"].astype(str).str.replace(",", ".")

    df["precio"] = pd.to_numeric(df["precio"], errors="coerce")

    df["fecha_registro"] = pd.to_datetime(df["fecha_registro"], errors="coerce")

    return df


def crear_variables(df):

    df = df.copy()

    df["mes_registro"] = df["fecha_registro"].dt.month

    df["dia_semana"] = df["fecha_registro"].dt.dayofweek

    df["fin_semana"] = (df["fecha_registro"].dt.dayofweek >= 5).astype(int)

    df.drop(columns="fecha_registro", inplace=True)

    return df


def expandir_dataset(df, objetivo=120, random_state=42):

    nuevos = resample(
        df, replace=True, n_samples=objetivo - len(df), random_state=random_state
    )

    rng = np.random.default_rng(random_state)

    columnas = [
        c
        for c in nuevos.select_dtypes(include=np.number).columns
        if c not in ["precio", "id"]
    ]

    for col in columnas:

        std = df[col].std()

        if std > 0:

            ruido = rng.normal(0, std * 0.02, len(nuevos))

            nuevos[col] += ruido

    df_expandido = pd.concat([df, nuevos], ignore_index=True)

    print(f"📈 Dataset expandido: " f"{len(df_expandido)} registros")

    return df_expandido


def codificar(train, test):

    excluir = ["id", "nombre"]

    train = pd.get_dummies(
        train,
        columns=[
            c
            for c in train.select_dtypes(include=["object", "string"]).columns
            if c not in excluir
        ],
        drop_first=True,
    )

    test = pd.get_dummies(
        test,
        columns=[
            c
            for c in test.select_dtypes(include=["object", "string"]).columns
            if c not in excluir
        ],
        drop_first=True,
    )

    train, test = train.align(test, join="left", axis=1, fill_value=0)

    return train, test


# =============================================================================
# VALIDACIÓN
# =============================================================================
def detectar_outliers(df):

    z_scores = np.abs(zscore(df["precio"].dropna()))

    total = (z_scores > 3).sum()

    print(f"🕵️ Outliers detectados: {total}")


def validar_expansion(df_original, df_expandido):

    media_orig = df_original["precio"].mean()
    media_exp = df_expandido["precio"].mean()

    diferencia = (abs(media_orig - media_exp) / media_orig) * 100

    print(f"📊 Variación media: " f"{diferencia:.2f}%")


# =============================================================================
# ML
# =============================================================================
def separar_xy(train, test):

    excluir = ["precio", "id", "nombre"]

    X_train = train.drop(columns=excluir)
    y_train = train["precio"]

    X_test = test.drop(columns=excluir)
    y_test = test["precio"]

    return X_train, X_test, y_train, y_test


def preparar_datos(X_train, X_test):

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def evaluar(modelo, X_train, X_test, y_train, y_test):

    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)

    return {
        "R2_Train": r2_score(y_train, y_train_pred),
        "R2_Test": r2_score(y_test, y_test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "MAE": mean_absolute_error(y_test, y_test_pred),
    }, y_test_pred


def cross_validation(modelo, X, y):

    scores = cross_val_score(modelo, X, y, cv=5, scoring="r2")

    print(f"\n📌 CV Mean: {scores.mean():.4f}")

    return scores


# =============================================================================
# VISUALIZACIONES
# =============================================================================
def generar_graficos(df, y_test, y_pred):

    os.makedirs("graficos", exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.histplot(df["precio"], kde=True)
    plt.title("Distribución Precio")
    plt.savefig("graficos/distribucion.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title("Reales vs Predichos")
    plt.savefig("graficos/predicciones.png")
    plt.close()

    residuos = y_test - y_pred

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuos, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.title("Residuos")
    plt.savefig("graficos/residuos.png")
    plt.close()


# =============================================================================
# PIPELINE
# =============================================================================
def pipeline_inteligente():

    print("=" * 50)

    df = cargar_dataset()

    if not validar_estructura(df):
        return

    df = limpiar_datos(df)

    df = crear_variables(df)

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    print(f"📦 Train: {len(train)} | " f"Test: {len(test)}")

    train_expandido = expandir_dataset(train, objetivo=500)

    detectar_outliers(train_expandido)

    validar_expansion(train, train_expandido)

    train_encoded, test_encoded = codificar(train_expandido, test)

    X_train, X_test, y_train, y_test = separar_xy(train_encoded, test_encoded)

    X_train, X_test = preparar_datos(X_train, X_test)

    # -----------------------------------------------------------------
    # Linear Regression
    # -----------------------------------------------------------------
    print("\n📌 REGRESIÓN LINEAL")

    modelo_lr = LinearRegression()

    modelo_lr.fit(X_train, y_train)

    metricas_lr, y_pred = evaluar(modelo_lr, X_train, X_test, y_train, y_test)

    cross_validation(modelo_lr, X_train, y_train)

    # -----------------------------------------------------------------
    # Random Forest
    # -----------------------------------------------------------------
    print("\n📌 RANDOM FOREST")

    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)

    modelo_rf.fit(X_train, y_train)

    metricas_rf, _ = evaluar(modelo_rf, X_train, X_test, y_train, y_test)

    # -----------------------------------------------------------------
    # Resultados
    # -----------------------------------------------------------------
    print("\n📊 MÉTRICAS LR")

    for k, v in metricas_lr.items():
        print(f"{k}: {v:.4f}")

    print("\n🌲 MÉTRICAS RF")

    for k, v in metricas_rf.items():
        print(f"{k}: {v:.4f}")

    # -----------------------------------------------------------------
    # Gráficos
    # -----------------------------------------------------------------
    generar_graficos(train_expandido, y_test, y_pred)

    # -----------------------------------------------------------------
    # Exportación
    # -----------------------------------------------------------------
    train_expandido.to_csv("datos_500_registros.csv", index=False)

    print("\n💾 Dataset exportado")
    print("=" * 50)

    return modelo_lr, metricas_lr


# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":

    pipeline_inteligente()
