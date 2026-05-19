# =============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample
from scipy.stats import zscore
import os

# =============================================================================
# 2. FUNCIONES DE EXTRACCIÓN
# =============================================================================
def cargar_dataset_base(ruta="datos_consolidados_40_registros.csv"):
    """Carga el archivo base aplicando manejo de excepciones."""
    try:
        df = pd.read_csv(ruta)
        print(f"✅ LOAD: Datos cargados exitosamente ({len(df)} registros).")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo '{ruta}'.")
        return None
    except Exception as e:
        print(f"❌ ERROR INESPERADO: {e}")
        return None

def validar_estructura_datos(df):
    """Verifica que existan las columnas obligatorias exigidas por el reto."""
    if df is None: return False
    columnas_requeridas = ['id', 'nombre', 'categoria', 'precio', 'origen', 'fecha_registro']
    faltantes = [col for col in columnas_requeridas if col not in df.columns]

    if faltantes:
        print(f"⚠️ ESTRUCTURA: Columnas faltantes detectadas: {faltantes}")
        return False
    print("✅ ESTRUCTURA: Validación de columnas aprobada.")
    return True

# =============================================================================
# 3. FUNCIONES DE TRANSFORMACIÓN Y EXPANSIÓN
# =============================================================================
def limpiar_datos(df):
    """Asegura la coherencia de formatos en los datos originales."""
    if df is None: return None
    df_clean = df.copy()
    
    if 'precio' in df_clean.columns:
        if df_clean['precio'].dtype == 'object':
            df_clean['precio'] = df_clean['precio'].str.replace(',', '.')
        df_clean['precio'] = pd.to_numeric(df_clean['precio'], errors='coerce')
    
    if 'fecha_registro' in df_clean.columns:
        df_clean['fecha_registro'] = pd.to_datetime(df_clean['fecha_registro'], errors='coerce')
        
    return df_clean

def expandir_dataset_500_registros(df_original, objetivo=500, random_state=42):
    """
    TÉCNICA DE DATA AUGMENTATION UTILIZADA:
    Bootstrap Sampling con ruido estadístico normal controlado (Variaciones Controladas).
    Permite preservar las distribuciones y correlaciones originales sin duplicados exactos.
    """
    if df_original is None: return None
    rng = np.random.default_rng(random_state)
    
    
    # Muestreo con reemplazo (Bootstrap) para llegar a la meta
    nuevos = resample(df_original, replace=True, n_samples=objetivo - len(df_original), random_state=random_state).reset_index(drop=True)
    
    # Inyección de ruido gaussiano (5% de la desviación estándar) en el precio
    std_precio = df_original['precio'].std()
    ruido = rng.normal(loc=0, scale=std_precio * 0.05, size=len(nuevos))
    nuevos['precio'] = (nuevos['precio'] + ruido).clip(df_original['precio'].min(), df_original['precio'].max())
    
    # Combinar sets
    df_expandido = pd.concat([df_original, nuevos], ignore_index=True)
    print(f"📈 EXPANSION: Dataset expandido exitosamente a {len(df_expandido)} registros.")
    return df_expandido

def crear_variables_derivadas(df):
    """Crea nuevas características predictoras a partir de las existentes."""
    df = df.copy()
    if 'fecha_registro' in df.columns and pd.api.types.is_datetime64_any_dtype(df['fecha_registro']):
        df['mes_registro'] = df['fecha_registro'].dt.month
        df['dia_semana_registro'] = df['fecha_registro'].dt.dayofweek
        # Variable binaria útil: ¿Es fin de semana?
        df['es_fin_semana'] = (df['fecha_registro'].dt.dayofweek >= 5).astype(int)
        df.drop(columns=['fecha_registro'], inplace=True)
    return df

def codificar_variables_categoricas(df):
    """Aplica One-Hot Encoding automático para preparar variables para ML."""
    # Excluimos variables de texto libre como 'nombre' o 'id' que arruinan la regresión
    excluir = ['id', 'nombre', 'precio']
    cols_a_codificar = [c for c in df.select_dtypes(include=['object','string']).columns if c not in excluir]
    
    df_encoded = pd.get_dummies(df, columns=cols_a_codificar, drop_first=True)
    return df_encoded

# =============================================================================
# 4. FUNCIONES DE VALIDACIÓN Y CALIDAD
# =============================================================================
def detectar_outliers(df):
    """Detecta anomalías estadísticas usando la técnica de Z-score."""
    if 'precio' not in df.columns: return
    z_scores_full= pd.Series(np.nan , index=df.index)
    precios_validos= df['precio'].dropna()

    if len(precios_validos)== 0:
        print("VALIDACION: la columna 'precios' esta vacia o contiene solo NaN")
        return None
    

    z_scores_full.loc[precios_validos.index] = np.abs(zscore(precios_validos))
    
    # Identificamos cuáles superan el umbral (los NaNs darán False automáticamente)
    es_outlier = z_scores_full > 3
    total_outliers = es_outlier.sum()
    print(f"🕵️ VALIDATION: Detectados {total_outliers} outliers potenciales en la columna 'precio' (Z-score > 3).")
    

def validar_calidad_expansion(df_orig, df_exp):
    """Calcula la desviación métrica de las medias para asegurar coherencia estadística."""
    m_orig = df_orig['precio'].mean()
    m_exp = df_exp['precio'].mean()
    diff_pct = abs(m_orig - m_exp) / m_orig * 100
    print(f"📊 CALIDAD: Variación de la media del precio: {diff_pct:.2f}% (Meta sugerida: <5%).")

# =============================================================================
# 5. FUNCIONES DE MODELADO ML
# =============================================================================
def preparar_datos_ml(df):
    """Separa features (X) de target (y), y aplica división de sets y scaling."""
    # Descartar columnas no numéricas o identificadores no predictivos
    columnas_validas = df.select_dtypes(include=[np.number,bool])
    
    X = columnas_validas.drop(columns=['precio', 'id'], errors='ignore')
    y = columnas_validas['precio']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # se llenan valores nulos 
    imputer = SimpleImputer(strategy='median')
    X_train_imp= imputer.fit_transform(X_train)
    X_test_imp= imputer.transform(X_test)
    
    # Escalado de datos para estabilidad de coeficientes
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluar_modelo(modelo, X_train, X_test, y_train, y_test):
    """Genera métricas de rendimiento y diagnósticos del modelo."""
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)
    
    metricas = {
        'R2_Train': r2_score(y_train, y_train_pred),
        'R2_Test': r2_score(y_test, y_test_pred),
        'RMSE_Test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE_Test': mean_absolute_error(y_test, y_test_pred)
    }
    return metricas, y_test_pred

# =============================================================================
# 6. FUNCIONES DE VISUALIZACIÓN
# =============================================================================
def graficar_distribucion_y_correlaciones(df):
    """Genera las visualizaciones obligatorias de análisis exploratorio."""
    os.makedirs("graficos", exist_ok=True)
    
    # 1. Distribución de variable objetivo (Precio)
    plt.figure(figsize=(6, 4))
    sns.histplot(df['precio'], kde=True, color='teal')
    plt.title('Distribución de Variable Objetivo (Precio)')
    plt.savefig('graficos/distribucion_precio.png')
    plt.close()
    
    # 2. Matriz de Correlación
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlación de Variables')
    plt.tight_layout()
    plt.savefig('graficos/matriz_correlacion.png')
    plt.close()

def graficar_resultados_modelo(y_test, y_pred):
    """Genera las visualizaciones obligatorias del rendimiento del regresor."""
    # 1. Reales vs Predichos
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.7, color='indigo')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Valores Reales vs. Predichos')
    plt.xlabel('Reales')
    plt.ylabel('Predichos')
    plt.savefig('graficos/predicciones_vs_reales.png')
    plt.close()
    
    # 2. Residuos
    plt.figure(figsize=(6, 4))
    residuos = y_test - y_pred
    plt.scatter(y_pred, residuos, alpha=0.7, color='crimson')
    plt.axhline(y=0, linestyle='--', color='black')
    plt.title('Gráfico de Residuos (Errores)')
    plt.xlabel('Predichos')
    plt.ylabel('Residuo')
    plt.savefig('graficos/residuos_modelo.png')
    plt.close()
    print("🎨 VISUALIZACIONES: Gráficos guardados exitosamente en la carpeta '/graficos'.")

# =============================================================================
# 7. PIPELINE PRINCIPAL (main)
# =============================================================================
def pipeline_inteligente():
    print("="*60)
    print("INICIANDO PIPELINE INTELIGENTE DE ANÁLISIS DE DATOS")
    print("="*60)
    
    # EXTRACT
    df_base = cargar_dataset_base("datos_consolidados_40_registros.csv")
    if not validar_estructura_datos(df_base): return
    
    # TRANSFORM & EXPAND
    df_limpio = limpiar_datos(df_base)
    df_expandido = expandir_dataset_500_registros(df_limpio)
    df_features = crear_variables_derivadas(df_expandido)
    df_final = codificar_variables_categoricas(df_features)
    
    # VALIDATE
    detectar_outliers(df_final)
    validar_calidad_expansion(df_limpio, df_expandido)
    
    # VISUALIZE (Fase 1)
    graficar_distribucion_y_correlaciones(df_expandido)
    
    # MODEL & EVALUATE
    X_train, X_test, y_train, y_test = preparar_datos_ml(df_final)
    
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    metricas, y_pred = evaluar_modelo(modelo, X_train, X_test, y_train, y_test)
    
    # VISUALIZE (Fase 2)
    graficar_resultados_modelo(y_test, y_pred)
    
    # LOAD & SAVE (Guardar entregables)
    df_expandido.to_csv("datos_500_registros.csv", index=False)
    print("\n💾 LOAD: Dataset final exportado como 'datos_500_registros.csv'")
    
    print("\n📊 --- RESULTADOS EN NEGOCIO Y MODELO ---")
    for k, v in metricas.items(): print(f"-> {k}: {v:.4f}")
    
    print("="*60)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    return modelo, metricas

# =============================================================================
# 8. EJECUCIÓN AUTOMÁTICA
# =============================================================================
if __name__ == "__main__":
    pipeline_inteligente()