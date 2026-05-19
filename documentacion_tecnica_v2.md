# Índice
1. [Introducción](#introducción)  
2. [Arquitectura del Sistema](#arquitectura-del-sistema)  
3. [Componentes y Servicios (Tabla Maestra)](#componentes-y‑servicios‑tabla‑maestra)  
4. [Matriz de Errores](#matriz-de-errores)  
5. [Flujo de Trabajo](#flujo-de-trabajo)  
6. [Detalles de Implementación](#detalles-de-implementación)  
7. [Métricas y Evaluación](#métricas-y‑evaluación)  
8. [Visualizaciones](#visualizaciones)  
9. [Exportación y Persistencia](#exportación-y‑persistencia)  
10. [Ejecutar la Pipeline](#ejecutar-la-pipeline)  
11. [Diagrama de Flujo (Mermaid)](#diagrama-de-flujo‑mermaid)  

---  

## Introducción
Este documento consolida la información técnica del script **pipeline_inteligente.py**, que implementa un proceso ETL‑ML completo: carga, validación, transformación, expansión sintética, codificación, entrenamiento de modelos (Regresión Lineal y Random Forest), evaluación, validación cruzada y generación de visualizaciones.

## Setup y Requisitos
- Python ≥ 3.8
- `pip install pandas numpy matplotlib seaborn scikit-learn scipy`
- Ejecutar: `python pipeline_inteligente_v2.py`  

---  

## Arquitectura del Sistema
El proyecto sigue una arquitectura modular basada en funciones puras que se agrupan en cuatro capas lógicas:

| Capa | Responsabilidad | Funciones principales |
|------|------------------|-----------------------|
| **Extracción** | Lectura y verificación del dataset origen. | `cargar_dataset`, `validar_estructura` |
| **Transformación** | Limpieza, generación de variables, expansión y codificación. | `limpiar_datos`, `crear_variables`, `expandir_dataset`, `codificar` |
| **Modelado** | División de datos, pre‑procesamiento, entrenamiento y evaluación. | `separar_xy`, `preparar_datos`, `evaluar`, `cross_validation` |
| **Visualización & Persistencia** | Creación de gráficos y guardado del dataset final. | `generar_graficos`, exportación CSV |

---  

## Componentes y Servicios (Tabla Maestra)

| Componente / Librería | Versión mínima recomendada | Servicio / Uso en el script |
|-----------------------|----------------------------|-----------------------------|
| `os` | — | Gestión de directorios para guardar gráficos. |
| `numpy` (`np`) | 1.20 | Operaciones numéricas, generación de ruido y cálculo de métricas. |
| `pandas` (`pd`) | 1.2 | Manipulación de DataFrames, I/O CSV, codificación one‑hot. |
| `matplotlib.pyplot` (`plt`) | 3.3 | Creación de figuras estáticas. |
| `seaborn` (`sns`) | 0.11 | Visualizaciones avanzadas (histogramas, heatmaps). |
| `scipy.stats.zscore` | 1.5 | Detección de outliers mediante Z‑score. |
| `sklearn.utils.resample` | 0.24 | Expansión sintética del dataset (bootstrap). |
| `sklearn.impute.SimpleImputer` | 0.24 | Imputación de valores faltantes (mediana). |
| `sklearn.preprocessing.StandardScaler` | 0.24 | Normalización de variables numéricas. |
| `sklearn.model_selection.train_test_split` | 0.24 | División entrenamiento / prueba. |
| `sklearn.model_selection.cross_val_score` | 0.24 | Validación cruzada (5‑fold, métrica R²). |
| `sklearn.linear_model.LinearRegression` | 0.24 | Modelo de regresión lineal. |
| `sklearn.ensemble.RandomForestRegressor` | 0.24 | Modelo de bosque aleatorio. |
| `sklearn.metrics.r2_score` | 0.24 | Cálculo de R². |
| `sklearn.metrics.mean_absolute_error` | 0.24 | Cálculo de MAE. |
| `sklearn.metrics.mean_squared_error` | 0.24 | Cálculo de MSE (para RMSE). |

---  

## Matriz de Errores

| Código | Origen | Descripción | Acción recomendada |
|--------|--------|-------------|--------------------|
| **E001** | `cargar_dataset` | Excepción al leer el CSV (ruta inexistente, permisos, formato). | Verificar ruta, permisos de lectura y formato del archivo. |
| **E002** | `validar_estructura` | Falta de columnas obligatorias. | Añadir columnas faltantes o actualizar la lista de columnas esperadas. |
| **E003** | `limpiar_datos` | Conversión de `precio` a numérico falla (valores no convertibles). | Revisar datos fuente, aplicar limpieza previa o usar `errors='coerce'` y tratar NaN. |
| **E004** | `limpiar_datos` | Conversión de `fecha_registro` a datetime falla. | Asegurar formato ISO o especificar `format=` en `pd.to_datetime`. |
| **E005** | `expandir_dataset` | `std` = 0 para alguna columna numérica (no hay variación). | Omitir ruido para esa columna; el código ya lo hace. |
| **E006** | `codificar` | Desalineación de columnas entre train y test después del one‑hot. | La función `align` ya rellena con 0; revisar que no haya columnas perdidas. |
| **E007** | `preparar_datos` | Imputación o escalado falla por tipos incompatibles. | Garantizar que `X_train`/`X_test` sean arrays numéricos. |
| **E008** | `cross_validation` | Insuficientes muestras para 5‑fold. | Aumentar tamaño del dataset o reducir `cv`. |
| **E009** | `generar_graficos` | Error al guardar imágenes (directorio sin permisos). | Verificar permisos de escritura en la carpeta `graficos`. |

---  

## Flujo de Trabajo
1. **Carga** del dataset desde CSV.  
2. **Validación** de la estructura (columnas esperadas).  
3. **Limpieza** de tipos (`precio`, `fecha_registro`).  
4. **Ingeniería de características**: extracción de mes, día de semana y flag de fin de semana.  
5. **División** en conjuntos de entrenamiento y prueba (80/20).  
6. **Expansión** del conjunto de entrenamiento mediante *bootstrap* y adición de ruido controlado.  
7. **Detección de outliers** y **validación de la expansión** (variación de la media).  
8. **Codificación** one‑hot de variables categóricas (excluyendo `id` y `nombre`).  
9. **Separación** de X e y.  
10. **Pre‑procesamiento**: imputación de medianas + escalado estándar.  
11. **Entrenamiento** de dos modelos (Regresión Lineal y Random Forest).  
12. **Evaluación** de métricas (R², RMSE, MAE) y **validación cruzada** para la regresión lineal.  
13. **Generación de gráficos** (distribución, correlaciones, predicciones vs reales, residuos).  
14. **Exportación** del dataset expandido a CSV.  

---  

## Detalles de Implementación

### 1. Extracción
```python
def cargar_dataset(ruta="datos_consolidados_40_registros.csv"):
    try:
        df = pd.read_csv(ruta)
        print(f"✅ Dataset cargado ({len(df)} registros)")
        return df
    except Exception as e:
        print(f"❌ Error: {e}")   # E001
        return None
```

### 2. Validación de Estructura
```python
def validar_estructura(df):
    columnas = ["id", "nombre", "categoria", "precio", "origen", "fecha_registro"]
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        print(f"⚠️ Faltan columnas: {faltantes}")   # E002
        return False
    return True
```

### 3. Transformaciones clave
- **Limpieza de precios**: reemplazo de comas y conversión a `float`.  
- **Conversión de fechas**: `pd.to_datetime`.  
- **Variables temporales**: `mes_registro`, `dia_semana`, `fin_semana`.  

### 4. Expansión Sintética
Utiliza `sklearn.utils.resample` para generar filas adicionales y añade ruido gaussiano (2 % de la desviación estándar) a columnas numéricas, preservando `precio` e `id`.

### 5. Codificación One‑Hot
Excluye identificadores (`id`, `nombre`) y aplica `pd.get_dummies(..., drop_first=True)`. Luego alinea columnas entre train y test para evitar desajustes.

### 6. Preparación de datos
```python
imputer = SimpleImputer(strategy="median")
scaler   = StandardScaler()
```
Aplica primero imputación y después escalado a ambos conjuntos.

### 7. Modelado
- **Regresión Lineal** (`LinearRegression`)  
- **Random Forest** (`RandomForestRegressor`, 100 árboles, `random_state=42`)  

Ambos se entrenan con `X_train`, `y_train`.  

### 8. Evaluación
```python
metricas = {
    "R2_Train": r2_score(y_train, y_train_pred),
    "R2_Test":  r2_score(y_test,  y_test_pred),
    "RMSE":    np.sqrt(mean_squared_error(y_test, y_test_pred)),
    "MAE":     mean_absolute_error(y_test, y_test_pred)
}
```
Se muestra también la validación cruzada (5‑fold) para la regresión lineal.

### 9. Visualizaciones
Se guardan cuatro imágenes en la carpeta `graficos`:
- Distribución del precio.  
- Matriz de correlaciones numéricas.  
- Comparación *Reales vs Predichos*.  
- Residuales vs predicciones.  

---  

## Métricas y Evaluación
| Métrica | Regresión Lineal | Random Forest |
|---------|------------------|---------------|
| **R2 (Train)** | `R2_Train` | — |
| **R2 (Test)**  | `R2_Test`  | `R2_Test` |
| **RMSE**       | `RMSE`     | `RMSE` |
| **MAE**        | `MAE`      | `MAE` |
| **CV Mean (R²)** | `cross_validation` (solo LR) | — |

Los valores se imprimen en consola con cuatro decimales.

---  

## Visualizaciones
Los archivos generados son:
- `graficos/distribucion.png`
- `graficos/correlaciones.png`
- `graficos/predicciones.png`
- `graficos/residuos.png`

Cada figura está cerrada (`plt.close()`) para liberar memoria.

---  

## Exportación y Persistencia
```python
train_expandido.to_csv("datos_500_registros.csv", index=False)
```
El dataset expandido (≈ 500 registros) se guarda para usos posteriores.

---  

## Ejecutar la Pipeline
```bash
python pipeline_inteligente.py
```
Al ejecutar el módulo como script (`if __name__ == "__main__":`), se lanza `pipeline_inteligente()` que devuelve el modelo de regresión lineal entrenado y sus métricas.

---  

## Diagrama de Flujo (Mermaid)

```mermaid
flowchart TD
    A[Inicio] --> B{Cargar CSV}
    B -->|Éxito| C[Validar estructura]
    B -->|Error (E001)| Z[Terminar]
    C -->|OK| D[Limpiar datos]
    D --> E[Crear variables temporales]
    E --> F[Train/Test split (80/20)]
    F --> G[Expandir train (bootstrap)]
    G --> H[Detectar outliers]
    G --> I[Validar expansión (media)]
    H --> J[Codificar (one‑hot)]
    I --> J
    J --> K[Separar X e y]
    K --> L[Imputar + Escalar]
    L --> M{Entrenar modelos}
    M --> N[Linear Regression]
    M --> O[Random Forest]
    N --> P[Evaluar LR]
    O --> Q[Evaluar RF]
    P --> R[Cross‑validation LR]
    Q --> S[Generar gráficos]
    R --> S
    S --> T[Exportar dataset expandido]
    T --> U[Fin]
    Z --> U
```

---  

*Fin del documento.*