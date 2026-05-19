# Documentación Técnica - Pipeline v1 (con Fuga de Datos)

## Introducción
Versión inicial del pipeline ETL-ML con fuga de datos en expansión sintética. Usa solo Regresión Lineal. Dataset base insuficiente.

## Setup
- Python ≥ 3.8
- `pip install pandas numpy matplotlib seaborn scikit-learn scipy`
- `python pipeline_inteligente_v1.py`

## Diferencias clave vs v2
- Expansión sin split previo (fuga de datos).
- Solo LinearRegression.
- Menos validaciones.
- Sin Random Forest ni cross-validation.

## Flujo
Similar a v2 pero con leakage en train_expandido.

## Advertencia
No usar en producción. v2 corrige el problema.