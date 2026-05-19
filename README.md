# Pipeline Inteligente de Análisis de Datos

Proyecto ETL + ML para predicción de precios usando dataset consolidado.

## Estructura
- `datos_consolidados_40_registros.csv`: Dataset base (40 registros, insuficiente para ML robusto).
- `pipeline_inteligente_v1.py`: Versión inicial con fuga de datos (data leakage en expansión).
- `pipeline_inteligente_v2.py`: Pipeline final corregido (sin fuga, validación cruzada, dos modelos).
- `documentacion_tecnica_v2.md`: Documentación técnica completa.

## Resultados ML
Resultados limitados por dataset básico y pocas features. R² bajo en test. Mejorable con más datos/variables.
```bash
MÉTRICAS LR
R2_Train: 0.9569
R2_Test: -0.5036
RMSE: 66.3217
MAE: 52.9653

🌲 MÉTRICAS RF
R2_Train: 0.9569
R2_Test: -0.3382
RMSE: 62.5677
MAE: 48.4084

```


## Ejecución
```bash
python pipeline_inteligente_v2.py
```

Genera `datos_500_registros.csv` + gráficos en `/graficos`.

## Reto
Ver "Reto Final.pdf" para requisitos originales.