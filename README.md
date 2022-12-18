# Impacto de la campaña de vacunación contra el COVID-19 en Chile 

## 1. Modelo compartimental
En la carpeta se puede encontrar la carpeta de simulación. En esta se encuentra el script en formato .py que permite resolver numéricamente la EDO del modelo compartimental, además se encuentran las tasas (parámetros) estimados en la parte de calibración para este modelo en formato .csv. Este archivo entrega todos los gráficos mostrados en el informe, además de todas las métricas del modelo presentadas, que también se guardan en formato .csv.

## 2. Modelo estadístico de desenlaces
En la carpeta se encuentra un archivo .ipynb que permite calibrar el modelo estadístico, tanto para toda la población como por edades y también con vacunación de refuerzo y sin refuerzo. Este mismo archivo simula los desenlaces con los casos dados por el modelo compartimental. Se encuentran los datos de desenlaces por rango etario y también por esquema de vacunación.
