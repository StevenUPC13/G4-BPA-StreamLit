
# Predicción de Éxito Comercial de Películas

Proyecto de Business Predictive Analytics orientado a predecir si una película puede alcanzar la categoría de blockbuster utilizando Machine Learning.

## Modelo utilizado
El modelo seleccionado fue Gradient Boosting, debido a que obtuvo el mejor desempeño general en la comparación de modelos, alcanzando un ROC-AUC de 0.866 y Accuracy de 0.827.

## Archivos principales
- app.py: aplicación Streamlit.
- modelo_blockbuster.pkl: modelo entrenado.
- features_modelo.pkl: lista de variables utilizadas.
- global_movies_dataset_1950_2026_app.csv: dataset para dashboard.
- comparacion_modelos.csv: resultados comparativos de modelos.
- feature_importance.csv: importancia de variables.
- requirements.txt: librerías necesarias.

## Ejecución local

pip install -r requirements.txt

streamlit run app.py

## Deployado

Alojado en el repositorio de github: https://github.com/StevenUPC13/G4-BPA-StreamLit.git 
