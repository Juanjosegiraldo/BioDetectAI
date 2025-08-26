# BioDetectAI
Proyecto que permite clasificar títulos y resúmenes de artículos médicos y clasificarlos según su categoría médica.

##Clasificación Automática de Artículos Biomédicos

Este proyecto implementa un modelo de **clasificación multilabel** de artículos científicos en 4 dominios médicos principales:  

- Cardiovascular  
- Neurological  
- Hepatorenal  
- Oncological  

El modelo se entrenó usando **TF-IDF** sobre `title + abstract` y un clasificador **One-vs-Rest con LinearSVC**.

---

### Estructura del Proyecto

- `model_svm.pkl` → modelo entrenado.  
- `tfidf_vectorizer.pkl` → vectorizador TF-IDF.  
- `model.py` → script de entrenamiento y guardado del modelo.  
- `predict.py` → script para realizar predicciones con nuevos artículos.  
- `Informe.ipynb` → análisis completo en Jupyter/Colab (EDA, limpieza, pruebas).  
- `README.md` → este archivo de documentación.  

---

### Instalación y uso

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Juanjosegiraldo/BioDetectAI.git
   cd BioDetectAI
   
2. (Opcional) Crear un entorno virtual
Esto es recomendable para no mezclar librerías con otros proyectos.

      2.1 En Windows:
   ```bash
    python -m venv venv
    venv\Scripts\activate
   ```
  
      2.2 En Linux/Mac:
  ```bash
    python3 -m venv venv
    source venv/bin/activate
  ```

3. Instalar dependencias
```bash
pip install -r requirements.txt
```
4. Entrenar el modelo
```bash
python train.py
```
Esto genera los archivos model_svm.pkl y tfidf_vectorizer.pkl.

6. Hacer una predicción
```bash
python predict.py
```
El programa pedirá un título y un abstract, luego devolverá las etiquetas predichas.

   
