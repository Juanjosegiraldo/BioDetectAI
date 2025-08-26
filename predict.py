import joblib
import pandas as pd

# Cargar modelo y vectorizador
model = joblib.load("model_svm.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
columns = ["cardiovascular", "neurological", "hepatorenal", "oncological"]

def predict_article(title, abstract):
    X_new = vectorizer.transform([title.lower() + " " + abstract.lower()])
    y_pred = model.predict(X_new)
    #output = pd.DataFrame(y_pred, columns=columns)
    predicted_labels = [columns[i] for i, val in enumerate(y_pred[0]) if val == 1]
    return predicted_labels

if __name__ == "__main__":
    title = input("Inserte el título: ")
    abstract = input("Inserte el resumen: ")
    result = predict_article(title, abstract)
    print("\nPredicción de dominios:")
    print(result)
