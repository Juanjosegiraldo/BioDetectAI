import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import joblib

# 1. Cargar dataset
df = pd.read_csv('challenge_data-18-ago.csv', sep=';', engine='python')

# 2. Preparar etiquetas multilabel
base_labels = ["cardiovascular", "neurological", "hepatorenal", "oncological"]
df["group"] = df["group"].str.split("|")
mlb = MultiLabelBinarizer(classes=base_labels)
y = mlb.fit_transform(df["group"])
y_df = pd.DataFrame(y, columns=mlb.classes_)

# 3. TF-IDF del título + abstract
df['title'] = df['title'].str.lower()
df['abstract'] = df['abstract'].str.lower()
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['title'] + " " + df['abstract'])

# 4. Entrenar Linear SVM multilabel
model_svm = OneVsRestClassifier(LinearSVC())
model_svm.fit(X, y_df)

# 5. Guardar modelo y vectorizador
joblib.dump(model_svm, "model_svm.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("✅ Modelo y vectorizador guardados!")
