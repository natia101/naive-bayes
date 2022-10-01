

# Importacion de librerias
import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB # cuando es texto se usa la multinomial
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#Importación de datos
url='https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv'
df = pd.read_csv(url)

# Transformación del tipo de dato
df[df.select_dtypes('object').columns]=df[df.select_dtypes('object').columns].astype('string')
df['polarity']=df['polarity'].astype('category')

#Transformaciones del texto para que sea uniforme
df['review']=df['review'].str.strip() # elimina espacios al comienzo y al final de la oracion
df['review']=df['review'].str.lower() # lleva todo a minuscula
df['review']=df['review'].str.replace('!','')
df['review']=df['review'].str.replace(',','')
df['review']=df['review'].str.replace('&','')
df['review']=df['review'].str.normalize('NFKC')
df['review']=df['review'].str.replace(r'([a-zA-Z])\1{2,}',r'\1',regex=True) # elimina caracteres repetidos mas de dos veces

# Función que estandariza las palabras a Normal Form Decomposed (NFD) para luego indicar que codificar en ascii ignorando los errores.
def normalize_str(text_string):
    if text_string is not None:
        result=unicodedata.normalize('NFD',text_string).encode('ascii','ignore').decode()
    else:
        result=None 
    return result

df['review']=df['review'].apply(normalize_str)

# Las transformaciones anteriores se deberan realizar a los datos nuevos con los que se quiera correr el modelo.
# Se definen las variables features y target
X=df['review']
y=df['polarity']

# Se separa la muestra de entrenamiento y prueba considerando como referencia para la estratificacion la variable objetivo.

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2007,stratify=y)

# Se crean matrices esparzas a partir de la variable explicativa que luego se utilizara para entrenar el modelo y testearlo.
vec=CountVectorizer(stop_words='english')
X_train=vec.fit_transform(X_train).toarray()
X_test=vec.transform(X_test).toarray()

# Se entrena un modelo de clasificacion con el algoritmo Naive Bayes Multinomial
nb=MultinomialNB()
nb.fit(X_train,y_train)
y_predict=nb.predict(X_test)

#Se guarda el modelo
import pickle
filename = '/workspace/naive-bayes/models/finalized_model.sav'
pickle.dump(nb, open(filename, 'wb'))
