# Tratamiento de datos
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd

# Gráficos
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Preprocesado y modelado
# ------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

#Obtenemos los datos contenidos en nuestro archivo
df = pd.read_excel("/content/estaciones-de-transmilenio.xlsx")

df.head(5)

name = df["geopoint"].str.split(',', expand=True)
name.columns = ['geopoint_x', 'geopoint_y']
df = pd.concat([df, name], axis=1)
df = df.drop('geopoint', axis=1)
df

#veamos si las variables se cargaron corectamente
df.info()

df['geopoint_x'] = df['geopoint_x'].astype(float)
df['geopoint_y'] = df['geopoint_y'].astype(float)
df.info()

#separamos las columnas predictoras de la columna que tiene la variable a predecir
#variables predictoras
x = df.iloc[ : ,3:7]
#variable a predecir
y = df.iloc[ : ,2]

x.head()

#dividimos nuestros datos en testing y training
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle = True, random_state = 123)

#creamos el modelo de arbol de decisión
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(random_state = 123)

#entrenamos el arbol
arbol = classifierDT.fit(X_train, y_train)

#graficamos el arbol de decisión
fig = plt.figure(figsize=(20,20))

plot_tree(arbol,feature_names=list(x.columns.values), class_names=list(y.values), filled=True)
plt.show()

fig.savefig("arbol.pdf")

#llevamos a cabo la prediccion con los datos optenidos en la tabla testing
y_pred = arbol.predict(X_test)

#calculando la precision del modelo
#Creamos la matrix de confucion
from sklearn.metrics import confusion_matrix
matriz_de_confusion = confusion_matrix(y_test, y_pred)
matriz_de_confusion

#Calculamos la precision global del modelo
precision_global = np.sum(matriz_de_confusion.diagonal())/np.sum(matriz_de_confusion)
precision_global