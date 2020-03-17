from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.classification import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn import model_selection, preprocessing
from sklearn.decomposition import PCA

cancer = datasets.load_breast_cancer() #Carga la base de datos
#Fíjate en las 30 características de entrada
cancer.feature_names
#Separa entre entrada y salida
x=cancer.data
t=cancer.target
'''Fíjate en los valores de entrada. Hay variables que tienen valores altos y
otros pequeños. Para poder utilizar el clasificador, debemos normalizar
la entrada. Para ello utilizamos la librería de preprocesamiento.'''

x_norm=preprocessing.scale(x)
tam_test = 0.30 #Tamaño para el test en %
semilla = 7 #semilla para generar aleatoriedad

xe, xt, te, tt = model_selection.train_test_split(x, t, test_size=tam_test,
random_state=semilla)

pca_model = PCA(n_components=2)
pca_model.fit(xe)
xe = pca_model.transform(xe)
xt = pca_model.transform(xt)
# 2-Dimensions
#print(xe[:5]) #Dos dimensiones

#Prediccion Vecinos
k = 5
clasificador = KNeighborsClassifier(k)
clasificador.fit(xe,te)
obtenido = clasificador.predict(xt)
clasificacion = accuracy_score(tt, obtenido)


#Prediccion Arbol
clasificador_Arbol = DecisionTreeClassifier()
clasificador_Arbol.fit(xe,te)
obtenido_Arbol = clasificador_Arbol.predict(xt)
clasificacion_Arbol = accuracy_score(tt, obtenido_Arbol)

#Prediccion Vector
clasificador_Vector = SVC()
clasificador_Vector.fit(xe,te)
obtenido_Vector = clasificador_Vector.predict(xt)
clasificacion_Vector = accuracy_score(tt, obtenido_Vector)
    
#Prediccion Red
clasificador_Red = MLPClassifier()
clasificador_Red.fit(xe,te)
obtenido_Red = clasificador_Red.predict(xt)
clasificacion_Red = accuracy_score(tt, obtenido_Red)


#Matriz confusion de Vecinos
print("\n<----------Vecinos----------->")
print("K=", k)
print("Clasifiacion: %.4f" % clasificacion)
matrizConfusion = confusion_matrix(tt, obtenido)
print (matrizConfusion)

#Matriz confusion de Arbol
print("\n<----------Arbol----------->")
print("Clasifiacion: %.4f" % clasificacion_Arbol)
matrizConfusion_Arbol = confusion_matrix(tt, obtenido_Arbol)
print (matrizConfusion_Arbol)

#Matriz confusion de Vector
print("\n<----------Vector----------->")
print("Clasifiacion: %.4f" % clasificacion_Vector)
matrizConfusion_Vector = confusion_matrix(tt, obtenido_Vector)
print (matrizConfusion_Vector)

#Matriz confusion de Red
print("\n<----------Red----------->")
print("Clasifiacion: %.4f" % clasificacion_Red)
matrizConfusion_Red = confusion_matrix(tt, obtenido_Red)
print (matrizConfusion_Red)


import pandas as pd
datos = pd.DataFrame(clasificador_Arbol.loss_curve_).plot()
print(datos)

