
from sklearn import datasets
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix #Porcentaje de acierto
from random import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# leemos el dataset
iris = datasets.load_iris()
#Creamos los conjuntos de entrenamiento y test.
x = iris.data #Las 4 características
t = iris.target #target 0, 1, 2
tam_test = 0.30 #Tamaño para el test en %
semilla = 7 #semilla para generar aleatoriedad
xe, xt, te, tt = model_selection.train_test_split(x, t, test_size=tam_test,
random_state=semilla)

for i in range(0,1): #Buscar alguno para mayor diferencia entre vecinos
    k = 5
    clasificador= KNeighborsClassifier(k)
    clasificador_Arbol = DecisionTreeClassifier()
    clasificador_Vector = SVC()
    clasificador_Red = MLPClassifier()
# print(clasificador)
#Predicción kNeightbors
    clasificador.fit(xe,te)
    obtenido = clasificador.predict(xt)
    clasificacion = accuracy_score(tt, obtenido)

#Prediccion Arbol
    clasificador_Arbol.fit(xe,te)
    obtenido_Arbol = clasificador_Arbol.predict(xt)
    clasificacion_Arbol = accuracy_score(tt, obtenido_Arbol)

#Prediccion Vector
    clasificador_Vector.fit(xe,te)
    obtenido_Vector = clasificador_Vector.predict(xt)
    clasificacion_Vector = accuracy_score(tt, obtenido_Vector)
    
#Prediccion Red
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


