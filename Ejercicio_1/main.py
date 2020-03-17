
from sklearn import datasets
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix #Porcentaje de acierto
from random import randint
from sklearn.tree import DecisionTreeClassifier


# leemos el dataset
iris = datasets.load_iris()
#Creamos los conjuntos de entrenamiento y test.
x = iris.data #Las 4 características
t = iris.target #target 0, 1, 2
tam_test = 0.30 #Tamaño para el test en %
semilla = 7 #semilla para generar aleatoriedad
xe, xt, te, tt = model_selection.train_test_split(x, t, test_size=tam_test,
random_state=semilla)


k = 0
for i in range(0,5): #Buscar alguno para mayor diferencia entre vecinos
    k += 3
    clasificador= KNeighborsClassifier(k)
   # print(clasificador)
#Predicción
    clasificador.fit(xe,te)
    obtenido = clasificador.predict(xt)
    clasificacion = accuracy_score(tt, obtenido)

#clefor i in range(5):
    print("")
    print("K=", k)
    print("Clasifiacion: %.4f" % clasificacion)
   # print ("4 Decimales %.4f" % clasificacion)
    matrizConfusion = confusion_matrix(tt, obtenido)
    print (matrizConfusion)