#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PREPARACIÓN DE LOS DATOS


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, cheby1, freqz


# In[5]:


dataC = pd.read_csv("C:/Users/andre/OneDrive/Desktop/MIA/Maestria/TESIS FINAL/Dataset/Cerrado.csv")
dataA = pd.read_csv("C:/Users/andre/OneDrive/Desktop/MIA/Maestria/TESIS FINAL/Dataset/Abierta.csv")
dataAC = pd.read_csv("C:/Users/andre/OneDrive/Desktop/MIA/Maestria/TESIS FINAL/Dataset/AC.csv")
dataCA = pd.read_csv("C:/Users/andre/OneDrive/Desktop/MIA/Maestria/TESIS FINAL/Dataset/CA.csv")


# In[6]:


# Graficar los datos de la columna RPM, Tiempo y Voltaje
# U= Entrada (V) X= Salida (rpm)

plt.plot(dataA['Channel'],'r')

plt.ylabel('Time')
plt.xlabel('Señal')
#plt.ylim(0, max(DF['Voltaje']) +3)
plt.title('Señal EMG con ruido')

# Mostrar el gráfico
plt.show()


# In[7]:


plt.plot(dataC['Channel'],'b')

plt.ylabel('Time')
plt.xlabel('Señal')
#plt.ylim(0, max(DF['Voltaje']) +3)
plt.title('Señal EMG con ruido')

# Mostrar el gráfico
plt.show()


# In[8]:


plt.plot(dataAC['Channel'],'y')

plt.ylabel('Time')
plt.xlabel('Señal')
#plt.ylim(0, max(DF['Voltaje']) +3)
plt.title('Señal EMG con ruido')

# Mostrar el gráfico
plt.show()


# In[9]:


plt.plot(dataC['Channel'],'r')
plt.plot(dataA['Channel'],'b')
plt.plot(dataAC['Channel'],'g')
plt.plot(dataCA['Channel'],'y')

plt.ylabel('Time')
plt.xlabel('Señal')
#plt.ylim(0, max(DF['Voltaje']) +3)
plt.title('Señal EMG con ruido')

# Mostrar el gráfico
plt.show()


# In[10]:


# ETAPA DE FILTRADO


# In[11]:


## Aplicar filtros a clase Cerrada
tiempo = dataC['Time']  # Datos de tiempo (ejemplo: de 0 a 1 segundos, 1000 puntos)
clase = dataC['Class']  # Datos de clase (ejemplo: etiquetas de clase aleatorias)
senal_emg = dataC['Channel']  # Señal EMG simulada (ejemplo)

# Parámetros del filtro
frecuencia_de_corte = 30
orden_del_filtro = 2  # Orden del filtro Butterworth

# Normalización de la frecuencia de corte
frecuencia_nyquist = 0.5 * 2000  # Frecuencia de Nyquist
frecuencia_corte_normalizada = frecuencia_de_corte / frecuencia_nyquist

# Crear el filtro pasabajas Butterworth
b, a = butter(orden_del_filtro, frecuencia_corte_normalizada, btype='low')

# Aplicar el filtro a la señal EMG
senal_filtrada = lfilter(b, a, senal_emg)


# In[12]:


plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(tiempo, senal_emg, 'b', label='Señal EMG Original')
plt.title('Señal EMG Original')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempo, senal_filtrada, 'r', label='Señal EMG Filtrada')
plt.title('Señal EMG Filtrada con Butterworth')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()


# In[13]:


# Parámetros del filtro Chebyshev
ripple = 0.5  # Parámetro de la ondulación, ajusta según tus requisitos
frecuencia_de_corte_cheby = 40  # Frecuencia de corte del filtro Chebyshev (ajusta según sea necesario)
orden_del_filtro_cheby = 2  # Orden del filtro Chebyshev

# Normalización de la frecuencia de corte
frecuencia_corte_normalizada_cheby = frecuencia_de_corte_cheby / frecuencia_nyquist

# Crear el filtro Chebyshev
b_cheby, a_cheby = cheby1(orden_del_filtro_cheby, ripple, frecuencia_corte_normalizada_cheby, btype='low')

# Aplicar el filtro Chebyshev a la señal filtrada previamente
senal_filtrada_cheby = lfilter(b_cheby, a_cheby, senal_filtrada)


# In[14]:


plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(tiempo, senal_emg, 'b', label='Señal EMG Original')
plt.title('Señal EMG Original')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempo, senal_filtrada, 'r', label='Señal EMG Filtrada Butterworth')
plt.title('Señal EMG Filtrada con Butterworth')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempo, senal_filtrada_cheby, 'g', label='Señal Filtrada Chebyshev')
plt.title('Señal Filtrada Clase Abierta')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()
plt.show()

plt.tight_layout()
plt.show()


# In[15]:


# Obtener la respuesta en frecuencia del filtro Chebyshev
frecuencia, respuesta_frecuencia = freqz(b_cheby, a_cheby)

# Calcular la magnitud de la respuesta en frecuencia (módulo de la respuesta)
magnitud_respuesta = np.abs(respuesta_frecuencia)

# Calcular la frecuencia en Hz
frecuencia_hz = frecuencia * (frecuencia_nyquist / np.pi)


# In[16]:


plt.figure(figsize=(8, 6))
plt.plot(frecuencia_hz, magnitud_respuesta, 'b')
plt.title('Respuesta en Frecuencia del Filtro Chebyshev')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud de la Respuesta')
plt.grid()
plt.show()


# In[17]:


## Aplicar filtros a clase Abierta
tiempoA = dataA['Time']  # Datos de tiempo (ejemplo: de 0 a 1 segundos, 1000 puntos)
claseA = dataA['Class']  # Datos de clase (ejemplo: etiquetas de clase aleatorias)
senal_emgA = dataA['Channel']  # Señal EMG simulada (ejemplo)

# Parámetros del filtro
frecuencia_de_corteA = 30  # Frecuencia de corte del filtro (ajustar según tus necesidades)
orden_del_filtroA = 2  # Orden del filtro Butterworth

# Normalización de la frecuencia de corte
frecuencia_nyquistA = 0.5 * 2000  # Frecuencia de Nyquist (mitad de la frecuencia de muestreo)
frecuencia_corte_normalizadaA = frecuencia_de_corteA / frecuencia_nyquistA

# Crear el filtro pasabajas Butterworth
c, d = butter(orden_del_filtroA, frecuencia_corte_normalizadaA, btype='low')

# Aplicar el filtro a la señal EMG
senal_filtradaA = lfilter(c, d, senal_emgA)


# In[18]:


# Visualización de la señal original y la señal filtrada
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(tiempoA, senal_emgA, 'b', label='Señal EMG Original')
plt.title('Señal EMG Original')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempoA, senal_filtradaA, 'r', label='Señal EMG Filtrada')
plt.title('Señal EMG Filtrada con Butterworth')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()


# In[19]:


# Parámetros del filtro Chebyshev
ripple = 0.5  # Parámetro de la ondulación, ajusta según tus requisitos
frecuencia_de_corte_chebyA = 30  # Frecuencia de corte del filtro Chebyshev (ajusta según sea necesario)
orden_del_filtro_chebyA = 2  # Orden del filtro Chebyshev

# Normalización de la frecuencia de corte
frecuencia_corte_normalizada_chebyA = frecuencia_de_corte_chebyA / frecuencia_nyquistA

# Crear el filtro Chebyshev
c_cheby, d_cheby = cheby1(orden_del_filtro_chebyA, ripple, frecuencia_corte_normalizada_chebyA, btype='low')

# Aplicar el filtro Chebyshev a la señal filtrada previamente
senal_filtrada_chebyA = lfilter(c_cheby, d_cheby, senal_filtradaA)


# In[20]:


# Visualización de la señal original y la señal filtrada
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(tiempoA, senal_emgA, 'b', label='Señal EMG Original')
plt.title('Señal EMG Original')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempoA, senal_filtradaA, 'r', label='Señal EMG Filtrada Butterworth')
plt.title('Señal EMG Filtrada con Butterworth')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempoA, senal_filtrada_chebyA, 'g', label='Señal Filtrada Chebyshev')
plt.title('Señal Filtrada Clase Cerrada')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.legend()
plt.show()


# In[21]:


# Obtener la respuesta en frecuencia del filtro Chebyshev
frecuenciaA, respuesta_frecuenciaA = freqz(c_cheby, d_cheby)

# Calcular la magnitud de la respuesta en frecuencia (módulo de la respuesta)
magnitud_respuestaA = np.abs(respuesta_frecuenciaA)

# Calcular la frecuencia en Hz
frecuencia_hzA = frecuenciaA * (frecuencia_nyquistA / np.pi)


# In[22]:


plt.figure(figsize=(8, 6))
plt.plot(frecuencia_hzA, magnitud_respuestaA, 'b')
plt.title('Respuesta en Frecuencia del Filtro Chebyshev')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud de la Respuesta')
plt.grid()
plt.show()


# In[23]:


## Aplicar filtros a intension cerrar
tiempoAC = dataAC['Time']  # Datos de tiempo (ejemplo: de 0 a 1 segundos, 1000 puntos)
claseAC = dataAC['Class']  # Datos de clase (ejemplo: etiquetas de clase aleatorias)
senal_emgAC = dataAC['Channel']  # Señal EMG simulada (ejemplo)

# Parámetros del filtro
frecuencia_de_corteAC = 30  # Frecuencia de corte del filtro (ajustar según tus necesidades)
orden_del_filtroAC = 2  # Orden del filtro Butterworth

# Normalización de la frecuencia de corte
frecuencia_nyquistAC = 0.5 * 2000  # Frecuencia de Nyquist (mitad de la frecuencia de muestreo)
frecuencia_corte_normalizadaAC = frecuencia_de_corteAC / frecuencia_nyquistAC

# Crear el filtro pasabajas Butterworth
f, e = butter(orden_del_filtroAC, frecuencia_corte_normalizadaAC, btype='low')

# Aplicar el filtro a la señal EMG
senal_filtradaAC = lfilter(f, e, senal_emgAC)


# In[24]:


# Visualización de la señal original y la señal filtrada
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(tiempoAC, senal_emgAC, 'b', label='Señal EMG Original')
plt.title('Señal EMG Original')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempoAC, senal_filtradaAC, 'r', label='Señal EMG Filtrada')
plt.title('Señal EMG Filtrada con Butterworth')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()


# In[25]:


# Parámetros del filtro Chebyshev
ripple = 0.5  # Parámetro de la ondulación, ajusta según tus requisitos
frecuencia_de_corte_chebyAC = 30  # Frecuencia de corte del filtro Chebyshev (ajusta según sea necesario)
orden_del_filtro_chebyAC = 2  # Orden del filtro Chebyshev

# Normalización de la frecuencia de corte
frecuencia_corte_normalizada_chebyAC = frecuencia_de_corte_chebyAC / frecuencia_nyquistAC

# Crear el filtro Chebyshev
f_cheby, e_cheby = cheby1(orden_del_filtro_chebyAC, ripple, frecuencia_corte_normalizada_chebyAC, btype='low')

# Aplicar el filtro Chebyshev a la señal filtrada previamente
senal_filtrada_chebyAC = lfilter(f_cheby, e_cheby, senal_filtradaAC)


# In[26]:


# Visualización de la señal original y la señal filtrada
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(tiempoAC, senal_emgAC, 'b', label='Señal EMG Original')
plt.title('Señal EMG Original')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempoAC, senal_filtradaAC, 'r', label='Señal EMG Filtrada Butterworth')
plt.title('Señal EMG Filtrada con Butterworth')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempoAC, senal_filtrada_chebyAC, 'g', label='Señal Filtrada Chebyshev')
plt.title('Señal Filtrada Clase AC (Transición de abierto a cerrado)')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.legend()
plt.show()


# In[27]:


## Aplicar filtros a intension abrir
tiempoCA = dataCA['Time']  # Datos de tiempo (ejemplo: de 0 a 1 segundos, 1000 puntos)
claseCA = dataCA['Class']  # Datos de clase (ejemplo: etiquetas de clase aleatorias)
senal_emgCA = dataCA['Channel']  # Señal EMG simulada (ejemplo)

# Parámetros del filtro
frecuencia_de_corteCA = 30  # Frecuencia de corte del filtro (ajustar según tus necesidades)
orden_del_filtroCA = 2  # Orden del filtro Butterworth

# Normalización de la frecuencia de corte
frecuencia_nyquistCA = 0.5 * 2000  # Frecuencia de Nyquist (mitad de la frecuencia de muestreo)
frecuencia_corte_normalizadaCA = frecuencia_de_corteCA / frecuencia_nyquistCA

# Crear el filtro pasabajas Butterworth
h, g = butter(orden_del_filtroCA, frecuencia_corte_normalizadaCA, btype='low')

# Aplicar el filtro a la señal EMG
senal_filtradaCA = lfilter(h, g, senal_emgCA)


# In[28]:


# Visualización de la señal original y la señal filtrada
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(tiempoCA, senal_emgCA, 'b', label='Señal EMG Original')
plt.title('Señal EMG Original')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempoCA, senal_filtradaCA, 'r', label='Señal EMG Filtrada')
plt.title('Señal EMG Filtrada con Butterworth')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()


# In[29]:


# Parámetros del filtro Chebyshev
ripple = 0.5  # Parámetro de la ondulación, ajusta según tus requisitos
frecuencia_de_corte_chebyCA = 40  # Frecuencia de corte del filtro Chebyshev (ajusta según sea necesario)
orden_del_filtro_chebyCA = 4  # Orden del filtro Chebyshev

# Normalización de la frecuencia de corte
frecuencia_corte_normalizada_chebyCA = frecuencia_de_corte_chebyCA / frecuencia_nyquistCA

# Crear el filtro Chebyshev
h_cheby, g_cheby = cheby1(orden_del_filtro_chebyCA, ripple, frecuencia_corte_normalizada_chebyCA, btype='low')

# Aplicar el filtro Chebyshev a la señal filtrada previamente
senal_filtrada_chebyCA = lfilter(h_cheby, g_cheby, senal_filtradaCA)


# In[30]:


# Visualización de la señal original y la señal filtrada
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(tiempoCA, senal_emgCA, 'b', label='Señal EMG Original')
plt.title('Señal EMG Original')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempoCA, senal_filtradaCA, 'r', label='Señal EMG Filtrada Butterworth')
plt.title('Señal EMG Filtrada con Butterworth')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 1)
plt.plot(tiempoCA, senal_filtrada_chebyCA, 'g', label='Señal Filtrada Chebyshev')
plt.title('Señal Filtrada Clase CA (Transición de cerrado a abierto)')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.legend()
plt.show()


# In[31]:


#SenalA = pd.DataFrame({'SenalA': senal_filtrada_chebyA})
#SenalC = pd.DataFrame({'SenalC': senal_filtrada_cheby})
SenalA = pd.DataFrame(senal_filtrada_chebyA)
SenalC = pd.DataFrame(senal_filtrada_cheby)
SenalAC = pd.DataFrame(senal_filtrada_chebyAC)
SenalCA = pd.DataFrame(senal_filtrada_chebyCA)
Senal = pd.concat([SenalA, SenalC,SenalAC, SenalCA],axis=0)
Senal.columns = ['Filtro']
Senal


# In[32]:


### MODELOS DE ENTRENAMIENTO


# In[33]:


from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns
import seaborn as sb

import time
import multiprocessing
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
s = StandardScaler()


# In[34]:


# Se carga el nuevo conjunto de datos con las señales filtradas

df = pd.concat([dataA, dataC,dataAC,dataCA],axis=0)
dfT = pd.concat([df, Senal],axis=1)
dfT


# In[35]:


# División de los datos en train y test

x = dfT.loc[:,['Time','Channel', 'Filtro']]
y = dfT['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=30)


# In[36]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[37]:


X_train_s = s.fit_transform(X_train)
X_test_s = s.transform(X_test)
X_train_s.shape, X_test_s.shape


# In[38]:


#COMPARATIVA DE LOS TRES MODELOS


# In[39]:


# Función para realizar la búsqueda de hiperparámetros
def hyperparameter_search(clf, params, X_train_s, y_train):
    grid_search = GridSearchCV(clf, params, scoring='accuracy', cv=5)
    grid_search.fit(X_train_s, y_train)
    return grid_search.best_params_, grid_search.best_score_

# Iniciar el temporizador
start_time = time.perf_counter()

# Definir los clasificadores y parámetros
classifiers_and_params = [
    (KNeighborsClassifier(), {'n_neighbors': [1, 2, 3, 5, 7],'metric':['minkowski', 'manhattan', 'euclidean'], 'p': [1, 2, 3]}),
    (SVC(), {'kernel': ['rbf', 'poly', 'linear'], 'gamma': ['auto', 'scale'], 'C': [1, 2]}),
    (RandomForestClassifier(), {'max_depth': [2, 4, 8, 16, 64, 128], 'criterion': ['gini', 'entropy']})
]

# Almacenar los resultados en una lista
results = []
classifiers_list = []

# Realizar la búsqueda de hiperparámetros y predicciones en paralelo
for i, (clf, params) in enumerate(classifiers_and_params):
    best_params, best_score = hyperparameter_search(clf, params, X_train_s, y_train)
    pred = clf.fit(X_train_s, y_train).predict(X_test_s)
    results.append((best_params, best_score, pred))
    classifiers_list.append(clf)
    
# Detener el temporizador y calcular el tiempo de ejecución total
elapsed_time = time.perf_counter() - start_time
        
# Imprimir resultados y tiempo de ejecución
for i, (best_params, best_score, pred) in enumerate(results):
    print(f"Clasificador {i} - Mejores hiperparámetros: {best_params}")
    print(f"Clasificador {i} - Mejor puntuación: {best_score}")
    #print(confusion_matrix(y_test, pred))

print("Tiempo de ejecución total:", elapsed_time, "segundos")


# In[40]:


classifiers_list


# In[47]:


from sklearn.model_selection import cross_val_score

# Define una lista para almacenar las puntuaciones de validación cruzada de cada modelo
cross_val_scores = []

# Realiza la validación cruzada para cada modelo
for clf in classifiers_list:
    # Crea un pipeline con el clasificador y el escalador
    clf_pipeline = make_pipeline(StandardScaler(), clf)
    
    # Realiza la validación cruzada y obtiene las puntuaciones
    scores = cross_val_score(clf_pipeline, X_train_s, y_train, cv=6, scoring='accuracy')
    
    # Calcula la media de las puntuaciones y la alamacena
    cross_val_mean = np.mean(scores)
    cross_val_scores.append(cross_val_mean)

    # Imprime la puntuación de validación cruzada para este modelo
    print(f'{clf.__class__.__name__} Cross-Validation Score: {cross_val_mean:.4f}')

# Imprime las puntuaciones de validación cruzada para todos los modelos
print('Cross-Validation Scores:', cross_val_scores)


# In[48]:


import warnings
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

warnings.filterwarnings('always') 

clf_performance = {'accuracy': [], 'f1score': [], 'recall': [], 'precision': [], 'training_time': [], 'testing_time': [], 'conf_matrix': []}

for clfs in classifiers_list:
    name = clfs.__class__.__name__
    accuracy = []
    f1score = []
    recall = []
    precision = []
    conf_matrix_list = []
    training_time = []
    testing_time = []

    training_start = time.time()
    clf = make_pipeline(StandardScaler(), clfs)
    clf.fit(X_train_s, y_train)
    training_end = time.time()

    testing_start = time.time()
    y_pred = clf.predict(X_test_s)
    testing_end = time.time()
    
    acc = accuracy_score(y_test, y_pred)
    recall_res = recall_score(y_test, y_pred, average='macro')
    precision_res = precision_score(y_test, y_pred, average='macro')
    f1score_res = f1_score(y_test, y_pred, average='macro')
    
    f,ax = plt.subplots(figsize=(6, 6))
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.set(font_scale=1.5)
    sns.heatmap(conf_matrix, annot=True, linewidths=0.01,cmap="Purples",linecolor="gray",ax=ax)
    plt.xlabel("Predicción")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix Validation {name} ")
    plt.show()
    
    
    accuracy.append(acc)
    recall.append(recall_res)
    precision.append(precision_res)
    f1score.append(f1score_res)
    conf_matrix_list.append(conf_matrix)  # Cambiado el nombre de la lista
    training_time.append(training_end - training_start)
    testing_time.append(testing_end - testing_start)
    print(name, np.mean(accuracy))

    clf_performance['accuracy'].append(accuracy)
    clf_performance['f1score'].append(f1score)
    clf_performance['recall'].append(recall)
    clf_performance['precision'].append(precision)
    clf_performance['conf_matrix'].append(conf_matrix_list) 
    clf_performance['training_time'].append(np.mean(training_time))
    clf_performance['testing_time'].append(np.mean(testing_time))


# In[277]:


import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

warnings.filterwarnings('always') 

clf_performance = {'accuracy': [], 'f1score': [], 'recall': [], 'precision': [], 'training_time': [], 'testing_time': [], 'conf_matrix': []}

# Crear una sola figura para todas las matrices de confusión
fig, axes = plt.subplots(nrows=len(classifiers_list), figsize=(8, 6*len(classifiers_list)))

for i, clfs in enumerate(classifiers_list):
    name = clfs.__class__.__name__
    accuracy = []
    f1score = []
    recall = []
    precision = []
    conf_matrix_list = []
    training_time = []
    testing_time = []

    training_start = time.time()
    clf = make_pipeline(StandardScaler(), clfs)
    clf.fit(X_train_s, y_train)
    training_end = time.time()

    testing_start = time.time()
    y_pred = clf.predict(X_test_s)
    testing_end = time.time()
    
    acc = accuracy_score(y_test, y_pred)
    recall_res = recall_score(y_test, y_pred, average='macro')
    precision_res = precision_score(y_test, y_pred, average='macro')
    f1score_res = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    accuracy.append(acc)
    recall.append(recall_res)
    precision.append(precision_res)
    f1score.append(f1score_res)
    conf_matrix_list.append(conf_matrix)
    training_time.append(training_end - training_start)
    testing_time.append(testing_end - testing_start)
    print(name, np.mean(accuracy))

    # Registrar las métricas de rendimiento
    clf_performance['accuracy'].append(np.mean(accuracy))
    clf_performance['f1score'].append(np.mean(f1score))
    clf_performance['recall'].append(np.mean(recall))
    clf_performance['precision'].append(np.mean(precision))
    clf_performance['conf_matrix'].append(np.mean(conf_matrix_list)) 
    clf_performance['training_time'].append(np.mean(training_time))
    clf_performance['testing_time'].append(np.mean(testing_time))
    
    # Mostrar la matriz de confusión en la subfigura correspondiente
    sns.heatmap(conf_matrix, annot=True, linewidths=0.01, cmap="Purples", linecolor="gray", ax=axes[i])
    axes[i].set_xlabel("Predicción")
    axes[i].set_ylabel("Actual")
    axes[i].set_title(f"Confusion Matrix Validation {name}")
    

plt.tight_layout()
plt.show()


# In[231]:


a = np.mean(np.array(clf_performance['accuracy']), axis=1)
f = np.mean(np.array(clf_performance['f1score']), axis=1)
r = np.mean(np.array(clf_performance['recall']), axis=1)
p = np.mean(np.array(clf_performance['precision']), axis=1)


# In[192]:


a = np.mean(np.array(clf_performance['accuracy']), axis=1)
a


# In[234]:


df=pd.DataFrame(np.stack((a,f,r,p)),columns=['RF','SVM','KNN'],
            index=['Accuracy','F1-score','recall','precision'])
df


# In[236]:


pd.DataFrame(zip(np.array(clf_performace['training_time'])*65,np.array(clf_performace['testing_time'])),
            columns=['training_time','testing_time'],
            index=['KNN','SVM','RF'])


# In[ ]:


##ENVIO DE DATOS A ARDUINO


# In[237]:


import serial, time

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[239]:


import tensorflow as tf
import numpy as np

class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train_s, y_train):
        self.X_train = X_train_s.astype(np.float32)
        self.y_train = y_train.astype(np.int32)

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def predict(self, X_test_s):
        X_test = tf.convert_to_tensor(X_test_s, dtype=tf.float32)
        X_test = tf.cast(X_test, dtype=tf.float32)
        distances = tf.reduce_sum(tf.abs(tf.subtract(self.X_train, tf.expand_dims(X_test, axis=1))), axis=2)
        _, top_k_indices = tf.nn.top_k(-distances, k=self.k)
        top_k_labels = tf.gather(self.y_train, top_k_indices)
        predictions, _ = tf.unique(tf.argmax(tf.reduce_sum(tf.one_hot(top_k_labels, depth=tf.reduce_max(self.y_train)+1), axis=1), axis=1))
        return predictions

# Crear una instancia del clasificador KNN
knn_classifier = KNNClassifier(k=2)

# Entrenar el clasificador con datos de entrenamiento
knn_classifier.fit(X_train_s, y_train)

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_concrete_functions([knn_classifier.predict.get_concrete_function()])
tflite_model = converter.convert()


# In[240]:


# Guardar el modelo en disco
with open("Hand_model.tflite", "wb") as f:
    f.write(tflite_model)


# In[241]:


import os
basic_model_size = os.path.getsize("Hand_model.tflite")
print("Model is %d bytes" % basic_model_size)


# In[242]:


# Abre el archivo "model.h" en modo de escritura
with open("model.h", "w") as f:
    # Escribe el encabezado inicial en el archivo
    f.write("const unsigned char model[] = {\n")
    
    # Escribe el contenido del modelo tflite_model a una cadena de bytes
    f.write(", ".join(str(byte) for byte in tflite_model))
    
    # Escribe el encabezado final en el archivo
    f.write("\n};")

# Imprime un mensaje indicando que se ha creado el archivo
print("Archivo model.h creado correctamente.")


# In[243]:


import os
model_h_size = os.path.getsize("model.h")
print(f"Header file, model.h, is {model_h_size:,} bytes.")
print("\nOpen the side panel (refresh if needed). Double click model.h to download the file.")


# In[244]:


ruta_model_h = os.path.abspath("model.h")

# Imprimir la ruta del archivo model.h
print("Ruta del archivo model.h:", ruta_model_h)


# In[ ]:




