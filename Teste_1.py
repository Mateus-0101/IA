# Bibliotecas
import sys
import scipy
import numpy as np
import matplotlib
import pandas as pd
import sklearn
import seaborn as sns

# Carregar Bibliotecas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Caregar dados através da URL:
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(url, names = attributes)
dataset.columns = attributes

# shape -> Quantidade de instâncias e atributos
print(dataset.shape)

# head -> Analisar dados
#      -> Visualizar as primeiras 20 linhas
print(dataset.head(30))

# descriptions -> Resumo Estatístico
print(dataset.describe())

# class distribution -> Oservar nº de instâncias em cada classe
print(dataset.groupby('class').size())

# Gráficos Univariados: Entender melhor cada atributo
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# Gráficos Multivariados: Entender melhor as relações entre os atributos
# scatter plot matrix -> Gráfico de dispersão
scatter_matrix(dataset)
plt.show()

# Avaliação de Algoritmos:

# Split-out validation dataset
# Treino = 80% e Teste = 20%
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

speed = 7
scoring = 'acurracy'

# Teste de algoritmos
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma ='auto')))

# Evaluate each model in turn
results = []
names = []

for name, model in models:
   kfold = model_selection.KFold(n_splits=10, random_state=seed)
   cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
   results.append(cv_results)
   names.append(name)
   msg ='%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
   print(msg)
