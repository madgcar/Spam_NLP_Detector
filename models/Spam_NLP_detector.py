#!pip install pandas
#!pip install regex
#!pip install matplotlib
#!pip install sklearn
#!pip install wordcloud
#!pip install seaborn

import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm
from sklearn.metrics import classification_report, accuracy_score
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# your code here

# Extraigo los datos

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv', sep=',')
df.info()

# EDA: elimino los duplicados
print("spam count: " +str(len(df.loc[df.is_spam==True])))
print("not spam count: " +str(len(df.loc[df.is_spam==False])))
print(df.shape)
df['is_spam'] = df['is_spam'].astype(int)

df = df.drop_duplicates()
df = df.reset_index(inplace = False)[['url','is_spam']]

clean_desc = []

for w in range(len(df.url)):
    desc = df['url'][w].lower()
    
    #remuevo las puntuaciones
    desc = re.sub('[^a-zA-Z]', ' ', desc)
    
    #remuevo los caracteres 
    desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)
    
    #remuevo los digitos y caracteres especiales
    desc=re.sub("(\\d|\\W)+"," ",desc)

    #remuevo los https
    desc=re.sub(r'(https://www|https://)', ' ',desc)

    #remuevo los https
    desc=re.sub(r"https", " " ,desc)

    #remuevo los www
    desc=re.sub(r"www", " " ,desc)
    
    clean_desc.append(desc)

#
df['url'] = clean_desc

df.head()

# Creo el vector para iniciar el modelo

vector = CountVectorizer().fit_transform(df['url'])

# Divido mis datos de entrenamiento y validacion

X_train, X_test, y_train, y_test = train_test_split(vector, df['is_spam'], stratify = df['is_spam'], random_state = 2207)

classifier = SVC(C = 1.0, kernel = 'linear', gamma = 'auto')

# Veamos los resultados del reporte estadistico
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))

# Tratamos con GridSearch para mejorar los hiperparametros

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(SVC(random_state=1234),param_grid,verbose=2)
grid.fit(X_train,y_train)

model = grid.best_estimator_
yfit = model.predict(X_test)
print(classification_report(y_test, yfit))

