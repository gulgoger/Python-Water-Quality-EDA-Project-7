import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import precision_score, confusion_matrix

from sklearn import tree

#Read and Analyse Data

df = pd.read_csv("water_potability.csv")
df.head()

describe = df.describe()

#Dependent Variable Analysis

d = pd.DataFrame(df["Potability"].value_counts())
fig = px.pie(d,values="Potability", names = ["Not Potable","Potable"],hole=0.4,opacity=0.8,
            labels={"label":"Potability","Potability": "Number of Samples"})
fig.update_layout(title = dict(text="Pie Chart of Potability Feature"))
fig.update_traces(textposition = "outside", textinfo = "percent+label")
fig.show()


#Correlation Between Features
df.corr()
sns.clustermap(df.corr(),cmap = "vlag",dendrogram_ratio = (0.1, 0.2), annot = True, linewidths =.8,figsize=(9,10))
plt.show()


#Distribution of Features

non_potable = df.query("Potability == 0")
potable = df.query("Potability == 1")

plt.figure(figsize= (15,15))
for ax, col in enumerate(df.columns[:9]):
    plt.subplot(3,3, ax+1)
    plt.title(col)
    sns.kdeplot(x = non_potable[col], label="Non Potable")
    sns.kdeplot(x = potable[col], label = "Potable")
    plt.legend()
plt.tight_layout()


#Preprocessing: Missing Value Problem

msno.matrix(df) # showing missing values
plt.show()

df.isnull().sum()

df["ph"].fillna(value = df["ph"].mean(), inplace = True)
df["Sulfate"].fillna(value = df["Sulfate"].mean(), inplace = True)
df["Trihalomethanes"].fillna(value = df["Trihalomethanes"].mean(), inplace = True)

#Preprocessing: Train-Test Split and Normalization

X = df.drop("Potability", axis = 1).values
y = df["Potability"].values

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=3)
print("X_train",X_train.shape)
print("X_test",X_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)

#normalization
x_train_max = np.max(X_train)
x_train_min = np.min(X_train)
X_train = (X_train - x_train_min)/(x_train_max - x_train_min)
X_test = (X_test - x_train_min)/(x_train_max - x_train_min)

#Modelling: Decison Tree and Random Forest Classifiers

models = [("DTC", DecisionTreeClassifier(max_depth = 3)),
         ("RF", RandomForestClassifier())]

finalResults = []
cmList = []
for name, model in models:
    model.fit(X_train, y_train)
    model_result = model.predict(X_test)
    score = precision_score(y_test, model_result)
    cm = confusion_matrix(y_test,model_result)
    
    finalResults.append((name,score))
    cmList.append((name, cm))
finalResults

for name, i in cmList:
    plt.figure()
    sns.heatmap(i, annot=True, linewidths = 0.8, fmt= ".1f")
    plt.title(name)
    plt.show()


#Visualize Decison Tree

dt_clf = models[0][1]
dt_clf

plt.figure(figsize=(20,19))
tree.plot_tree(dt_clf, feature_names=df.columns.tolist()[:-1],class_names=["0","1"],filled=True,
               precision=5)
plt.show()   #gini saflık derecesidir denebilir formülü olan bişey

#Random Forest Hyperparameter Tuning

model_params = {
    "Random Forest":
    {
        "model": RandomForestClassifier(),
        "params":
        {
            "n_estimators":[10,50,100],
            "max_features":["auto","sqrt","log2"],
            "max_depth":list(range(1,15,3))
        }
    }
}
model_params

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
scores = []
for model_name, params in model_params.items():
    rs = RandomizedSearchCV(params["model"],params["params"],cv=cv, n_iter = 10)
    rs.fit(X,y)
    scores.append([model_name,dict(rs.best_params_),rs.best_score_])
scores


