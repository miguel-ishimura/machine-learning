
# coding: utf-8

# In[8]:


import pandas as pd #Manipulação de Dataframes  
import numpy as np    # cálculo numérico
np.random.seed(42)
np.set_printoptions(precision=3)
import seaborn as sns # biblioteca de gráficos estatísticos
import matplotlib.pyplot as plt #biblioteca gráfica básica
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #for train test split


# In[9]:


#Exportar a Base de Dados do excel
df_credit = pd.read_excel('/home/miguel/Desktop/german.xlsx', sheet_name = 'german')
df_credit.info() #Como observa-se, nao existe NaN


# In[10]:


df = df_credit.dropna()
#transformar as variaveis de sexo em variaveis de casado e solteiro
df['pers_status'].replace({"A91":"Casado","A92":"Casado","A94":'Casado',"A93":"Solteiro","A95":"Solteiro"})


# In[11]:


#Transformar response em categórica
convert_dict = {"response": "category"} 
df1 = df_credit.astype(convert_dict) 


# In[12]:


#Converter categóricas em dummies
df2=pd.get_dummies(df1)
df2.info()


# In[13]:


#Deixando as variáveis em escala (desvio padrão)
ss = StandardScaler()
# Subset das variaveis a deixar em escala
df_subset = df2.iloc[:,0:7]

# aplica o Scaler
df3 =pd.DataFrame(ss.fit_transform(df_subset),columns=df_subset.columns+'(scaled)')
df3.var()


# In[14]:


#concatenar o subset ao data frame original
df3.index=df2.index
dfscaled=pd.concat([df2,df3],axis=1,join="inner")
dfscaled


# In[15]:


#dropar as variaveis quantitativas originais e deixar apenas as em Scaled
df=dfscaled.drop(df_subset.columns,axis=1)
sns.heatmap(df.corr(),cmap="YlGnBu")
df


# In[19]:


#Preparar as DF de features (X) e resposta (y)
X = df.drop(['response_1','response_2'], axis = 1) #inputs
y = df['response_2'] #variável dependente
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X.info()


# In[20]:


###############################################################################
#iniciar modelo de regressao logistica
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_logreg=classifier.predict(X_test) #valores obtidos do modelo
y_logreg_prob=classifier.predict_proba(X_test)[:,1]

classifier.coef_ #coeficientes da Regressao

def metricas(y_test,y_logreg,y_logreg_prob):
    print("Confusion Matrix:\n",metrics.confusion_matrix(y_test,y_logreg))
    print("Accuracy:",metrics.accuracy_score(y_test,y_logreg))
    print("Precision:",metrics.precision_score(y_test,y_logreg))
    print("Recall:",metrics.recall_score(y_test,y_logreg))
    print("AUC:",metrics.roc_auc_score(y_test,y_logreg_prob))

def plotRoc(y_test,y_logreg_prob):
    auc=metrics.roc_auc_score(y_test,y_logreg_prob)

#plotting the ROC curve
    fpr,tpr,thresholds=metrics.roc_curve(y_test,y_logreg_prob)
    plt.plot(fpr,tpr,'b', label='AUC = %0.2f'% auc)
    plt.plot([0,1],[0,1],'r-.')
    plt.xlim([-0,1])
    plt.ylim([-0,1])
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return plt.show()

metricas(y_test,y_logreg,y_logreg_prob) #logReg

plotRoc(y_test,y_logreg_prob)


# In[21]:


#Random forest
rf=RandomForestClassifier(n_estimators=100) #cria o modelo de Random Forest
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)
y_tree=rf.predict(X_test)
y_tree_prob=rf.predict_proba(X_test)[:,1]

#resultados
metricas(y_test,y_tree,y_tree_prob) #random Forest
plotRoc(y_test,y_tree_prob)


# In[22]:


# Random Forest utilizando regressao logistica
Xmod=pd.DataFrame(classifier.predict(X),columns=["Prediction on RegLog"])

Xmod.index=X.index
Xmod=pd.concat([X,Xmod],axis=1,join="inner")#adiciona a prediction como feature

X_trainM, X_testM, y_trainM, y_testM = train_test_split(Xmod, y, random_state=0,stratify=y)

#cria modelo
rfMod=RandomForestClassifier(n_estimators=100)
rfMod.fit(X_trainM, y_trainM)

y_treemod=rfMod.predict(X_testM)
y_treemod_prob=rfMod.predict_proba(X_testM)[:,1]

#resultados
metricas(y_testM,y_treemod,y_treemod_prob) #randomforest Mod
plotRoc(y_test,y_tree_prob)

