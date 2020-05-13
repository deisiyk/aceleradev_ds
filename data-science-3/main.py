#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from yellowbrick.model_selection import RFECV

from loguru import logger


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

sns.set()


# In[3]:


fifa = pd.read_csv("fifa.csv")


# In[4]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
fifa.head()


# In[6]:


#Quantidade de linhas e colunas
fifa.shape


# In[7]:


#Dataframe auxiliar para análise dos dados
fifa_aux = pd.DataFrame({'Columns': fifa.columns,
                      'Type': fifa.dtypes,
                      'Missing': fifa.isna().sum(),
                      'Size': fifa.shape[0]
                     })
fifa_aux['Missing_%']= fifa_aux.Missing/fifa_aux.Size * 100
fifa_aux


# In[8]:


fifa_semNan = fifa.dropna()


# In[9]:


fifa_semNan.shape


# In[10]:


fifa_semNan.describe()


# In[11]:


pca = PCA().fit(fifa_semNan)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# In[12]:


#Representação numérica do gráfico acima
#Com 1 componente consegue-se explicar 56% das variância total
#Com 2 componentes consegue-se explicar 74% das variância total e assim por diante
list(zip(range(1,38),np.cumsum(pca.explained_variance_ratio_)))


# # Utilizando dados padronizados

# In[13]:


#transforma o dataframe padronizado
fifa_semNan_stand = sct.zscore(fifa_semNan)      


# In[14]:


pca_norm = PCA().fit(fifa_semNan_stand)


# In[15]:


# PCA Padronizado e não padronizado. Verifique que existe uma diferença sutil mas existe. 
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2 , figsize=(11,5))     
ax1.plot(np.cumsum(pca.explained_variance_ratio_))
ax2.plot(np.cumsum(pca_norm.explained_variance_ratio_))
ax1.set(title='PCA NÃO PADRONIZADA', xlabel='number of components', ylabel='cumulative explained variance')   
ax2.set(title='PCA PADRONIZADA', xlabel='number of components', ylabel='cumulative explained variance') 
fig.tight_layout()


# In[16]:


#Passando o n_components como um inteiro, o PCA compreende que deseja retornar somente 1 componente
#1 componente PCA consegue explicar 56% da variância total
pca = PCA(n_components = 1).fit(fifa_semNan)
pca.explained_variance_ratio_
# round(float(pca.explained_variance_ratio_),3)


# In[17]:


#Passando o n_componentes como um float, o PCA comprrende que deseja retornar as variâncias até 
#atingir o valor informado ou mais
#No exemplo abaixo ira retornar as variâncias dos componentes até atingir o 0.95 ou mais
#Somando as variancia dos 15 componentes gerados, atingirá os 0.95
pca_porc = PCA(n_components = 0.95).fit(fifa_semNan)
pca_porc.explained_variance_ratio_


# In[18]:


len(pca_porc.explained_variance_ratio_)


# ### Considerando o grafico PCA gerado com 2 componentes
# Encontrar as coordenadas do ponto X

# In[19]:


pca2 = PCA(n_components=2).fit(fifa_semNan)


# In[20]:


#retorna as axes da matriz rotacionada para os dois componentes gerados no PCA2
pca2.components_


# In[21]:


#para encontrar as coordenadas do ponto x neste gráfico gerado
x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]
pca2.components_.dot(x)


# ## Realizar RFE

# In[22]:


fifa_semNan.head()


# In[23]:


#target Overall
X= fifa_semNan.loc[:,fifa_semNan.columns != 'Overall']
y = fifa_semNan.Overall


# In[24]:


#instancia a classe
reg = LinearRegression()


# In[25]:


#Ranking das features com RFE com cross validation
#O ponto tracejado representa o score maximo com 27 features
rfecv = RFECV(reg,step=1, cv=3)
rfecv.fit(X,y)
rfecv.show()


# In[26]:


#lista de features utilizadas pelo modelo
list(zip(X.columns, rfecv.support_))


# In[27]:


#seleciona 5 features para o modelo utilizando somente RFE
rfe = RFE(reg, n_features_to_select=5, step=1 )
rfe = rfe.fit(X,y)


# In[28]:


rfe.support_


# In[29]:


rfe.estimator_.coef_


# In[30]:


#lista de features utilizadas pelo modelo
#list(zip(X.columns, rfe.support_==True))
result = pd.DataFrame({
    'coluna': X.columns,
    'bool': rfe.support_})


# In[31]:


result.loc[result['bool']==True, 'coluna'].tolist()


# In[ ]:





# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[ ]:


def q1():
    pca = PCA(n_components = 1).fit(fifa_semNan)
    return round(float(pca.explained_variance_ratio_),3)


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[ ]:


def q2():
    pca_porc = PCA(n_components = 0.95).fit(fifa_semNan)
    pca_porc.explained_variance_ratio_
    return len(pca_porc.explained_variance_ratio_)


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[ ]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[ ]:


def q3():
    pca2 = PCA(n_components=2).fit(fifa_semNan)
    pca2.components_.dot(x)
    return tuple(np.round(pca2.components_.dot(x),3))


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[ ]:


def q4():
    X= fifa_semNan.loc[:,fifa_semNan.columns != 'Overall']
    y = fifa_semNan.Overall   #variavel target
    rfe = RFE(reg, n_features_to_select=5, step=1 )  #seleciona 5 variaveis
    rfe = rfe.fit(X,y)
    result = pd.DataFrame({
    'coluna': X.columns,
    'bool': rfe.support_})
    return result.loc[result['bool']==True, 'coluna'].tolist()

