#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
import scipy.stats as sct

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from math import sqrt, log


# In[4]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[5]:


countries = pd.read_csv("countries.csv")


# In[6]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[7]:


# Sua análise começa aqui.

#Criação do dataset auxiliar para análise inicial dos dados
df_aux = pd.DataFrame({
                    'Columns': countries.columns,
                    'Type': countries.dtypes,
                    'Nan': countries.isnull().sum()
                    })
df_aux['Nan%'] = round(df_aux.Nan/countries.shape[0]*100,2)
df_aux


# In[8]:


#por apresentar poucos valores distintos Climate poderia ser considerada uma variável categórica
countries.Climate.value_counts()


# In[9]:


col = ['Pop_density', 'Coastline_ratio', 'Net_migration', 'Infant_mortality',
       'Literacy', 'Phones_per_1000', 'Arable', 'Crops', 'Other','Climate',
       'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']


# In[10]:


# nas colunas que devem ser transformadas em numéricas
# substitui a virgula por ponto
# convert de string para float

def replace(x):
    if pd.isnull(x):
        pass
    else:
        x = x.replace(',','.')
        x = float(x)
    return x
            
for i in col:
    countries[i] = list(map(replace, countries[i] ))


# ## Análise Questão 1

# In[11]:


# identificando espaços nas colunas Country e Region
countries.Country[0]


# In[12]:


countries.Region[0]


# In[13]:


# remove espaços em branco no começo e no final da string
# caso fosse remover somente em uma das posições lstrip() rstrip()
# como a string é imutável, é necessário que outra variável receba seu valor
countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()


# In[14]:


sorted(countries.Region.value_counts().index.tolist()) 


# ## Análise questão 2

# In[15]:


countries.Pop_density.hist()


# In[16]:


# existem apenas dois registros nos intervalos acima maiores que 2000
countries[countries.Pop_density>2000]


# In[17]:


#Criação de variável que representa intevalos, semelhante ao pd.cut
discretizer = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
discretizer.fit(countries[['Pop_density']])


# In[18]:


# limites para cada range
discretizer.bin_edges_


# In[19]:


# numericamente representa em qual intervalo ficara cada linha
score_bins = discretizer.transform(countries[['Pop_density']])


# In[20]:


score_bins[:,0]


# In[21]:


bin_edges_quantile = discretizer.bin_edges_[0]
len(bin_edges_quantile)


# In[22]:


#lista a quantidade de registros por intervalo de 'Pop_density'
#cada intervalo contém 10% dos registros
def get_intervals(bin_idx, bin_edge):
    return f"{np.round(bin_edge[bin_idx],2):.2f} to {np.round(bin_edge[bin_idx+1],2):.2f}"

for i in range(len(bin_edges_quantile) -1):
    print(f"{get_intervals(i, bin_edges_quantile)} : qtde registros {sum(score_bins[:,0] == i)} ")


# In[23]:


#considerando que cada linha é um país
#a quantidade de paises que se encontram acima de 90 percentil
sum(score_bins[:,0]>=9)


# ## Análise questão 3

# In[24]:


#quantidade de registros por Region
countries.Region.value_counts()


# In[25]:


#quantidade de registros por Climate
countries.Climate.value_counts(dropna=False)


# In[26]:


#registros cujo Climate esta nan
countries[countries['Climate'].isnull()]


# In[27]:


#preencher estes registros com 0
countries[['Climate']] = countries[['Climate']].fillna(value=0)


# In[28]:


onehotencoder_Region = preprocessing.OneHotEncoder().fit(countries[['Region']])


# In[29]:


onehotencoder_Region.categories_


# In[30]:


len(onehotencoder_Region.categories_[0])


# In[31]:


countries['Climate'].value_counts()


# In[32]:


onehotencoder_Climate = preprocessing.OneHotEncoder(categories='auto').fit(countries[['Climate']])
onehotencoder_Climate.categories_


# In[33]:


len(onehotencoder_Climate.categories_[0])


# ## Análise questão 4

# In[34]:


countries.select_dtypes(include=[np.number])


# In[35]:


pipeline = Pipeline(steps=[
    ("imputer_median", SimpleImputer(strategy="median")),
    ("standardscaler", preprocessing.StandardScaler())
])


# In[36]:


#comando único seria pipeline.fit_transform(countries_number)
countries_number = countries.select_dtypes(include=[np.number])
fit = pipeline.fit(countries_number)
transf = fit.transform(countries_number)


# In[37]:


#retorna todas as colunas do tipo numerico padronizadas
transf[0]


# In[38]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

transf_test = pipeline.transform([test_country[2:]])


# In[39]:


transf_test


# In[40]:


#retorna a posição (index) da coluna Arable
countries_number.columns.get_loc('Arable')


# In[41]:


arable = transf_test[:, countries_number.columns.get_loc('Arable') ]


# In[42]:


float(np.around(arable.item(),3))


# ## Análise questão 5
# Apenas para estudo e comparação, serão aplicados as 3 técnicas de remoção de outlier nos dados: IQR, Histograma

# In[43]:


#IQR
plt.figure(figsize=(10,5))
sns.boxplot(countries.Net_migration)


# In[44]:


#tudo que estiver neste intervalo não é considerado outlier
q1= countries.Net_migration.quantile(0.25)
q3 = countries.Net_migration.quantile(0.75)
iqr = q3 - q1
non_outlier_iqr = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
non_outlier_iqr


# In[45]:


#quantidade abaixo e acima
len(countries.Net_migration[(countries.Net_migration < non_outlier_iqr[0]) ])
len(countries.Net_migration[(countries.Net_migration > non_outlier_iqr[1]) ])


# In[46]:


outlier_iqr = countries.Net_migration[(countries.Net_migration<non_outlier_iqr[0] )|
                                      (countries.Net_migration>non_outlier_iqr[1])]


# In[47]:


# 22% dos dados são considerados outliers por esta técnica
print('Quantidade de outliers:' ,len(outlier_iqr))
print('Porcentagem % :' , round(len(outlier_iqr)/len(countries.Net_migration)*100,2))


# In[48]:


#como existe 3 registros com Nan, estes campos serão preenchidos para completar a análise
Net_migration_semNAN = countries.Net_migration.fillna(value=0)


# In[49]:


#Histograma
sns.distplot(Net_migration_semNAN)


# In[50]:


#Pelo histograma uma forma de identificar os outliers é o dado estar fora do intervalo
#[media - k * std, media + k * std ], onde k é geralmente 1.5, 2, 2.5 ou 3
#pela analise visual, vamos considerar k=2.0
mean = Net_migration_semNAN.mean()
std = Net_migration_semNAN.std()
print(mean, std)


# In[51]:


non_outlier_hist = [mean - 2.0 * std, q1 + 2.0 * std]
non_outlier_hist


# In[52]:


outlier_hist = Net_migration_semNAN[(Net_migration_semNAN<non_outlier_hist[0] )|
                                      (Net_migration_semNAN>non_outlier_hist[1])]
# 8% dos dados são considerados outliers por esta técnica
print('Quantidade de outliers:' ,len(outlier_hist))
print('Porcentagem % :' , round(len(outlier_hist)/len(Net_migration_semNAN)*100,2))


# ## Análise questão 6

# In[53]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[54]:


#Bunch é uma subclasse da classe DIC. Ele permite usar keys como atributos
type(newsgroup)


# In[55]:


#chaves do dataset
newsgroup.keys()


# In[56]:


#forma de acessar os dados
newsgroup.data[0]


# In[57]:


#converte um coleção de documentos em uma matriz com o total de tokens
vectorizer = CountVectorizer(analyzer='word')
vectorizer_fit = vectorizer.fit_transform(newsgroup.data)


# In[58]:


#A palavra phone esta no vetor na posição 19211. Lembrando que os indices em python iniciam com 0
vectorizer.vocabulary_['phone']


# In[59]:


#tamanho do vetor
vectorizer_fit.shape


# In[60]:


#quantidade de vezes que a palavra aparece
vectorizer_fit[:,19211].sum()


# ## Análise questão 7

# In[61]:


tfid = TfidfVectorizer()
tfid_fit = tfid.fit_transform(newsgroup.data)


# In[62]:


tfid.vocabulary_['phone']


# In[63]:


tfid_fit.shape


# In[64]:


#score dado pela quantidade de vezes que a palavra aparece. 
#Quanto menos aparecer maior o score
tfid_fit[:,19211].sum()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[65]:


def q1():
    return sorted(countries.Region.value_counts().index.tolist()) 
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[66]:


def q2():
    #Criação de variável que representa intevalos, semelhante ao pd.cut
    discretizer = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    discretizer.fit(countries[['Pop_density']])
    return int(sum(score_bins[:,0]>=9))
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[67]:


def q3():
    #preencher estes registros com 0
    countries[['Climate']] = countries[['Climate']].fillna(value=0)
    onehotencoder_Region = preprocessing.OneHotEncoder().fit(countries[['Region']])
    #necessário colocar parametro categories=auto para não juntar 1 e 1.5
    onehotencoder_Climate = preprocessing.OneHotEncoder(categories='auto').fit(countries[['Climate']])
    soma = len(onehotencoder_Region.categories_[0])+ len(onehotencoder_Climate.categories_[0])
    return soma
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[ ]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[68]:


def q4():
    pipeline = Pipeline(steps=[
    ("imputer_median", SimpleImputer(strategy="median")),
    ("standardscaler", preprocessing.StandardScaler())
    ])
    #comando único seria pipeline.fit_transform(countries_number)
    countries_number = countries.select_dtypes(include=[np.number])
    fit = pipeline.fit(countries_number)
    transf = fit.transform(countries_number)
    #aplica o pipeline para os dados fornecidos
    transf_test = pipeline.transform([test_country[2:]])
    ##retorna a posição (index) da coluna Arable e busca no vetor
    arable = transf_test[:, countries_number.columns.get_loc('Arable') ]
    return float(np.around(arable.item(),3))
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[69]:


def q5():
    #tudo que estiver neste intervalo não é considerado outlier
    q1= countries.Net_migration.quantile(0.25)
    q3 = countries.Net_migration.quantile(0.75)
    iqr = q3 - q1
    non_outlier_iqr = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
    #quantidade abaixo e quantidade acima.
    #analise indicou 30% de outlier, uma quantidade significativa que naão deve ser removida
    lower = len(countries.Net_migration[countries.Net_migration<non_outlier_iqr[0] ])
    upper = len(countries.Net_migration[countries.Net_migration>non_outlier_iqr[1] ])
    return (lower, upper, False)
q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[70]:


def q6():
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    vectorizer = CountVectorizer(analyzer='word')
    vectorizer_fit = vectorizer.fit_transform(newsgroup.data)
    return int((vectorizer_fit[:,vectorizer.vocabulary_['phone']].sum()))
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[71]:


def q7():
    tfid = TfidfVectorizer()
    tfid_fit = tfid.fit_transform(newsgroup.data)
    return float(round(tfid_fit[:,tfid.vocabulary_['phone']].sum(),3))
q7()


# In[ ]:




