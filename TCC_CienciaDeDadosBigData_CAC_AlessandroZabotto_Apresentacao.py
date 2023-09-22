#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install -U pip setuptools wheel
#!pip install -U spacy
#!python -m spacy download pt_core_news_sm


# In[ ]:


#Instalação das bibliotecas que utilizaremos
#!pip install freetype-py
#!pip install -U pip setuptools wheel
#!pip install -U spacy
#!python -m spacy download pt_core_news_sm
#!pip install unidecode
#!pip install wordcloud
#!pip install folium
#!pip install yellowbrick
#!pip install opencv-python


# In[ ]:


#Principais bibliotecas Python para trabalhar com NLP e plotagem de gráficos
import pandas as pd             #manipulação e análise de dados
import numpy as np              #trabalhos matemáticos
import re                       #tratamento de expressões regulares
import nltk                     #tratamento de linguagem natural (PLN ou NPL)
import spacy                    #mesmo conceito do nltk mais com recursos mais avançados
import matplotlib.pyplot as plt #plotagem gráficos 2D/3D, visualizações estáticas, animadas e interativas
import seaborn as sns           #criação de gráficos estatísticos elegantes e informativos
import os                       #comandos do sistema operacional
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#código para inibir as mensagens do sistema (Ex.: DeprecationWarning)
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# In[ ]:


os.chdir('C:\PucMinas\TCC_CAC\BD')  #definição do diretorio padrão onde constam os datasets


# In[ ]:


############################################## dataset CAC ##############################################################
#
#
#########################################################################################################################


# In[ ]:


#leitura do dataset CAC
dsCAC = pd.read_csv('CAC.csv', sep=";", encoding = 'cp1252')


# In[ ]:


#informações gerais sobre o dataset dsCAC
#Ex.: qtde de registros, colunas, tipo dos dados, etc.
dsCAC.info()


# In[ ]:


dsCAC.head()


# In[ ]:


dsCAC.tail()


# In[ ]:


#verificando quantos registros temos duplicados no dataset
dsCAC.duplicated().value_counts()


# In[ ]:


#exclusão da coluna NrEvento
dsCAC.drop(['NrEvento'],axis=1,inplace=True)


# In[ ]:


#incluindo a coluna TamDescriccao e TamExpectativaCliente com a qtde de caracteres contidos
#em cada coluna
dsCAC['Descricao'].fillna('',inplace=True)  #preenchendo os registros nulos
dsCAC['ExpectativaCliente'].fillna('',inplace=True)  #preenchendo os registros nulos
dsCAC.loc[:,'TamDescricao'] = dsCAC.Descricao.apply(lambda x: len(str(x)))
dsCAC.loc[:,'TamExpectativaCliente'] = dsCAC.ExpectativaCliente.apply(lambda x: len(str(x)))


# In[ ]:


dsCAC.head()


# In[ ]:


#concatenado (juntando) as colunas Descricao e ExpectativaCliente
dsCAC['ExpectativaCliente'].fillna('',inplace=True) #preenchendo os registros nulos da coluna ExpectativaCliente
dsCAC['Descricao'] = (dsCAC['Descricao']+ " " + dsCAC['ExpectativaCliente']) #fazendo a concatenação
dsCAC['Descricao'] = dsCAC.Descricao.apply(lambda x: "" if str(x) ==" " else x)


# In[ ]:


#atualizando a coluna TamanhoDescriccao com a qtde de caracteres contidos na coluna Descricao (após concatenação)
dsCAC.loc[:,'TamDescricao'] = dsCAC.Descricao.apply(lambda x: len(str(x)))
dsCAC.head()


# In[ ]:


#excluindo a coluna ExpectativaCliente e TamExpectativaCliente
dsCAC.drop(['ExpectativaCliente'],axis=1,inplace=True)
dsCAC.drop(['TamExpectativaCliente'],axis=1,inplace=True)


# In[ ]:


#monstrando os 5 primeiros registros
dsCAC.head()


# In[ ]:


#função para inverter a data do formto "d/m/yyyy" para "yyy/m/d"
def AcertaData(strData):
    strDataSplit = strData.replace(" 00:00:00", "")
    strDataSplit = strDataSplit.split("/")
    Ano = str(strDataSplit[2])
    Mes = str("0" + strDataSplit[1])
    Mes = Mes[-2:]
    Dia = str("0" + strDataSplit[0])
    Dia = Dia[-2:]
    strDataSplit = Ano + "/" + Mes + "/" + Dia
    
    return strDataSplit 


# In[ ]:


#Alterar ordem do campo data para "aaaa/mm/dd"
dsCAC['Data'] = dsCAC.Data.apply(AcertaData)


# In[ ]:


#mostrando 5 registro aleatórios
dsCAC.sample(5)


# In[ ]:


#converter o campo "Data" para formato datetime
dsCAC['Data'] = pd.to_datetime(dsCAC['Data'])


# In[ ]:


dsCAC.info()


# In[ ]:


#para análise estatistica, vamos incluir duas colunas,
#extraídas do campo "Data": "Mes" e "Ano"
dsCAC.loc[:,'Mes'] = dsCAC['Data'].dt.month
dsCAC.loc[:,'Ano'] = dsCAC['Data'].dt.year
dsCAC.info()


# In[ ]:


dsCAC.head()


# In[ ]:


#Extraindo a data mais antiga e mais nova do dataset CAC
print(dsCAC["Data"].min())
print(dsCAC["Data"].max())


# In[ ]:


############################################## dataset RVAT #############################################################
#
#
#########################################################################################################################


# In[ ]:


#leitura do dataset RAVT (rede de vendas e assistência técnica)
dsRVAT = pd.read_csv('RVAT.csv', sep=";", encoding = 'cp1252',decimal=',')


# In[ ]:


#informações gerais sobre o dataset dsRVAT
#Ex.: qtde de registros, colunas, tipo dos dados, etc.
dsRVAT.info()


# In[ ]:


#amostragem de dados utilizando o "sample"
dsRVAT.sample(6)


# In[ ]:


#ordenando o data set e retornando as UF
dsRVAT.sort_values('UF', ascending=True, inplace=True)
print(dsRVAT['UF'].unique())


# In[ ]:


############################################## ANÁLISE E EXPLORAÇÃO DOS DADOS ###########################################
#
#                                                        dsCAC                                                          #
#
#########################################################################################################################


# In[ ]:


#Verificando os segmentos do produto e serviço
dsCAC['SegProdutoServico'].value_counts()


# In[ ]:


#Mantendo no dataset dsCAC somente os resgistros que são das unidades de negócio Truck ou Bus
dsCAC = dsCAC.query('SegProdutoServico=="Truck" or SegProdutoServico=="Bus"')


# In[ ]:


dsCAC.info()
dsCAC['SegProdutoServico'].value_counts()


# In[ ]:


#Demonstração gráfica do volume de atendimento por ano
fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(data=dsCAC,x="Ano",palette='rocket')
plt.title('Atendimentos CAC por ano',fontsize=16, fontweight='bold')
ax.set_xlabel('Ano',fontsize=16, fontweight='bold')
ax.set_ylabel('Qtde. atendimentos',fontsize=12, fontweight='bold')
plt.show()


# In[ ]:


dsCAC_Agrup = dsCAC.copy()


# In[ ]:


dsCAC_Agrup = dsCAC_Agrup.sort_values(by=['Classificacao'],ascending=True)
dsCAC_Agrup = dsCAC_Agrup.groupby(["Classificacao"])
dsCAC_Agrup.describe()


# In[ ]:


#Visualizando em forma gráfica os atendimentos por ano e classificação
fix, ax = plt.subplots(figsize=(8,5))
sns.countplot(data=dsCAC, x="Ano", hue="Classificacao")
plt.grid(True, axis='y')
plt.legend(loc = "upper right")
plt.show()


# In[ ]:


#Analisando o tamanho da coluna descrição através do grafico de distribuição
fig, ax = plt.subplots(figsize=(10,0))
sns.set_theme(style="ticks", palette="dark:#5A9_r")
sns.displot(dsCAC['TamDescricao'],height=8,kde=True)
plt.title('Qtde. caracteres da descrição atendimento CAC', fontsize=12, fontweight='bold')


# In[ ]:


#Analisando o tamanha da coluna descrição através de diagrama de caixa
fig, aux = plt.subplots(figsize=(15,5))
sns.boxplot(x=dsCAC['TamDescricao'],notch=True, showcaps=True,palette="Set2",
            flierprops={"marker": "x"}, medianprops={"color": "red"})
plt.title('Distribuição tamanho da descrição', fontsize=14, fontweight='bold')


# In[ ]:


#Estatística geral da qtde caracteres da coluna Descricao
print("Estatistica qtde caracteres:")
print("Mínima        : ", + dsCAC['TamDescricao'].min())
print("Máxima        : ", + dsCAC['TamDescricao'].max())
print("Média         : ", + int(dsCAC['TamDescricao'].mean()))
print("Mediana       : ", + int(np.median(dsCAC['TamDescricao'])))
print("Desvio Padrão : ", + int(np.std(dsCAC['TamDescricao'])))


# In[ ]:


#fazendo uma cópia do dsCAC e retornando as descrições com tamanho = 0
dsCAC_Estatistica = dsCAC.copy()
dsCAC_Estatistica[dsCAC['TamDescricao']==0]


# In[ ]:


dsCAC_Estatistica[dsCAC['TamDescricao']==2197]


# In[ ]:


print('Maior descrição : {}'.format(dsCAC_Estatistica.loc[13094]['Descricao']))


# In[ ]:


############################################## ANÁLISE E EXPLORAÇÃO DOS DADOS ###########################################
#
#                                                        dsRVAT                                                         #
#
#########################################################################################################################


# In[ ]:


dsRVAT.info()


# In[ ]:


#formato do dataset
dsRVAT.shape


# In[ ]:


#Demonstração gráfica da qtde de RVAT por UF
fig, ax = plt.subplots(figsize=(15,5))
sns.countplot(data=dsRVAT,x='UF',palette='pastel')
plt.title('Qtde. RVAT por UF',fontsize=16, fontweight='bold')
ax.set_xlabel('UF',fontsize=16, fontweight='bold')
ax.set_ylabel('Qtde. RVAT',fontsize=12, fontweight='bold')
plt.grid(True, axis='y')
plt.show()


# In[ ]:


UF_Agrupado = dsRVAT.groupby(['UF', 'Municipio'])
UF_Agrupado.describe()


# In[ ]:


#leitura do dataset FrotaCirculante
dsFrota = pd.read_excel("FrotaCirculante.xlsx", sheet_name="FrotaCirculante")
dsFrota = dsFrota.sort_values(by=['FrotaCirculanteTotal'],ascending=False)


# In[ ]:


#unindo os datasets "RVAT" e "FrotaCirculante"
UF_GSSN = dsRVAT.groupby(['UF']).count()
UF_GSSN = pd.DataFrame(UF_GSSN)
dsRVAT_FC = UF_GSSN.merge(dsFrota, how = 'inner', on = 'UF')
dsRVAT_FC.sort_values('FrotaCirculanteTotal', ascending=False, inplace=True)
dsRVAT_FC = pd.DataFrame(dsRVAT_FC)
dsRVAT_FC =  dsRVAT_FC[['UF','GSSN','FrotaCirculanteTotal']]
dsRVAT_FC.columns = ['UF','Qtde_RVAT','FrotaCirculanteTotal']
def formatar(valor):
    return "{:,.2f}".format(valor)

dsRVAT_FC['FrotaCirculanteTotal']=dsRVAT_FC['FrotaCirculanteTotal'].apply(formatar)
dsRVAT_FC


# In[ ]:


#Demonstração gráfica da frota circulante
fig, ax = plt.subplots(figsize=(15,5))
ax.bar(dsFrota['UF'],dsFrota['FrotaCirculanteTotal'])
plt.title('Frota circulante por UF: Truck e Bus',fontsize=16, fontweight='bold')
ax.set_xlabel('UF',fontsize=12, fontweight='bold')
ax.set_ylabel('Qtde. veículos',fontsize=12, fontweight='bold')
plt.grid(True, axis='y')
plt.show()


# In[ ]:


############################################## ANÁLISE E EXPLORAÇÃO DOS DADOS ###########################################
#
#                                                        dsCAC e dsRVAT                                                 #
#
#########################################################################################################################


# In[ ]:


dsCAC.info()


# In[ ]:


dsCAC.isnull().sum()


# In[ ]:


#relacionando os datasets dsRVAT e dsCAC
dsRVAT_CAC = dsRVAT.merge(dsCAC, how = 'inner', on = 'GSSN')


# In[ ]:


dsRVAT_CAC.info()


# In[ ]:


#display(dsRVAT_CAC)


# In[ ]:


#Gerando gráfico de barras horizontais para demonstrar
#qtde de atendimentos via CAC por UF
dsUF_Atend = dsRVAT_CAC.groupby(by=['UF'])['GSSN'].count().reset_index()
dsUF_Atend = dsUF_Atend.sort_values(by=['GSSN'])
plt.figure(figsize=(10,10))
plt.title('Atendimento CAC por UF',fontsize=16, fontweight='bold')
plt.xlabel('Qtde.',fontsize=16,fontweight='bold')
plt.ylabel('UF',fontsize=16,fontweight='bold')
plt.barh(dsUF_Atend['UF'], dsUF_Atend['GSSN'], align='center')
plt.grid(True, axis='x')


# In[ ]:


###### Gerando gráfico de calor através do Folium
#importando a biblioteca Folium
import folium
from folium import plugins
from folium.plugins import HeatMap
from branca.colormap import LinearColormap  # Create a colormap instance
import json


# In[ ]:


dfLatLon = pd.DataFrame(dsRVAT_CAC, columns=['Latitude','Longitude']).values.tolist()
dsUF_Atend.rename(columns={'GSSN': 'QtdeAtend'}, inplace = True)


# In[ ]:


#definindo ponto inicial do mapa, zoom inicial e demais parametros
mapa = folium.Map(width='100%',height='100%', location=[-15.77972, -47.92972],
                  zoom_start=4.45, tile='Stamen Terrain')
colormap = LinearColormap(colors=['white', 'green','blue','yellow', 'red'],
                          vmin=dsUF_Atend['QtdeAtend'].min(), vmax=dsUF_Atend['QtdeAtend'].max())
colormap.caption = 'Índice de atendimento CAC nacional'
colormap.add_to(mapa)

#gerando o mapa de calor baseado na qtde de atendimentos CAC por latitude e longitude
HeatMap(dfLatLon, radius = 15).add_to(mapa)

# Criando o circulo e os tooltips com informações
dsMun_Atend= dsRVAT_CAC.groupby(by=['UF','Municipio','Latitude','Longitude'])['GSSN'].count().reset_index()
dsMun_Atend.rename(columns={'GSSN': 'QtdeAtend'}, inplace = True)

for i in range(0, len(dsMun_Atend)):
    folium.Circle(
        location = [dsMun_Atend.iloc[i]['Latitude'], dsMun_Atend.iloc[i]['Longitude']],
        color = '#000000',        
        fill = '#00A1B3',
        tooltip = '<li><bold> Municipio: ' + str(dsMun_Atend.iloc[i]['Municipio']) + 
        '<li><bold> Estado: ' + str(dsMun_Atend.iloc[i]['UF']) +
        '<li><bold> Qtde. atendimentos: ' + str(int(dsMun_Atend.iloc[i]['QtdeAtend'])),
        radius = 10
    ).add_to(mapa)
mapa


# In[ ]:


#########################################################################################################################
#
#                                            Criação de Modelos de Machine Learning                                     #
#
#########################################################################################################################


# In[ ]:


# removendo os outliers do dataset dsCAC
#dataset completo "dsCAC"
print("Dataset completo: ", dsCAC.shape)

#limite inferior do quartil
Q1 = np.percentile(dsCAC['TamDescricao'], 25, interpolation = 'midpoint')
 
#limite superior do quartil
Q3 = np.percentile(dsCAC['TamDescricao'], 75, interpolation = 'midpoint')

#interquartil
IQR = Q3 - Q1

#removendo os registros outliers (inferiores e superiores)
dsCAC_SemOutliers=dsCAC[(dsCAC.TamDescricao>=int(Q1-1.5*IQR)) & (dsCAC.TamDescricao<=int(Q3+1.5*IQR))]
dsCAC_SemOutliers = pd.DataFrame(dsCAC_SemOutliers)

#dataset excluindo os outliers
print("Novo dataset: ", dsCAC_SemOutliers.shape)


# In[ ]:


#importando as bibliotecas nltk e Spacy
import pt_core_news_sm
import re
import nltk
import unidecode
nltk.download('stopwords')
nltk.download('punkt')
spc_pt = pt_core_news_sm.load()
spc_pt = spacy.load('pt_core_news_sm')


# In[ ]:


def limpa_texto(descricao):
    
    #removendo todos os caracteres que não são ASCII e substituindo pelo caracter ASCII mais próximo
    descricao_ = unidecode.unidecode(descricao)
 
    # Remover caracteres que não são letras e tokenização
    descricao_ =  re.findall(r'\b[A-zÀ-úü]+\b', descricao_.lower())

    #Remover stopwords
    stopwords = nltk.corpus.stopwords.words('portuguese')
    
    #Adicionando stopwords que não estão na lista origiral
    stopwords.append("'")
    stopwords.append("area")
    stopwords.append("aberta")
    stopwords.append("abraco")
    stopwords.append("abraço")
    stopwords.append("veiculo")
    stopwords.append("contato")
    stopwords.append("atraves")
    stopwords.append("atrave")
    stopwords.append("cliente")
    stopwords.append("br")
    stopwords.append("km")
    stopwords.append("dia")
    stopwords.append("informar")
    stopwords.append("concessionario")
    stopwords.append("concessionaria")
    stopwords.append("conc")
    stop = set(stopwords)

    meaningful_words = [w for w in descricao_ if w not in stopwords]
    meaningful_words_string = " ".join(meaningful_words)

    #Instanciando o objeto spacy
    spc_descricao_ =  spc_pt(meaningful_words_string)

    #Lemmização 
    tokens = [token.lemma_ if token.pos_ == 'VERB' else str(token) for token in spc_descricao_]
    tokens_ = tokens

    #tratamento específico para o verbo "ir"
    ir = ['vou', 'vais', 'vai', 'vamos', 'ides', 'vão']
    tokens = ['ir' if token in ir else str(token) for token in tokens]
    
    return " ".join(tokens)


# In[ ]:


#aplica a função "limpa_texto" na coluna "Descricao" do dataset dsCAC_SemOutliers
#dsCAC_SemOutliers['Descricao']=dsCAC_SemOutliers['Descricao'].apply(limpa_texto)


# In[ ]:


#Gerando arquivo do dataset pre-processado
#dsCAC_SemOutliers.to_csv('dsCAC_preprocessado.csv', sep=";", index= False, columns= ['Classificacao','SubClassificacao','Data','Descricao','SegProdutoServico','GSSN','TamDescricao','Mes','Ano'])


# In[ ]:


#lendo aquivo dsCAC_SemOutliers já tratado
dsCAC_SemOutliers = pd.read_csv('dsCAC_preprocessado.csv', sep=";")


# In[ ]:


#filtrando somente os registros que possuem dados na coluna "Descricao"
dsCAC_SemOutliers = dsCAC_SemOutliers[dsCAC_SemOutliers['Descricao'].notnull()]


# In[ ]:


# Importando o CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#converte o texto em uma matrix de tokens contabilizados/contados
#min_df=5 --> desconsiderar palavra que apareça em menos de 5 registros
#max_df = .50 --> desconsiderar palavra que apareça em mais de 50% dos registros
#max_features=None --> será analisado todas as “fetures”
count_vect = CountVectorizer(min_df=5, max_df=.5, max_features=None)
bow_vector = count_vect.fit_transform(dsCAC_SemOutliers['Descricao'])
feature_names = count_vect.get_feature_names_out()

#Transformando uma matrix contabilizada/contada em uma representação tf/tf-idf normalizada
tfidf_transformer = TfidfTransformer().fit(bow_vector)

#transformando BoW em corpus TF-IDF
tfidf_vector = tfidf_transformer.transform(bow_vector) #train_data


# In[ ]:


tfidf_vector


# In[ ]:


tfidf_vector.shape[1]


# In[ ]:


#pip install opencv-python


# In[ ]:


#Nuvem de palavras com "Descricao" do dataset dsCAC_SemOutliers
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

# definindo uma imagem como máscara
mask1 = np.array(Image.open("MapaBrasil.png"))

font_path = "GoldUnderTheMud-Regular.ttf" #utilizando fonte não padrão
text = ' '.join(texto for texto in dsCAC_SemOutliers['Descricao'])

wordcloud = WordCloud(background_color="white",width=1000, height=500,
                      font_path=font_path, colormap="copper",
                      collocations = False,mask=mask1, min_word_length=3,max_words=200)
wordcloud.generate(text)
plt.figure(figsize=(20,15))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:


#identificando as palabras mais frequentes
from nltk.tokenize import word_tokenize
text_ = word_tokenize(text)
from nltk.probability import FreqDist
fdist = FreqDist(text_)
print(fdist.most_common(30))


# In[ ]:


#Gerando arquivo do dataset pre-processado
dsPalavrasContagem = pd.DataFrame(fdist.most_common(8000))
dsPalavrasContagem.to_csv('dsPalavrasContagem.csv', sep=";", index= False)


# In[ ]:


#plotagem do gráfico com as palavras mais frequentes
plt.figure(figsize=(10,5))
fd = nltk.FreqDist(text_)
fd.plot(30, title='Palavras x Frequência', cumulative = False)


# In[ ]:


CAC_text = nltk.Text(text_)
CAC_text.concordance('desbloqueio')


# In[ ]:


CAC_text.similar('desbloqueio')


# In[ ]:


################################## Análise da redução da dimensionalidade ##############################################
#
#                                                TruncatedSVD / LSA
########################################################################################################################


# In[ ]:


#redução da dimensionalidade em 3.000 componentes
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=3000)
svd_vec=svd.fit_transform(tfidf_vector)


# In[ ]:


#gerando gráfico da redução da dimensionalidade - identificação da qtde de componentes ideal
explained_variances=[i/np.sum(svd.explained_variance_ratio_) for i in svd.explained_variance_ratio_]
variances=[]
temp=0
for i in explained_variances:
    temp=temp+i
    variances.append(temp)
plt.plot(variances,label='Explained Variances')
plt.xlabel("explained Variances")
plt.show()


# In[ ]:


################################################# KMeans #########################################################
#
##################################################################################################################


# In[ ]:


#identificando a qtde de clusters ideal através do método do cotovelo
from sklearn.cluster import KMeans
wcss =[]
for i in range (1,30):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
  kmeans = kmeans.fit(tfidf_vector)
  print(i, kmeans.inertia_)
  wcss.append(kmeans.inertia_)


# In[ ]:


# MÉTODO DO COTOVELO: gerando gráfico para análise dos dados
plt.plot(range(1,30),wcss)
plt.title("Método do cotovelo", fontweight='bold')
plt.xlabel("Número de clusters")
plt.ylabel("WSS - soma dos quadrados")
plt.show()


# In[ ]:


#MÉTODO DA SILHOUETTE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
Sil = []
for i in range(2,30):
    clusterer = KMeans(n_clusters=i, n_init='auto', random_state=0)
    preds=clusterer.fit_predict(tfidf_vector)
    score = silhouette_score(tfidf_vector, preds)
    Sil.append(score)
    print('Silhouette para ' + str(i) + ' clusters :' + str(score))


# In[ ]:


#MÉTODO DA SILHOUETTE - gerando gráfico para análise dos dados
plt.plot(range(2,30), Sil)
plt.title("Silhouette", fontweight = 'bold')
plt.xlabel("Qtde de clusters")
plt.ylabel("Coeficiente")
plt.show()


# In[ ]:


#Treinamento do modelo
from sklearn.cluster import KMeans
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters,  init='k-means++',
                n_init=10, max_iter = 300, random_state=0)
kmeans = kmeans.fit(tfidf_vector)
centroides = kmeans.cluster_centers_
labels = kmeans.labels_
dsCAC_SemOutliers['Cluster'] = kmeans.fit_predict(tfidf_vector)


# In[ ]:


print(dsCAC_SemOutliers['Cluster'].value_counts())


# In[ ]:


#Demonstração gráfica dos clusters
fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(data=dsCAC_SemOutliers,x="Cluster",palette='rocket')
plt.title('Alocação dos atendimentos por cluster',fontsize=16, fontweight='bold')
ax.set_xlabel('Cluster',fontsize=16, fontweight='bold')
ax.set_ylabel('Qtde. atendimentos',fontsize=12, fontweight='bold')
plt.grid(True, axis='y')
plt.show()


# In[ ]:


print('Palavras mais frequentes por cluster:')
order_centroids = centroides.argsort()[:,::-1]
terms = count_vect.get_feature_names_out()
for i in range(10):
    print('Cluster --> {}:'.format(i))
    for ind in order_centroids[i,:10]:
        print(' %s' % terms[ind],end='')
    print()


# In[ ]:


fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(data=dsCAC_SemOutliers, x="Cluster", hue="Ano", palette='YlGnBu')
plt.title('Quantidade de atendimentos Cluster X Ano', fontsize=16, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('Quantidade', fontsize=12, fontweight='bold')
plt.show()


# In[ ]:


#preparação do dataset 
bow_transformer = count_vect.fit(dsCAC_SemOutliers['Descricao'])

print('Palavras mais frequentes por cluster:')
order_centroids = centroides.argsort()[:,::-1]
lista_terms=[]

for i in range(0,10):
    print('\n Cluster {}: '.format(i))
    for x in order_centroids[i,:10]:
        cluster = i
        term=feature_names[x]
        valor = tfidf_transformer.idf_[bow_transformer.vocabulary_[term]]
        print(term, ': ', valor)
        lista_terms.append([i, term, valor])
dsPalavras = pd.DataFrame(lista_terms, columns=['Cluster','Palavra', 'TF-IDF'])


# In[ ]:


dsPalavras


# In[ ]:


dsPalavras.head(15)


# In[ ]:


#Gerando arquivo do dataset pre-processado
dsPalavras.to_csv('dsPalavras.csv', sep=";", index= False, columns= ['Cluster','Palavra','TF-IDF'])


# In[ ]:


#plotagem do gráfico dos 10 clustes com suas palavras e respectivos TF-IDF
plt.rcParams["figure.figsize"] = [15.00, 12.00]
plt.rcParams["figure.autolayout"] = True
num_linhas = 5
num_colunas = 2
fig, axes = plt.subplots(num_linhas,num_colunas)

linha = 0
coluna = 0

for i in range(0,10):
    sns.barplot(data=dsPalavras[dsPalavras['Cluster']==i], x='TF-IDF',y='Palavra',
                orient='h', palette='YlGnBu',ax=axes[linha][coluna])
    plt.title('Palavras mais frequentes no Cluster {}'.format(i), fontsize=14, fontweight='bold')
    plt.xlabel('TF-IDF', fontsize=12, fontweight='bold')
    plt.ylabel('Palavra', fontsize=12, fontweight='bold')
    coluna += 1
    if coluna == num_colunas:
        linha += 1
        coluna = 0   
plt.show()    


# In[ ]:


dsPercCluster = dsCAC_SemOutliers['Cluster'].value_counts().reset_index()


# In[ ]:


dsPercCluster['Percentual'] = dsPercCluster['count']/dsCAC_SemOutliers.shape[0]*100


# In[ ]:


dsPercCluster = dsPercCluster.sort_values(by=['Percentual'], ascending=False)


# In[ ]:


dsPercCluster


# In[ ]:


dsCAC_SemOutliers[dsCAC_SemOutliers['Cluster'] ==0]['Descricao'] 


# In[ ]:


dsCAC


# In[ ]:


dsCAC.groupby(by=['Classificacao','SubClassificacao'])['Data'].count().reset_index()


# In[ ]:


dsCAC.groupby(by=['SubClassificacao'])['Data'].count().reset_index()


# In[ ]:


dsUF_Atend = dsRVAT_CAC.groupby(by=['UF'])['GSSN'].count().reset_index()


# In[ ]:


dsUF_Atend


# In[ ]:





# In[ ]:




