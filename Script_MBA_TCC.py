#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Instalação dos pacotes

'''
!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install playsound
!pip install pingouin
!pip install emojis
!pip install statstests
!pip3 install PyObjC
!pip install networkx
pip install tensorflow
'''

#%% Importação dos pacotes
from scipy.special import inv_boxcox
import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
import plotly.graph_objects as go # gráficos 3D
from sklearn.preprocessing import LabelEncoder # transformação de dados
from sklearn.experimental import enable_iterative_imputer  # Necessário para habilitar o IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, mean_squared_error
import pingouin as pg # outro modo para obtenção de matrizes de correlações
import emojis # inserção de emojis em gráficos
from statstests.process import stepwise # procedimento Stepwise
from statstests.tests import shapiro_francia # teste de Shapiro-Francia
import statsmodels.api as sm # estimação de modelos
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.robust.robust_linear_model import RLM
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import boxcox # transformação de Box-Cox
from scipy.stats import norm # para plotagem da curva normal
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr # correlações de Pearson
from scipy import stats # utilizado na definição da função 'breusch_pagan_test'
from scipy.stats import shapiro
import networkx as nx
import matplotlib.cm as cm
from sklearn.ensemble import IsolationForest
from patsy import dmatrices
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.patches as patches
import math


#%% Arquivo 01 com ganho80

# Carregar os dados
dados_ganho80 = pd.read_excel('../Fenotipos_ajustados/lucro_whole.xls')
dados_ganho80.columns

arquivo_01 = dados_ganho80[["CGA", "ganho_80"]]
arquivo_01 = arquivo_01.rename(columns={"ganho_80": "GP80"})

#%% Arquivo 02

dados = pd.read_table("lucro_ims_frame_ag_elisa.txt", sep = " ")

#Selecionando as colunas
dados.columns

dados = dados.drop(['gc2', 'idade_80', 'idade_qd', 'lucro_80_', 'ano', 'NFA', 'sx', 'GCN240', \
                    'GCN365', 'GCN455', 'GCN550', 'C365', 'CIVP', 'GCims', 'ID_D0', 'idade', 'idade2', 'P365', 'P550', \
                        'gcus', 'GCPN', 'civp_pn'], axis = 1)

dados = dados.rename(columns={'cga': 'CGA', 'lucratividade': 'Lucratividade', "AOL_CM2": 'AOL', "EGP8_MM":"EGG", "ims": "IMS"})

dados.head()

#Criar colunas de Ganho de Peso 
#Esse código deve criar a coluna GP de forma eficiente e garantir que, se houver NaN nas colunas de entrada, o resultado também será NaN)

    # Ganho de peso a desama:
dados['GP_PRED'] = np.where(dados['PN'].isna() | dados['P240'].isna(), np.nan, 
                                     (dados['P240'] - dados['PN']) / 240)
    
    #Ganho peso pós desmama = peso ao ano - peso desmama / (365 - 240)
dados['GP_POSD'] = np.where(dados['P455'].isna() | dados['P240'].isna(), np.nan, 
                                     (dados['P455'] - dados['P240']) / (455-240))

#%% Arquivo CAR

dados_car = pd.read_excel('dados_ea.xlsx')
dados_car.columns

arquivo_02 = dados_car[["cga", "car"]]
arquivo_02 = arquivo_02.rename(columns={'cga': 'CGA', "car": "CAR"})

#%% Juntando os arquivos

# Primeiro merge: dados com arquivo_01
temp = pd.merge(dados, arquivo_01, how='inner', on='CGA')
# Segundo merge: resultado anterior com arquivo_02
arquivo_final = pd.merge(temp, arquivo_02, how='inner', on='CGA')

arquivo_final = arquivo_final.drop(columns=['CGA'])
arquivo_final.columns

#Substituir zeros por NA
arquivo_final = arquivo_final.replace(0, pd.NA)


nova_ordem = [
    'Lucratividade', 'PN', 'P240', 'P455', 'GP80', 
    'AOL', 'EGG', 'CAR', 'IMS' , 'GP_PRED', 'GP_POSD'
]
# Reordena o DataFrame
arquivo_final = arquivo_final[nova_ordem]

arquivo_final.info()

#Converter objeto para numérico
# Contar NaNs antes da conversão
na_antes = arquivo_final.isna().sum()
#Conversão
arquivo_final = arquivo_final.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
# Contar NaNs depois da conversão
na_depois = arquivo_final.isna().sum()
# Diferença nos NaNs por coluna
diferenca = na_depois - na_antes
# Exibir colunas com aumento de NaNs
diferenca[diferenca > 0]

arquivo_final = arquivo_final.drop(columns=['GP_PRED', 'GP_POSD']) #eu fiz as análises com GP, porém deu multicolinearidade, então estou removendo aqui
arquivo_final = arquivo_final.dropna()

arquivo_final = arquivo_final.drop(columns='GP80')

arquivo_final.to_excel('Dados_brutos_final.xlsx', index=False)


#%% Remocao Outliers

# Copiar o DataFrame original
df = arquivo_final.copy()

# Colunas a analisar (exceto Lucratividade)
colunas = [col for col in df.columns if col != 'Lucratividade']

# Dicionário para contar outliers por coluna
outliers_count = {}

# Boxplot antes da remoção
plt.figure(figsize=(16, 4 * len(colunas)))
for i, col in enumerate(colunas, 1):
    plt.subplot(len(colunas), 2, 2*i - 1)
    sns.boxplot(x=df[col])
    plt.title(f'Antes da remoção - {col}')

# Aplicar remoção: marcar outliers como NaN usando IQR * 2
df_out = df.copy()
for col in colunas:
    Q1 = df_out[col].quantile(0.25)
    Q3 = df_out[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 2 * IQR
    limite_superior = Q3 + 2 * IQR
    
    # Contar outliers antes de substituir
    mask_outliers = (df_out[col] < limite_inferior) | (df_out[col] > limite_superior)
    outliers_count[col] = mask_outliers.sum()
    
    df_out.loc[mask_outliers, col] = np.nan

# Boxplot depois da remoção (com NaN no lugar)
for i, col in enumerate(colunas, 1):
    plt.subplot(len(colunas), 2, 2*i)
    sns.boxplot(x=df_out[col])
    plt.title(f'Depois da remoção - {col}')

plt.tight_layout()
plt.show()

# Remover linhas com NaN para o DataFrame final limpo
df_limpo = df_out.dropna()

# Exibir número de observações removidas por variável (outliers detectados)
print("Número de outliers removidos (por variável):")
for col, n in outliers_count.items():
    print(f"{col}: {n}")

# df_limpo é o DataFrame final sem outliers
dados_final_sem_outliers_BP = df_limpo

#%% BoxPlot customizado com IQR de 2:

def boxplot_iqr2_estetico(data, ax, label=None):
    data = data.dropna()
    if data.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 2 * IQR
    limite_superior = Q3 + 2 * IQR
    
    data_no_outliers = data[(data >= limite_inferior) & (data <= limite_superior)]
    mediana = np.median(data)
    
    pos = 1
    width = 0.4  # largura da caixa
    
    ax.set_facecolor('white')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Caixa (retângulo preenchido)
    rect = patches.FancyBboxPatch(
        (pos - width/2, Q1), width, Q3 - Q1,
        boxstyle="round,pad=0.02",
        linewidth=2, edgecolor='#4c72b0', facecolor='#a6bddb', alpha=0.7)
    ax.add_patch(rect)
    
    # Linha mediana mais grossa e vermelha
    ax.plot([pos - width/2, pos + width/2], [mediana, mediana], color='#e6550d', linewidth=3)
    
    if not data_no_outliers.empty:
        ax.plot([pos, pos], [data_no_outliers.min(), Q1], color='#4c72b0', linewidth=1.5, linestyle='-')
        ax.plot([pos, pos], [Q3, data_no_outliers.max()], color='#4c72b0', linewidth=1.5, linestyle='-')
        
        ax.plot([pos - width/4, pos + width/4], [data_no_outliers.min(), data_no_outliers.min()], color='#4c72b0', linewidth=1.5)
        ax.plot([pos - width/4, pos + width/4], [data_no_outliers.max(), data_no_outliers.max()], color='#4c72b0', linewidth=1.5)
    
    outliers = data[(data < limite_inferior) | (data > limite_superior)]
    ax.scatter(np.repeat(pos, len(outliers)), outliers.values, color='#fdae61', edgecolors='#d73027',
               alpha=0.8, s=60, zorder=3)
    
    ax.set_xticks([pos])
    if label:
        ax.set_xticklabels([label], fontsize=12)
    ax.set_xlim(pos - 1, pos + 1)
    
    ymin = min(data.min(), data_no_outliers.min() if not data_no_outliers.empty else data.min()) - IQR * 0.3
    ymax = max(data.max(), data_no_outliers.max() if not data_no_outliers.empty else data.max()) + IQR * 0.3
    ax.set_ylim(ymin, ymax)
    
    ax.set_title(label, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# ------------------------------------------------------

df = arquivo_final.copy()

colunas = [col for col in df.columns if col != 'Lucratividade']

outliers_count = {}

# Como cada variável tem 2 gráficos (antes e depois),
# e queremos 4 gráficos por linha, o número de colunas é 4
n_cols = 8
n_vars = len(colunas)
n_rows = math.ceil(n_vars * 2 / n_cols)  # total gráficos / 4 por linha

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows), constrained_layout=True)

# Flatten eixo para facilitar indexação quando n_rows=1
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes_flat = axes.flatten()

# Plot antes da remoção
for i, col in enumerate(colunas):
    idx_antes = 2 * i     # índice do gráfico "antes"
    boxplot_iqr2_estetico(df[col], axes_flat[idx_antes], label=f'Antes - {col}')

# Aplicar remoção de outliers e marcar NaN
df_out = df.copy()
for col in colunas:
    Q1 = df_out[col].quantile(0.25)
    Q3 = df_out[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 2 * IQR
    limite_superior = Q3 + 2 * IQR
    
    mask_outliers = (df_out[col] < limite_inferior) | (df_out[col] > limite_superior)
    outliers_count[col] = mask_outliers.sum()
    df_out.loc[mask_outliers, col] = np.nan

# Plot depois da remoção
for i, col in enumerate(colunas):
    idx_depois = 2 * i + 1  # índice do gráfico "depois"
    boxplot_iqr2_estetico(df_out[col], axes_flat[idx_depois], label=f'Depois - {col}')

# Desligar eixos extras se houver
for j in range(n_vars * 2, n_rows * n_cols):
    axes_flat[j].axis('off')

plt.show()

df_limpo = df_out.dropna()

print("Número de outliers removidos (por variável):")
for col, n in outliers_count.items():
    print(f"{col}: {n}")

dados_final_sem_outliers_BP = df_limpo

#%% Describe apos QC

# Ajusta a largura máxima para evitar quebra de linha
pd.set_option('display.width', 1000)         # aumenta largura total do display
pd.set_option('display.max_columns', None)   # exibe todas as colunas
pd.set_option('display.max_colwidth', 100)   # aumenta largura máxima de cada coluna

desc = dados_final_sem_outliers_BP.describe().T  # Transpor para facilitar
desc['median'] = dados_final_sem_outliers_BP.median()
desc = desc.T  # Voltar para o formato original (opcional)

print(desc)
#%% Matriz Correlacoes

# Função para calcular matriz de p-valores
def calculate_pvalues(df):
    df_numeric = df.select_dtypes(include=[np.number])
    pvals = pd.DataFrame(np.ones((df_numeric.shape[1], df_numeric.shape[1])),
                         columns=df_numeric.columns, index=df_numeric.columns)
    for col1 in df_numeric.columns:
        for col2 in df_numeric.columns:
            if col1 != col2:
                _, p = pearsonr(df_numeric[col1], df_numeric[col2])
                pvals.loc[col1, col2] = p
            else:
                pvals.loc[col1, col2] = 0  # p-valor de uma variável com ela mesma
    return pvals

# Calcula a matriz de correlação e de p-valores
correlation_matrix = dados_final_sem_outliers_BP.corr()
pval_matrix = calculate_pvalues(dados_final_sem_outliers_BP)

# Criar matriz de strings com r e p formatados
annot_matrix = correlation_matrix.copy().astype(str)
for i in range(annot_matrix.shape[0]):
    for j in range(annot_matrix.shape[1]):
        r = correlation_matrix.iloc[i, j]
        p = pval_matrix.iloc[i, j]
        annot_matrix.iloc[i, j] = f"{r:.2f}\n(p={p:.3f})" if i != j else f"{r:.2f}"

# Heatmap com r e p
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(correlation_matrix, annot=annot_matrix.values, fmt="",
                      cmap=plt.cm.viridis_r, vmin=-1, vmax=1,
                      annot_kws={'size': 11})
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=12, rotation=45)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=12, rotation=45)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
plt.title("Matriz de Correlação com p-valores", fontsize=14)
plt.tight_layout()
plt.show()


#%% Dados para as regressões

dados_final_sem_outliers_BP.columns
lucro = dados_final_sem_outliers_BP.dropna()

#%% Diagnóstico de multicolinearidade (Variance Inflation Factor e Tolerance)

# Calculando os valores de VIF
X1 = sm.add_constant(lucro[['AOL', 'CAR', 'EGG','PN', 'P240', \
                                    'P455', 'IMS']])
VIF = pd.DataFrame()
VIF["Variável"] = X1.columns[1:]
VIF["VIF"] = [variance_inflation_factor(X1.values, i+1)
              for i in range(X1.shape[1]-1)]

# Calculando as Tolerâncias
VIF["Tolerância"] = 1 / VIF["VIF"]
VIF


#%% Treinamento e teste

# Separação treino e teste (80/20)
X = lucro[['AOL', 'CAR', 'EGG', 'PN', 'P240', 'P455', 'IMS']]
y = lucro['Lucratividade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar DataFrames para treino e teste com as variáveis e resposta
dados_train = X_train.copy()
dados_train['Lucratividade'] = y_train

dados_test = X_test.copy()
dados_test['Lucratividade'] = y_test

#%% Modelo reggressao Linear Múltipla

modelo_linear = sm.OLS.from_formula('Lucratividade ~ AOL + CAR + EGG + PN + P240 + \
                                    P455 + IMS', dados_train).fit()
                                    
# Parâmetros do 'modelo_linear'
modelo_linear.summary()


#%% Procedimento Stepwise

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

# Estimação do modelo por meio do procedimento Stepwise (BoxPlot)
modelo_step = stepwise(modelo_linear, pvalue_limit=0.05)


#%% Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Francia (n >= 30)
# Carregamento da função 'shapiro_francia' do pacote 'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/


# Teste de Shapiro-Francia: interpretação
teste_sf = shapiro_francia(modelo_step.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')


#%% Histograma dos resíduos do 'modelo_step' com curva normal teórica para comparação das distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

# Ajusta os parâmetros da distribuição normal aos resíduos
mu, sigma = norm.fit(modelo_step.resid)

# Define os limites com base no intervalo dos resíduos, com uma margem extra
residuos = modelo_step.resid
x_min = residuos.min() - abs(residuos.std() * 0.5)
x_max = residuos.max() + abs(residuos.std() * 0.5)
x = np.linspace(x_min, x_max, 100)
p = norm.pdf(x, mu, sigma)

# Geração do gráfico
plt.figure(figsize=(12,6))
sns.histplot(residuos, bins=15, kde=True, stat="density",
             color='red', alpha=0.4)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Stepwise Linear', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.show()


#%% Função para o teste de Breusch-Pagan para a elaboração de diagnóstico de heterocedasticidade

# Criação da função 'breusch_pagan_test'

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value


#%% Teste de Breusch-Pagan 

breusch_pagan_test(modelo_step)
# Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

# H0 do teste: ausência de heterocedasticidade.
# H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de variável relevante!

# Interpretação
teste_bp = breusch_pagan_test(modelo_step) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')


#%% Adicionando fitted values e resíduos do 'modelo_step' no dataframe 'lucro'

lucro['fitted_step'] = modelo_step.fittedvalues
lucro['residuos_step'] = modelo_step.resid
lucro

#%% MSE Modelo OLS Linear Step

# Previsões
y_pred_train = modelo_step.predict(dados_train)
y_pred_test = modelo_step.predict(dados_test)

# MSE para treino e teste
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"MSE (treino): {mse_train:.2f}")
print(f"MSE (teste): {mse_test:.2f}")
#%% Modelo regressao Lienar Múltipla Transformação de Box-Cox
# A transformação Box-Cox só pode ser aplicada em dados positivos e estritamente maiores que zero.

# Como temos variáveis de lucro com valor negativo, precisamos somar um valor constante para tornar o menos valor positivo
# Passo 1: encontrar o menor valor atual
min_lucro = lucro['Lucratividade'].min()
# Passo 2: calcular a constante para tornar o menor valor positivo
constante = -min_lucro + 2  # ou use +0.01 se quiser só um valor levemente positivo
# Passo 3: aplicar a constante à coluna
lucro['lucro_acumulado_ajustado'] = lucro['Lucratividade'] + constante


# Para o cálculo do lambda de Box-Cox
# 'yast' é uma variável que traz os valores transformados (Y*)
# 'lmbda' é o lambda de Box-Cox
yast, lmbda = boxcox(lucro['lucro_acumulado_ajustado'])
#yast, lmbda = yeojohnson(lucro['lucro_acumulado_ajustado'])

print("Lambda: ",lmbda)

#Inserindo o lambda de Box-Cox no dataset para a estimação de um
#novo model
lucro['bc_lucro_acumulado'] = yast
lucro

#%% Estimação de um modelo OLS com transformação BoxCox

# Separar X e y transformado
X_bc = lucro[['AOL', 'CAR', 'EGG', 'PN', 'P240', 'P455', 'IMS']]
y_bc = lucro['bc_lucro_acumulado']
y_original = lucro['Lucratividade']

# Separação treino e teste (80/20)
X_train_bc, X_test_bc, y_train_bc, y_test_bc, y_train_original, y_test_original = train_test_split(
    X_bc, y_bc, y_original, test_size=0.2, random_state=42
)

# DataFrames de treino e teste
dados_train_bc = X_train_bc.copy()
dados_train_bc['bc_lucro_acumulado'] = y_train_bc

dados_test_bc = X_test_bc.copy()
dados_test_bc['bc_lucro_acumulado'] = y_test_bc

# Estimação do modelo
modelo_bc = sm.OLS.from_formula('bc_lucro_acumulado ~ AOL + CAR + EGG + PN + P240 + P455 + IMS', dados_train_bc).fit()

# Parâmetros do 'modelo_linear'
modelo_bc.summary()# Parâmetros do 'modelo_linear'

#%% Procedimento Stepwise no 'modelo_bc'

modelo_step_bc = stepwise(modelo_bc, pvalue_limit=0.05)

#%% Teste de verificação da aderência à normalidade dos resíduos do novo 'modelo_step_bc'

# Teste de Shapiro-Francia: interpretação
teste_sf = shapiro_francia(modelo_step_bc.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')
    

#%% Histograma dos resíduos do 'modelo_step_bc' com curva normal teórica para comparação das distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

# Ajusta os parâmetros da distribuição normal aos resíduos
mu, sigma = norm.fit(modelo_step_bc.resid)

# Define os limites com base no intervalo dos resíduos, com uma margem extra
residuos = modelo_step_bc.resid
x_min = residuos.min() - abs(residuos.std() * 0.5)
x_max = residuos.max() + abs(residuos.std() * 0.5)
x = np.linspace(x_min, x_max, 100)
p = norm.pdf(x, mu, sigma)

# Geração do gráfico
plt.figure(figsize=(12,6))
sns.histplot(residuos, bins=15, kde=True, stat="density",
             color='red', alpha=0.4)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Stepwise Transformação BoxCox', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.show()


#%% Teste de Breusch-Pagan

breusch_pagan_test(modelo_step_bc)

# Interpretação
teste_bp = breusch_pagan_test(modelo_step_bc) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')
    

#%% Adicionando fitted values e resíduos do 'modelo_step_bc' no dataframe 'lucro'

# Fitted: subtrair a constante para voltar à escala original
lucro['fitted_step_bc'] = modelo_step_bc.fittedvalues - constante
# Resíduos: mantêm-se como estão
lucro['residuos_step_bc'] = modelo_step_bc.resid

#%% MSE

# Definir lambda da transformação Box-Cox
lambda_bc = lmbda

# Previsões na escala transformada
y_pred_train_bc = modelo_step_bc.predict(dados_train_bc)
y_pred_test_bc = modelo_step_bc.predict(dados_test_bc)

# Reverter previsões para a escala original (ajuste com constante)
y_pred_train_original = inv_boxcox(y_pred_train_bc, lambda_bc) - constante
y_pred_test_original = inv_boxcox(y_pred_test_bc, lambda_bc) - constante

# Imprimir quantidade de NaNs para diagnóstico
print("NaNs em y_train_original:", np.sum(pd.isna(y_train_original)))
print("NaNs em y_pred_train_original:", np.sum(pd.isna(y_pred_train_original)))
print("NaNs em y_test_original:", np.sum(pd.isna(y_test_original)))
print("NaNs em y_pred_test_original:", np.sum(pd.isna(y_pred_test_original)))

print("Índices com NaN em y_train_original:", np.where(pd.isna(y_train_original))[0])
print("Índices com NaN em y_pred_train_original:", np.where(pd.isna(y_pred_train_original))[0])

# Criar máscaras para remover pares com NaNs
mask_train = (~pd.isna(y_train_original)) & (~pd.isna(y_pred_train_original))
mask_test = (~pd.isna(y_test_original)) & (~pd.isna(y_pred_test_original))

# Calcular MSE ignorando NaNs
mse_train_original = mean_squared_error(y_train_original[mask_train], y_pred_train_original[mask_train])
mse_test_original = mean_squared_error(y_test_original[mask_test], y_pred_test_original[mask_test])

# Exibir resultados
print(f"MSE Box-Cox revertido (treino): {mse_train_original:.4f}")
print(f"MSE Box-Cox revertido (teste): {mse_test_original:.4f}")


#%% Modelo Regressao Múltipla Transformacao Polinomial grau 2

# Separar treino e teste no DataFrame original 'lucro'
train_idx, test_idx = train_test_split(lucro.index, test_size=0.2, random_state=42)

# 2. Criar DataFrames de treino e teste
lucro_train = lucro.loc[train_idx]
lucro_test = lucro.loc[test_idx]

# Definir fórmula do modelo polinomial grau 2
formula_poli = '''Lucratividade ~ 
    AOL + I(AOL**2) +
    CAR + I(CAR**2) +
    EGG + I(EGG**2) +
    PN + I(PN**2) +
    P240 + I(P240**2) +
    P455 + I(P455**2) +
    IMS + I(IMS**2)

'''

# Criar matrizes de treino (y_train, X_train)
y_train, X_train = dmatrices(formula_poli, data=lucro_train, return_type='dataframe')

# 5. Ajustar o modelo OLS no treino
modelo_poli = sm.OLS(y_train, X_train).fit()
modelo_poli.summary()

#%% Procedimento Stepwise no 'modelo_poli'

# Realizando o procedimento stepwise
modelo_poli_step = stepwise(modelo_poli, pvalue_limit=0.05)
print(modelo_poli_step.summary())

#%% Teste de Shapiro-Francia para normalidade dos resíduos
teste_sf = shapiro_francia(modelo_poli_step.resid)  # Criação do objeto 'teste_sf'
teste_sf = teste_sf.items()  # Retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf  # Definindo os elementos da lista (tupla)

# Exibindo os resultados do teste
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))

# Interpretação do Teste de Normalidade
alpha = 0.05  # Nível de significância
if p[1] > alpha:
    print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
    print('Rejeita-se H0 - Distribuição não aderente à normalidade')
    
    
#%% Histograma dos resíduos do 'modelo_poli_step' com curva normal teórica para comparação das distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

# Ajusta os parâmetros da distribuição normal aos resíduos
mu, sigma = norm.fit(modelo_poli_step.resid)

# Define os limites com base no intervalo dos resíduos, com uma margem extra
residuos = modelo_poli_step.resid
x_min = residuos.min() - abs(residuos.std() * 0.5)
x_max = residuos.max() + abs(residuos.std() * 0.5)
x = np.linspace(x_min, x_max, 100)
p = norm.pdf(x, mu, sigma)

# Geração do gráfico
plt.figure(figsize=(12,6))
sns.histplot(residuos, bins=15, kde=True, stat="density",
             color='red', alpha=0.4)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Stepwise Linear Polinomial Grau 2', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.show()

#%% Teste de Breusch-Pagan

breusch_pagan_test(modelo_poli_step)
# Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

# H0 do teste: ausência de heterocedasticidade.
# H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de variável relevante!

# Interpretação
teste_bp = breusch_pagan_test(modelo_poli_step) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')
    
#%% Adicionando fitted values e resíduos do 'modelo_step_log' no dataframe 'lucro'

# Fitted: subtrair a constante para voltar à escala original
lucro['fitted_step_poli'] = modelo_poli_step.fittedvalues - constante
# Resíduos: mantêm-se como estão
lucro['residuos_step_poli'] = modelo_poli_step.resid

#%% MSE
# Criar matrizes de teste (y_test, X_test) — ATENÇÃO: deve usar a mesma fórmula!
y_test, X_test = dmatrices(formula_poli, data=lucro_test, return_type='dataframe')

# Previsões para treino e teste com o modelo stepwise (usar somente as colunas selecionadas pelo stepwise)
# Note que stepwise retorna um modelo statsmodels já ajustado, então podemos usar predict diretamente:
y_pred_train = modelo_poli_step.predict(X_train)
y_pred_test = modelo_poli_step.predict(X_test)

# Calcular MSE para treino e teste
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"MSE treino (polinomial grau 2): {mse_train:.2f}")
print(f"MSE teste (polinomial grau 2): {mse_test:.2f}")

#%% MULTINIVEL


#%% Arquivo 01 com ganho80

# Carregar os dados
dados_ganho80 = pd.read_excel('../Fenotipos_ajustados/lucro_whole.xls')
dados_ganho80.columns

arquivo_01 = dados_ganho80[["CGA", "ganho_80"]]
arquivo_01 = arquivo_01.rename(columns={"ganho_80": "GP80"})


#%% Arquivo 02

dados = pd.read_table("lucro_ims_frame_ag_elisa.txt", sep = " ")

#Selecionando as colunas
dados.columns

dados = dados.drop(['idade_80', 'idade_qd', 'lucro_80_', 'ano', 'NFA', 'sx', 'GCN240', \
                    'GCN365', 'GCN455', 'GCN550', 'C365', 'CIVP', 'GCims', 'ID_D0', 'idade', 'idade2', 'P365', 'P550', \
                        'gcus', 'GCPN', 'civp_pn'], axis = 1)

dados = dados.rename(columns={'cga': 'CGA', 'lucratividade': 'Lucratividade', "AOL_CM2": 'AOL', "EGP8_MM":"EGG", "ims": "IMS"})

dados.head()

#Criar colunas de Ganho de Peso 
#Esse código deve criar a coluna GP de forma eficiente e garantir que, se houver NaN nas colunas de entrada, o resultado também será NaN)

    # Ganho de peso a desama:
dados['GP_PRED'] = np.where(dados['PN'].isna() | dados['P240'].isna(), np.nan, 
                                     (dados['P240'] - dados['PN']) / 240)
    
    #Ganho peso pós desmama = peso ao ano - peso desmama / (365 - 240)
dados['GP_POSD'] = np.where(dados['P455'].isna() | dados['P240'].isna(), np.nan, 
                                     (dados['P455'] - dados['P240']) / (455-240))

#%% Arquivo CAR
dados_car = pd.read_excel('dados_ea.xlsx')
dados_car.columns

arquivo_02 = dados_car[["cga", "car"]]
arquivo_02 = arquivo_02.rename(columns={'cga': 'CGA', "car": "CAR"})

#%% Juntando arquivos

# Primeiro merge: dados com arquivo_01
temp = pd.merge(dados, arquivo_01, how='inner', on='CGA')
# Segundo merge: resultado anterior com arquivo_02
arquivo_final = pd.merge(temp, arquivo_02, how='inner', on='CGA')

arquivo_final = arquivo_final.drop(columns=['CGA'])
arquivo_final.columns

nova_ordem = [
    'gc2', 'Lucratividade', 'PN', 'P240', 'P455', 'GP80', 
    'AOL', 'EGG', 'CAR', 'IMS' , 'GP_PRED', 'GP_POSD'
]

# Reordena o DataFrame
arquivo_final = arquivo_final[nova_ordem]

arquivo_final.loc[:, arquivo_final.columns != 'CAR'] = arquivo_final.loc[:, arquivo_final.columns != 'CAR'].replace(0, pd.NA)

arquivo_final.info()

# Conversão de colunas 'object' para numérico, exceto 'gc2'
arquivo_final = arquivo_final.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.name != 'gc2' and col.dtype == 'object' else col)


arquivo_final = arquivo_final.drop(columns=['GP_PRED', 'GP_POSD']) #eu fiz as análises com GP, porém deu multicolinearidade, então estou removendo aqui
arquivo_final = arquivo_final.dropna()

arquivo_final = arquivo_final.drop(columns='GP80')

#arquivo_final.to_excel('Dados_brutos_final.xlsx', index=False)

arquivo_final = arquivo_final.dropna()

#%% Identificacao outliers (BoxPlot) e remocao


def remover_outliers(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 2 * IQR
    limite_superior = Q3 + 2 * IQR
    filtro = ((df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)) | (df[coluna].isna())
    return df[filtro]

# Cópia do DataFrame original
dados_final_sem_outliers_BP2 = arquivo_final.copy()

# Separar colunas numéricas, excluindo 'gc2'
colunas_numericas = dados_final_sem_outliers_BP2.select_dtypes(include='number').columns
colunas_para_filtrar = [col for col in colunas_numericas if col not in ['gc2', 'Lucratividade']]

# Contagem antes
contagem_antes = arquivo_final.notna().sum().to_frame(name='Antes')

# Ajustar tamanho da figura com base nas variáveis numéricas
n_var = len(colunas_para_filtrar)
plt.figure(figsize=(20, 3 * n_var))

# Remover outliers e fazer boxplots
for i, coluna in enumerate(colunas_para_filtrar, 1):
    dados_final_sem_outliers_BP2 = remover_outliers(dados_final_sem_outliers_BP2, coluna)

    # Boxplot original
    plt.subplot(n_var, 2, 2*i - 1)
    plt.boxplot(arquivo_final[coluna].dropna(), vert=False)
    plt.title(f'Boxplot Original ({coluna})')

    # Boxplot sem outliers
    plt.subplot(n_var, 2, 2*i)
    plt.boxplot(dados_final_sem_outliers_BP[coluna].dropna(), vert=False)
    plt.title(f'Sem Outliers ({coluna})')

plt.tight_layout()
plt.show()

# Contagem depois
contagem_depois = dados_final_sem_outliers_BP2.notna().sum().to_frame(name='Depois')
comparacao_outliers = contagem_antes.join(contagem_depois)
comparacao_outliers['Diferença'] = comparacao_outliers['Antes'] - comparacao_outliers['Depois']
print("\nComparativo de contagem de dados por variável antes e depois da remoção de outliers:")
print(comparacao_outliers)

# Estatísticas descritivas com % de NA
descricao = dados_final_sem_outliers_BP2.describe(include='all')
total_nan = dados_final_sem_outliers_BP2.isna().sum()
porcent_nan = (total_nan / len(dados_final_sem_outliers_BP2)) * 100
missing_df = pd.DataFrame({
    'missing_total': total_nan,
    'missing_percent': porcent_nan.round(2)
}).T
descricao_com_missing = pd.concat([descricao, missing_df])
print(descricao_com_missing)

#%% Filtrar Dataset com base no número de GC

# Contar a frequência de observações na variável 'gc2'
frequencia_gc2 = dados_final_sem_outliers_BP2['gc2'].value_counts()
# Exibir o resultado
print(frequencia_gc2)

# Identificar os grupos com apenas uma ocorrência
grupos_unicos = dados_final_sem_outliers_BP2['gc2'].value_counts()
grupos_para_remover = grupos_unicos[grupos_unicos <= 5].index

# Remover essas linhas do dataframe
df_filtrado = dados_final_sem_outliers_BP2[~dados_final_sem_outliers_BP2['gc2'].isin(grupos_para_remover)]

# Verificar o resultado
print(df_filtrado['gc2'].value_counts())
df_filtrado['gc2'].nunique()

#%% Multicolinearidade

# Calculando os valores de VIF
X1 = sm.add_constant(df_filtrado[['AOL', 'CAR', 'EGG','PN', 'P240', \
                                    'P455', 'IMS']])
VIF = pd.DataFrame()
VIF["Variável"] = X1.columns[1:]
VIF["VIF"] = [variance_inflation_factor(X1.values, i+1)
              for i in range(X1.shape[1]-1)]

# Calculando as Tolerâncias
VIF["Tolerância"] = 1 / VIF["VIF"]
VIF

#%% Funçoes auxiliares

def lrtest(model_base, model_comp):
    """
    Likelihood Ratio Test entre dois modelos (OLS ou MixedLM).
    """
    ll_base = model_base.llf
    ll_comp = model_comp.llf

    # Determinar número de parâmetros
    # MixedLM tem df_modelwc, OLS usa df_model + 1 (intercepto)
    def n_params(model):
        if hasattr(model, 'df_modelwc'):   # MixedLM
            return model.df_modelwc
        elif hasattr(model, 'df_model'):   # OLS
            return model.df_model + 1
        else:
            raise ValueError("Modelo não suportado para LR test")
    
    df_diff = n_params(model_comp) - n_params(model_base)
    LR_stat = -2 * (ll_base - ll_comp)
    p_val = stats.chi2.sf(LR_stat, df_diff)
    
    print("==========================================")
    print("Likelihood Ratio Test")
    print(f"-2*(LL_base - LL_comp): {LR_stat:.3f}")
    print(f"Grau(s) de liberdade: {df_diff}")
    print(f"P-valor: {p_val:.4f}")
    
    if p_val <= 0.05:
        print("Resultado: Rejeita H0 → modelo completo melhora significativamente o ajuste")
    else:
        print("Resultado: Não rejeita H0 → nenhum ganho significativo no ajuste")
    print("==========================================\n")

def efeito_aleatorio_significativo(modelo, idx_se_table=1):
    """
    Testa de forma aproximada a significância do efeito aleatório.
    """
    var_re = float(modelo.cov_re.iloc[0,0])
    se = float(pd.DataFrame(modelo.summary().tables[1]).iloc[idx_se_table,1])
    z = var_re / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    
    print(f"Estatística z efeito aleatório: {z:.3f}")
    print(f"P-valor aproximado: {p_val:.3f}")
    
    if p_val >= 0.05:
        print("Ausência de significância estatística do efeito aleatório (95% confiança)")
    else:
        print("Efeito aleatório significativo (95% confiança)")
    print("==========================================\n")
    return z, p_val


#%% Modelo nulo OLS para comparação

modelo_ols_nulo = sm.OLS.from_formula('Lucratividade ~ 1', data=df_filtrado).fit()
print(modelo_ols_nulo.summary())

# Teste de razão de verossimilhança

#%% Modelo nulo (MixedLM sem covariáveis)

modelo_nulo_hlm2_gc = sm.MixedLM.from_formula(
    'Lucratividade ~ 1',
    groups='gc2',
    re_formula='1',
    data=df_filtrado
).fit(reml=False)

print(modelo_nulo_hlm2_gc.summary())

# ICC
var_entre_gc = modelo_nulo_hlm2_gc.cov_re.iloc[0,0]
var_residual = modelo_nulo_hlm2_gc.scale
icc_gc = var_entre_gc / (var_entre_gc + var_residual)
print(f"ICC (proporção de variância entre GC2): {icc_gc:.3f}")

# Significância do efeito aleatório
efeito_aleatorio_significativo(modelo_nulo_hlm2_gc)

lrtest(modelo_ols_nulo, modelo_nulo_hlm2_gc)

#%% Modelo com interceptos aleatórios e covariáveis

# Separar grupos treino/teste
grupos = df_filtrado['gc2'].unique()
grupos_train, grupos_test = train_test_split(grupos, test_size=0.2, random_state=42)
df_train = df_filtrado[df_filtrado['gc2'].isin(grupos_train)].copy()
df_test  = df_filtrado[df_filtrado['gc2'].isin(grupos_test)].copy()

# Ajustar modelo multinível
modelo_intercept_hlm2_gc = sm.MixedLM.from_formula(
    'Lucratividade ~ AOL + CAR + EGG + PN + P240 + P455 + IMS',
    groups='gc2',
    re_formula='1',
    data=df_train
).fit(reml=False)

print(modelo_intercept_hlm2_gc.summary())

# Significância do efeito aleatório
efeito_aleatorio_significativo(modelo_intercept_hlm2_gc, idx_se_table=2)

# Teste de razão de verossimilhança vs modelo nulo
lrtest(modelo_nulo_hlm2_gc, modelo_intercept_hlm2_gc)

#%% Predição e avaliação de desempenho

# Atenção: grupos novos no teste não terão efeito aleatório
grupos_novos = set(df_test['gc2']) - set(df_train['gc2'])
if grupos_novos:
    print(f"Atenção: {len(grupos_novos)} grupo(s) novo(s) no teste, efeitos aleatórios serão ignorados para eles")

# Predição
y_pred_train = modelo_intercept_hlm2_gc.predict(df_train)
y_pred_test  = modelo_intercept_hlm2_gc.predict(df_test)

# MSE
mse_train = mean_squared_error(df_train['Lucratividade'], y_pred_train)
mse_test  = mean_squared_error(df_test['Lucratividade'], y_pred_test)

print(f"MSE treino: {mse_train:.2f}")
print(f"MSE teste: {mse_test:.2f}")
    


#%% Ajuste de Parâmetros no K-Means

# Determinando o número ideal de clusters usando apenas variáveis quantitativas padronizadas

# 1. Selecionar apenas variáveis numéricas (sem 'gc2' e 'Lucratividade')
colunas_padronizar = df_filtrado.drop(columns=['gc2', 'Lucratividade']).select_dtypes(include='number').columns

# 2. Padronizar
scaler = StandardScaler()
df_filtrado_padronizado = df_filtrado.copy()
df_filtrado_padronizado[colunas_padronizar] = scaler.fit_transform(df_filtrado[colunas_padronizar])

# 3. Extrair os dados padronizados para o KMeans
X_kmeans = df_filtrado_padronizado[colunas_padronizar].copy()

# 4. Calcular inércia para o gráfico do cotovelo
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_kmeans)
    inertia.append(kmeans.inertia_)

# 5. Plotar o método do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Método do Cotovelo")
plt.xlabel("Número de Clusters")
plt.ylabel("Inércia")
plt.grid(True)
plt.show()


#%% Adicionado Kmeans ao Dataset
#o gráfico da dobre do joelho sugere 3, mas eu vou colocar 5 senão nao converge no treino e teste

# 1. Aplicar o KMeans com 5 clusters nos dados numéricos padronizados
kmeans = KMeans(n_clusters=5, random_state=42)
df_filtrado['grupo_cluster'] = kmeans.fit_predict(X_kmeans).astype(str)

# 2. Mostrar as primeiras linhas para conferir
df_filtrado.head()


#%% Modelo nulo MixedLM por cluster

modelo_nulo_hlm2_cluster = sm.MixedLM.from_formula(
    'Lucratividade ~ 1',
    groups='grupo_cluster',
    re_formula='1',
    data=df_filtrado
).fit(reml=False)

print(modelo_nulo_hlm2_cluster.summary())

# ICC
var_entre_cluster = modelo_nulo_hlm2_cluster.cov_re.iloc[0, 0]
var_residual = modelo_nulo_hlm2_cluster.scale
icc_cluster = var_entre_cluster / (var_entre_cluster + var_residual)
print(f"ICC (proporção de variância entre clusters): {icc_cluster:.3f}")

#%% Comparação com OLS nulo e MixedLM GC2

# LR test para OLS nulo vs cluster nulo
lrtest(modelo_ols_nulo, modelo_nulo_hlm2_cluster)

# LR test para GC2 nulo vs cluster nulo
lrtest(modelo_nulo_hlm2_gc, modelo_nulo_hlm2_cluster)

#%% Modelo com interceptos aleatórios + covariáveis

# Separar clusters treino/teste
clusters = df_filtrado['grupo_cluster'].unique()
clusters_train, clusters_test = train_test_split(clusters, test_size=0.2, random_state=42)

df_train = df_filtrado[df_filtrado['grupo_cluster'].isin(clusters_train)].copy()
df_test  = df_filtrado[df_filtrado['grupo_cluster'].isin(clusters_test)].copy()

# Ajustar modelo
modelo_intercept_hlm2_cluster = sm.MixedLM.from_formula(
    'Lucratividade ~ AOL + CAR + EGG + PN + P240 + P455 + IMS',
    groups='grupo_cluster',
    re_formula='1',
    data=df_train
).fit(reml=False)

print(modelo_intercept_hlm2_cluster.summary())

# LR test: modelo nulo cluster vs modelo completo cluster
lrtest(modelo_nulo_hlm2_cluster, modelo_intercept_hlm2_cluster)

#%% Predição e avaliação de desempenho

grupos_novos = set(df_test['grupo_cluster']) - set(df_train['grupo_cluster'])
if grupos_novos:
    print(f"Atenção: {len(grupos_novos)} clusters novos no teste → efeitos aleatórios ignorados para eles")

# Predição
y_pred_train = modelo_intercept_hlm2_cluster.predict(df_train)
y_pred_test  = modelo_intercept_hlm2_cluster.predict(df_test)

# MSE
mse_train = mean_squared_error(df_train['Lucratividade'], y_pred_train)
mse_test  = mean_squared_error(df_test['Lucratividade'], y_pred_test)

print(f"MSE treino (MixedLM - cluster): {mse_train:.2f}")
print(f"MSE teste (MixedLM - cluster): {mse_test:.2f}")


#%% Gráfico para comparação visual dos logLiks dos modelos estimados até o momento (Multiniveis)

# Criando o DataFrame com os logLiks dos modelos
df_llf = pd.DataFrame({
    'modelo': ['OLS Nulo', 'HLM2 Nulo GC',
               'HLM2 com Int. Aleat. GC', 'HLM2 Nulo Cluster', 'HLM2 com Int. Aleat. Cluster'],
    'loglik': [modelo_ols_nulo.llf, modelo_nulo_hlm2_gc.llf, 
               modelo_intercept_hlm2_gc.llf, modelo_nulo_hlm2_cluster.llf, modelo_intercept_hlm2_cluster.llf]
})

# Ordenar o DataFrame com base nos valores de loglik (ordem crescente)
df_llf = df_llf.sort_values(by='loglik')

# Criando o gráfico de barras horizontais
fig, ax = plt.subplots(figsize=(15, 10))

# Definindo as cores das barras
c = ['dimgray', 'darkslategray', 'indigo']

# Plotando as barras
ax1 = ax.barh(df_llf.modelo, df_llf.loglik, color=c)

# Adicionando os rótulos ao centro das barras
ax.bar_label(ax1, label_type='center', color='white', fontsize=20)

# Ajustando os rótulos dos eixos
ax.set_ylabel("Modelo Proposto", fontsize=24)
ax.set_xlabel("LogLik", fontsize=24)

# Ajustando o tamanho da fonte dos ticks
ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)

# Exibindo o gráfico
plt.show()


#%% REGRESSAO LINEAR X MULTINIVEL (AIC e BIC) 

# Função para extrair AIC e BIC dos modelos de regressão linear
def comparar_modelos(modelos_dict):
    resultados = []
    for nome, modelo in modelos_dict.items():
        aic = modelo.aic
        bic = modelo.bic
        resultados.append({'Modelo': nome, 'AIC': aic, 'BIC': bic})
    return pd.DataFrame(resultados).sort_values(by="AIC")

# Lista dos modelos de regressão linear ajustados
modelos_regressao = {
    'modelo_step': modelo_step,
    'modelo_step_bc': modelo_step_bc,
    'modelo_poli_step': modelo_poli_step
}

# Comparar AIC e BIC para os modelos de regressão linear
resultados_regressao = comparar_modelos(modelos_regressao)

# Função para calcular AIC e BIC para modelos multinível
def calcular_aic_bic(log_likelihood, n_params, n_observations):
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + np.log(n_observations) * n_params
    return aic, bic

# Lista de modelos multinível
modelos = {
    'modelo_ols_nulo': modelo_ols_nulo,
    'modelo_nulo_hlm2_gc': modelo_nulo_hlm2_gc,
    'modelo_intercept_hlm2_gc': modelo_intercept_hlm2_gc,
    'modelo_nulo_hlm2_cluster': modelo_nulo_hlm2_cluster,
    'modelo_intercept_hlm2_cluster': modelo_intercept_hlm2_cluster
}

# Número de observações
n_observations = len(df_filtrado)

# Número de parâmetros para cada modelo multinível
n_params = {
    'modelo_ols_nulo': 1,  # 1 parâmetro (intercepto)
    'modelo_nulo_hlm2_gc': 2,  # 1 intercepto + 1 para a variância do efeito aleatório
    'modelo_intercept_hlm2_gc': 10,  # 8 coef. fixos + 1 intercepto + 1 para a variância do efeito aleatório
    'modelo_intercept_hlm2_gc_step': 8,  # 6 coef. fixos + 1 intercepto + 1 para a variância do efeito aleatório
    'modelo_nulo_hlm2_cluster': 2,  # 1 intercepto + 1 para a variância do efeito aleatório
    'modelo_intercept_hlm2_cluster': 10  # 8 coef. fixos + 1 intercepto + 1 para a variância do efeito aleatório
}

# Dicionários para armazenar AIC e BIC dos modelos multinível
aic_values = {}
bic_values = {}

# Cálculo de AIC e BIC para cada modelo multinível
for model_name, model in modelos.items():
    log_likelihood = model.llf  # Extrai automaticamente o log-likelihood
    params = n_params[model_name]
    
    # Calcular AIC e BIC usando a função
    aic, bic = calcular_aic_bic(log_likelihood, params, n_observations)
    
    # Armazenar os resultados
    aic_values[model_name] = aic
    bic_values[model_name] = bic

# Preparar os dados para a tabela dos modelos multinível
dados_modelos_multinivel = []
for model_name in modelos.keys():
    dados_modelos_multinivel.append({
        'Modelo': model_name,
        'AIC': aic_values.get(model_name, 'NaN'),
        'BIC': bic_values.get(model_name, 'NaN')
    })

# Criar DataFrame para modelos multinível
df_modelos_multinivel = pd.DataFrame(dados_modelos_multinivel)

# Juntar as duas tabelas em uma tabela final
tabela_final = pd.concat([resultados_regressao, df_modelos_multinivel], ignore_index=True)


# Ordenar a tabela pelo AIC e, em caso de empate, pelo BIC
tabela_final_sorted = tabela_final.sort_values(by=['AIC', 'BIC'], ascending=[True, True])

# Exibir a tabela final ordenada
print("Tabela Final Ordenada com AIC e BIC dos Modelos:")
print(tabela_final_sorted.to_string(index=False))



#%% REDES NEURAIS

#%% Ajustando os dados de entrada
dados_final_sem_outliers_BP.columns


# Separar features e variável alvo
X = dados_final_sem_outliers_BP.drop(['Lucratividade'], axis=1)
y = dados_final_sem_outliers_BP['Lucratividade']

# Dividir em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Validação cruzada Tunning Fino 

# Etapa 1 – Tuning de Alpha e Learning Rate (Arquitetura fixa)

# Parâmetros para tuning fino
param_dist_1 = {
    'alpha': uniform(1e-5, 0.1),  # valores contínuos de 1e-5 até 0.1
    'learning_rate_init': uniform(1e-4, 0.01),  # de 1e-4 até ~0.01
}

# Modelo base
mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                   activation='relu',
                   solver='adam',
                   learning_rate='adaptive',
                   early_stopping=True,
                   max_iter=3000)

# Random Search
random_search_1 = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_dist_1,
    n_iter=30,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

random_search_1.fit(X_train_scaled, y_train)

print("Melhor configuração (Etapa 1):", random_search_1.best_params_)
print("Melhor R² validação cruzada:", random_search_1.best_score_)


# Etapa 2 – Tuning de Arquitetura (com alpha otimizado da Etapa 1)

# Insira os melhores valores da Etapa 1 aqui:
best_alpha = random_search_1.best_params_['alpha']
best_lr_init = random_search_1.best_params_['learning_rate_init']

# Arquiteturas a testar
param_dist_2 = {
    'hidden_layer_sizes': [
        (32,), (64,), (128,), 
        (64, 32), (128, 64), 
        (128, 64, 32), #(256, 128, 64),
        (100, 50), (100, 50, 25)
    ]
}

# Novo modelo com alpha e learning rate fixos
mlp_tuned = MLPRegressor(
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    alpha=best_alpha,
    learning_rate_init=best_lr_init,
    early_stopping=True,
    max_iter=3000
)

random_search_2 = RandomizedSearchCV(
    estimator=mlp_tuned,
    param_distributions=param_dist_2,
    n_iter=8,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

random_search_2.fit(X_train_scaled, y_train)

print("Melhor arquitetura (Etapa 2):", random_search_2.best_params_)
print("Melhor R² validação cruzada:", random_search_2.best_score_)

#%% Ajuste do modelo com os melhores parâmetros encontrados 

nn_tuned = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25), 
    activation='relu', 
    solver='adam',
    alpha=0.09,  # Melhor valor de alpha
    learning_rate_init=0.009,  # Melhor valor de learning rate
    learning_rate='adaptive',
    early_stopping=True,
    max_iter=1000,
    random_state=42
)

# Treinando o modelo
nn_tuned.fit(X_train_scaled, y_train)

# Predições no conjunto de teste
nn_predictions = nn_tuned.predict(X_test_scaled)

#%% Gráficos de perda e MSE

# Curvas de perda (loss) ao longo das épocas
loss_curve = nn_tuned.loss_curve_

# Plotando as curvas de perda e MSE
plt.figure(figsize=(12, 6))

# Curva de Perda
plt.subplot(1, 2, 1)
plt.plot(loss_curve, label='Loss', color='blue')
plt.title('Curva de Perda (Loss) durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda (Loss)')
plt.grid(True)

# Curva de MSE
train_mse = mean_squared_error(y_train, nn_tuned.predict(X_train_scaled))
test_mse = mean_squared_error(y_test, nn_predictions)

# Apenas um valor de MSE, sem iteração
plt.subplot(1, 2, 2)
plt.bar(['MSE Treino', 'MSE Teste'], [train_mse, test_mse], color=['red', 'green'])
plt.title('MSE para Treinamento e Teste')
plt.ylabel('MSE')
plt.grid(True)

plt.tight_layout()
plt.show()

#%% Avaliação do Modelo Final

# Avaliar o modelo no conjunto de teste para verificar sua generalização
final_mse = mean_squared_error(y_test, nn_predictions)

# Exibindo a avaliação final
print("\n--- Avaliação do Overfitting/Subfitting na Rede Neural ---")
print(f"MSE Treino: {train_mse:.4f} | MSE Teste: {test_mse:.4f} | Diferença: {abs(train_mse - test_mse):.4f}")


#%% Comparar redes neurais com  regressao e random forest

# Regressão Linear
lm = LinearRegression()
lm.fit(X_train_scaled, y_train)
lm_predictions = lm.predict(X_test_scaled)
lm_mse = mean_squared_error(y_test, lm_predictions)


# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)


#%% Resultados dos três métodos

print("Regressão Linear - MSE:", lm_mse)
print("Random Forest - MSE:", mean_squared_error(y_test, rf_preds))  # rf_preds são as predições do Random Forest
print("Rede Neural - MSE:", mean_squared_error(y_test, nn_predictions))  # nn_predictions são as predições da Rede Neural

# Calcular R² para os três modelos
print("R² - Regressão Linear:", r2_score(y_test, lm_predictions))  # lm_predictions são as predições da Regressão Linear
print("R² - Random Forest:", r2_score(y_test, rf_preds))  # rf_preds são as predições do Random Forest
print("R² - Rede Neural:", r2_score(y_test, nn_predictions))  # nn_predictions são as predições da Rede Neural
