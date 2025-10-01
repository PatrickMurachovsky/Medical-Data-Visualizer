# Importa as bibliotecas necessárias
import pandas as pd               # Para manipulação de dados (DataFrame, CSV, etc.)
import seaborn as sns             # Para visualização de dados estatísticos
import matplotlib.pyplot as plt   # Para gráficos
import numpy as np                # Para cálculos numéricos e matrizes

# Carrega o dataset a partir de um arquivo CSV
df = pd.read_csv('medical_examination.csv')

# Calcula o IMC (peso / altura^2) e cria coluna binária indicando sobrepeso (1) ou não (0)
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# Normaliza colesterol e glicose:
# No dataset: 1 = normal, 2 ou 3 = acima do normal.
# Após normalização: 0 = normal, 1 = acima do normal
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Função para gerar o gráfico categórico (catplot)
def draw_cat_plot():
    # Define a ordem das variáveis que aparecerão no gráfico
    cat_order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    
    # "Derrete" o DataFrame para transformar colunas em linhas (long format)
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],       # Mantém a coluna 'cardio' como identificador
        value_vars=cat_order      # Colunas que serão transformadas em variáveis
    )

    # Conta quantas vezes cada valor aparece por variável e por condição de 'cardio'
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')  # Renomeia a contagem para 'total'
    )

    # Cria o gráfico categórico de barras comparando variáveis entre cardio=0 e cardio=1
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        order=cat_order,
        hue_order=[0, 1]
    )

    # Ajusta rótulos e títulos
    g.set_axis_labels("variable", "total")
    g.set_titles("cardio = {col_name}")
    g.despine(left=True)  # Remove a linha vertical esquerda
    g.set_xticklabels(rotation=90)  # Rotaciona os nomes no eixo X para melhor leitura

    fig = g.fig  # Pega a figura final do gráfico

    # Salva o gráfico como arquivo PNG
    fig.savefig('catplot.png')
    return fig

# Função para gerar o mapa de calor (heatmap)
def draw_heat_map():
    # Cria cópia do dataset para não modificar o original
    df_heat = df.copy()

    # Remove registros inválidos onde pressão diastólica > pressão sistólica
    df_heat = df_heat[df_heat['ap_lo'] <= df_heat['ap_hi']]

    # Remove outliers (altura e peso fora dos percentis 2.5% e 97.5%)
    h_low, h_high = df['height'].quantile([0.025, 0.975])
    w_low, w_high = df['weight'].quantile([0.025, 0.975])

    df_heat = df_heat[
        (df_heat['height'] >= h_low) & (df_heat['height'] <= h_high) &
        (df_heat['weight'] >= w_low) & (df_heat['weight'] <= w_high)
    ]

    # Calcula a matriz de correlação entre as variáveis numéricas
    corr = df_heat.corr()

    # Cria máscara para esconder a parte superior da matriz (triângulo superior)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Cria figura para o heatmap
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plota o heatmap com correlações
    sns.heatmap(
        corr,
        annot=True,         # Mostra os valores dentro dos quadrados
        fmt=".1f",          # Uma casa decimal
        mask=mask,          # Aplica a máscara para não repetir valores
        square=True,        # Quadrados proporcionais
        linewidths=0.5,     # Linhas entre os quadrados
        cbar_kws={"shrink": 0.5},  # Ajusta a barra de cor
        ax=ax
    )

    # Salva o heatmap como arquivo PNG
    fig.savefig('heatmap.png')
    return fig
