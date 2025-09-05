import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('medical_examination.csv')

bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

def draw_cat_plot():
    cat_order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=cat_order
    )

    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

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

    g.set_axis_labels("variable", "total")
    g.set_titles("cardio = {col_name}")
    g.despine(left=True)
    g.set_xticklabels(rotation=90)

    fig = g.fig

    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    df_heat = df.copy()

    df_heat = df_heat[df_heat['ap_lo'] <= df_heat['ap_hi']]

    h_low, h_high = df['height'].quantile([0.025, 0.975])
    w_low, w_high = df['weight'].quantile([0.025, 0.975])

    df_heat = df_heat[
        (df_heat['height'] >= h_low) & (df_heat['height'] <= h_high) &
        (df_heat['weight'] >= w_low) & (df_heat['weight'] <= w_high)
    ]

    corr = df_heat.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 12))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".1f",
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    fig.savefig('heatmap.png')
    return fig
