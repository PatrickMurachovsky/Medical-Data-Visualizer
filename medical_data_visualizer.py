# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('medical_examination.csv')

# Calculate BMI and determine if the person is overweight (1) or not (0)
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# Normalize cholesterol and glucose values:
# 0 means normal (1 in dataset), 1 means above normal (2 or 3 in dataset)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Function to draw the categorical plot
def draw_cat_plot():
    # Define the order of categories to be shown
    cat_order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    
    # Reshape the DataFrame so we can group by variable and value
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],       # Keep the 'cardio' column
        value_vars=cat_order      # Melt the other columns
    )

    # Group by cardio, variable and value to count occurrences
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')  # Rename the count column to 'total'
    )

    # Create the categorical bar plot
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

    # Customize labels and title
    g.set_axis_labels("variable", "total")
    g.set_titles("cardio = {col_name}")
    g.despine(left=True)  # Remove left spine
    g.set_xticklabels(rotation=90)

    fig = g.fig

    # Save the plot to a file
    fig.savefig('catplot.png')
    return fig

# Function to draw the heat map
def draw_heat_map():
    # Create a copy of the dataset to filter
    df_heat = df.copy()

    # Remove entries where diastolic pressure > systolic pressure
    df_heat = df_heat[df_heat['ap_lo'] <= df_heat['ap_hi']]

    # Filter out outliers in height and weight using 2.5th and 97.5th percentiles
    h_low, h_high = df['height'].quantile([0.025, 0.975])
    w_low, w_high = df['weight'].quantile([0.025, 0.975])

    df_heat = df_heat[
        (df_heat['height'] >= h_low) & (df_heat['height'] <= h_high) &
        (df_heat['weight'] >= w_low) & (df_heat['weight'] <= w_high)
    ]

    # Compute correlation matrix
    corr = df_heat.corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create the heat map figure
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

    # Save the heat map to a file
    fig.savefig('heatmap.png')
    return fig
