import config.config as config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from scipy.stats import f_oneway

def get_meta_columns() -> list:
    """loads the columns from the meta data files
        returns the fields as list"""

    columns_list = []

    with open(config.META_DIR, "r") as metadata_file:
        meta_data = metadata_file.readlines()
        meta_data = meta_data[81:121]

        line_count = 0
        for line in meta_data:
            columns_list.append(line[line.find("(") + 1:line.find(")")])
            if line_count == 23:
                columns_list.append('instance weight')
            line_count += 1

        columns_list.append('Income')

    return columns_list


def plot_categoricial(df, var) -> None:
    """Plots the categorical variables
    in the dataset provided"""
    temp = df[['Income', var]].groupby(var).mean().reset_index()
    axis = sns.barplot(x=var, y='Income', data=temp)
    axis.set(ylabel="Probability of earning over 50K")
    axis.set_xticklabels(config.vis_dict[var], rotation=75)
    plt.title(f"Effect of {var} on Income")
    plt.show()


def plot_log(df, var) -> None:
    """Plots the log of a numeric variable
    in the dataset provided"""

    df = df.copy()
    df[var] = np.log(df[var] + 1)
    sns.displot(df, x=var)


def plot_discrete(df, var) -> None:
    """Plots a discrete variable in a barchart
    against the income variable"""

    temp = df[['Income', var]].groupby(var).mean().reset_index()
    axis = sns.barplot(x=var, y='Income', data=temp)
    axis.set(ylabel="Probability of earning over 50K")
    plt.title(f"Effect of {var} on Income")
    if var == 'weeks worked in year':
        plt.title(f"Effect of Months worked in year on Income")
        axis.set(xlabel="Months worked in year")
    plt.show()


def plot_continuous(df, var) -> None:
    """Plots a continous variable in a boxplot,
    apply logging to skewed variables
    against the income variable"""

    df = df.copy()
    df['Income'] = df['Income'].astype('category')
    var_name = var
    if var in config.SKEWED_NUMERIC_VARS:
        df[var] = df[var]
        var_name = 'Log of ' + var_name
    ax = sns.boxplot(x='Income', y=var, data=df)
    ax.set(ylabel=var_name, title=f"Effect of {var} on Income")
    plt.show()


def plot_temp(df, var):
    temp = df[['Income', var]].groupby(var).mean().reset_index()
    axis = sns.barplot(x=var, y='Income', data=temp)
    axis.set(ylabel="Probability of earning over 50K")
    axis.set_xticklabels(labels=axis.get_xticklabels(), rotation=75)
    plt.show()


def get_feature_importance(model, top_features=10):
    column_list = config.FEATURES.copy()
    dropped_cols = model.named_steps['DropCorrelated'].columns_to_drop.copy()
    dropped_cols.append('instance weight')
    for i in range(len(dropped_cols)):
        column_list.remove(dropped_cols[i])
    importance = pd.DataFrame(
        data=model.named_steps['rf'].feature_importances_,
        columns=['weight']
    )
    importance['variable'] = column_list
    return importance.sort_values(by='weight', ascending=False)[0:top_features]


def plot_occupation(df, var):
    temp = df[['Income', var]].groupby(var).mean().reset_index()
    axis = sns.barplot(x='Income', y=var, data=temp)
    axis.set(xlabel="Probability of earning over 50K")
    axis.set(ylabel=var, title=f"Effect of {var} on Income")
    # axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
    plt.show()


def plot_skewed(df, var):
    df[var] = np.where(df[var] > 0, 1, 0)
    temp = df[['Income', var]].groupby(var).mean().reset_index()
    axis = sns.barplot(x=var, y='Income', data=temp)
    axis.set(ylabel="Probability of earning over 50K")
    plt.title(f"Effect of {var} on Income")
    plt.show()


def get_anova_relationships(df, variable_list, target, threshold=0.05):

    """Gets the anova relationship between a list of variable and the
    target variable
    Returns: variables correlated with targets below threshold
    """
    selected_predictors = []
    for variable in variable_list:
        CategoryGroupLists = df.groupby('Income')[variable].apply(list)
        anova_results = f_oneway(*CategoryGroupLists)
        # If the ANOVA P-Value is <0.05, that means we reject H0
        if (anova_results[1] < threshold):
            print(variable, 'is correlated with', target, '| P-Value:', anova_results[1])
            selected_predictors.append(variable)
        else:
            print(variable, 'is NOT correlated with', target, '| P-Value:', anova_results[1])
    return selected_predictors
