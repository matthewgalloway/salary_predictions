import config.config as config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

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
	df[var] = np.log(df[var].replace(0, np.nan))
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


def plot_continuous(df, var)-> None:
	"""Plots a continous variable in a boxplot,
	apply logging to skewed variables
	against the income variable"""

	df = df.copy()
	df['Income'] = df['Income'].astype('category')
	var_name = var
	if var in config.SKEWED_NUMERIC_VARS:
		df[var] = np.log(df[var].replace(0, np.nan))
		var_name = 'Log of ' + var_name
	ax = sns.boxplot(x='Income', y=var, data=df)
	ax.set(ylabel=var_name, title=f"Effect of {var} on Income")
	plt.show()
