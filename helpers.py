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


def plot_categoricial(df, var):
	temp = df[['Income', var]].groupby(var).mean().reset_index()
	axis = sns.barplot(x=var, y='Income', data=temp)
	axis.set(ylabel="Probability of earning over 50K")
	axis.set_xticklabels(config.vis_dict[var], rotation=75)
	plt.title(f"Effect of {var} on Income")
	plt.show()


def plot_log(df, var):
	df = df.copy()
	df[var] = np.log(df[var].replace(0, np.nan))
	sns.displot(df, x=var)


def plot_discrete(df, var):
	temp = df[['Income', var]].groupby(var).mean().reset_index()
	axis = sns.barplot(x=var, y='Income', data=temp)
	axis.set(ylabel="Probability of earning over 50K")
	plt.title(f"Effect of {var} on Income")
	plt.show()


def plot_continuous(df, var):
	df = df.copy()
	df['Income'] = df['Income'].astype('category')
	if var in config.SKEWED_NUMERIC_VARS:
		df[var] = np.log(df[var].replace(0, np.nan))
	sns.boxplot(x='Income', y=var, data=df)
	plt.title(f"Effect of {var} on Income")
	plt.show()


def train_models(models, X_train, X_test, y_train, y_test):
	results = {
		'Model': [],
		'Accuracy': [],
		'F1_Score': [],
		'Recall_Score': []
	}
	# models = [rf_pipeline, sm_rf_pipeline, lr_pipeline, sm_lr_pipeline]

	for key in models:
		models[key].fit(X_train, y_train)
		y_pred = models[key].predict(X_test)
		results['Model'].append(key)
		results['Accuracy'].append(accuracy_score(y_test, y_pred))
		results['F1_Score'].append(f1_score(y_test, y_pred))
		results['Recall_Score'].append(recall_score(y_test, y_pred))

	print(pd.DataFrame(results))
	return models

# def plot_distributions():
#     for var in config.NUMERIC_VALS:
#         sns.displot(data, x=var)

# def get_meta_dict() -> dict:
# 	"""loads the meta data full list
# 		returns the fields as dictionary"""
#
# 	meta_dict = {}
#
# 	with open(config.META_DIR, "r") as metadata_file:
# 		meta_data = metadata_file.readlines()
# 		meta_data = meta_data[23:68]
#
# 		for i in meta_data:
# 			line = i.split('\t')
# 			meta_dict[line[-1].split('\n')[0]] = line[0][2:].rstrip()
