import config.config as config
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class EncodeNotInUniverse(BaseEstimator, TransformerMixin):
	def __init__(self, variables=None):
		self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X = X.replace(' Not in universe', np.NaN)

		return X


class DropDuplicates(BaseEstimator, TransformerMixin):
	def __init__(self, variables=None):
		self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X = X.drop(self.variables, axis=1)

		return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
	"""Converts categorical variables to numerical
		values are ordered by the target variable."""

	def __init__(self, variables=None):
		self.variables = variables
		self.orderd_labels_dict = {}

	def fit(self, X, y):
		data = pd.concat([X, y], axis=1)
		data.columns = list(X.columns) + ["Income"]

		for var in self.variables:
			ordered_labels = data.groupby([var])["Income"].mean().sort_values().index
			self.orderd_labels_dict[var] = {value: index for index, value in enumerate(ordered_labels, 0)}
		return self

	def transform(self, X):
		X = X.copy()
		for feature in self.variables:
			X[feature] = X[feature].map(self.orderd_labels_dict[feature])

		return X

	def encode_categorical(df, variable):
		ordered_labels = df.groupby([variable])['Income'].mean().sort_values().index
		orderd_labels_dict = {value: index for index, value in enumerate(ordered_labels)}
		return df[variable].map(orderd_labels_dict)


class FillNAEncoder(BaseEstimator, TransformerMixin):
	def __init__(self, variables=None):
		self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X[self.variables] = X[self.variables].fillna(value='Missing', axis=1)

		return X


class ScaleNumeric(BaseEstimator, TransformerMixin):
	def __init__(self, variables=None):
		self.variables = variables
		self.scalar = MinMaxScaler()

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X[self.variables] = self.scalar.fit_transform(X[self.variables])

		return X


class EducationEncoder(BaseEstimator, TransformerMixin):
	def __init__(self, variables=None):
		self.variables = variables
		self.scalar = MinMaxScaler()

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X[self.variables] = X[self.variables].map(config.education_dict)

		return X


class Min_Max_Scalar(BaseEstimator, TransformerMixin):
	def __init__(self, variables=None):
		self.variables = variables
		self.scalar = MinMaxScaler()

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X[self.variables] = self.scalar.fit_transform(X[self.variables])

		return X


class VisEducationEncoder(BaseEstimator, TransformerMixin):
	def __init__(self, variables=None):
		self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X[self.variables] = X[self.variables].map(config.vis_education_dict)

		return X

class NumericLogger(BaseEstimator, TransformerMixin):
	def __init__(self, variables=None):
		self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		for variable in self.variables:
			X[variable] = np.log(X[variable].replace(0, np.nan))

		X[self.variables] = X[self.variables].fillna(value=0)

		return X