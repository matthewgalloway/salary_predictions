import config.config as config
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class EncodeNotInUniverse(BaseEstimator, TransformerMixin):
	"""Encodes Not in universe as NA values"""

	def __init__(self, variables=None):
		self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X = X.copy()
		X = X.replace(' Not in universe', np.NaN)

		return X


class DropDuplicates(BaseEstimator, TransformerMixin):
	"""drops duplicate values"""

	def __init__(self, variables=None):
		self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
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
	"""fills na values with the string "missing"
	"""

	def __init__(self, variables=None):
		self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X[self.variables] = X[self.variables].fillna(value='Missing', axis=1)

		return X


class WeeksToMonths(BaseEstimator, TransformerMixin):
	"""turns weeks to month"
	"""

	def __init__(self, variables=None):
		self.variables = variables
		self.months_dict = {}
		for i in range(53):
			self.months_dict[i] = divmod(i, 4)[0]
		self.months_dict[52] = 12

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X[self.variables] = X[self.variables].map(self.months_dict)
		return X


class EducationEncoder(BaseEstimator, TransformerMixin):
	"""Encodes education using domain knowledge to
	 reduce number of variables"""

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
	"""applys min_max scalar"""

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


class Standard_Scalar(BaseEstimator, TransformerMixin):
	"""applys standard scalar"""

	def __init__(self, variables=None):
		self.variables = variables
		self.scalar = StandardScaler()

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X[self.variables] = self.scalar.fit_transform(X[self.variables])

		return X


class VisEducationEncoder(BaseEstimator, TransformerMixin):
	"""feature engineering for education"""

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
	"""logs numeric values"""

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

class Skewed2Cat(BaseEstimator, TransformerMixin):
	"""logs numeric values"""

	def __init__(self, variables=None):
		self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()

		for variable in self.variables:
			X[variable] = np.where(X[variable]>0,1,0)

		return X


class DropCorrelated(BaseEstimator, TransformerMixin):
	"""drops correlated values above a threshold provided"""

	def __init__(self, threshold=1):
		self.threshold = threshold
		self.columns_to_drop = None

	def fit(self, X, y=None):
		data = pd.concat([X, y], axis=1)
		correlation_matrix = data.corr().abs()
		corr_selection = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
		self.columns_to_drop = [column for column in corr_selection.columns if
								any(corr_selection[column] > self.threshold)]
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X = X.drop(self.columns_to_drop, axis=1)
		return X


class ScaleNumeric(BaseEstimator, TransformerMixin):
	"""applys scalar to numeric values"""

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


class DropInstanceWeight(BaseEstimator, TransformerMixin):
	"""drops the variable instance weight"""

	def __init__(self, variables=None):
		self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# encode labels
		X = X.copy()
		X = X.drop(self.variables, axis=1)
		return X


class RareEncoder(BaseEstimator, TransformerMixin):
	"""Converts categorical variables to numerical
		values are ordered by the target variable."""

	def __init__(self, variables=None, threshold=0.01):
		self.variables = variables
		self.frequent_lables = {}
		self.threshold = threshold

	def fit(self, X, y):

		for var in self.variables:
			# the encoder will learn the most frequent categories
			temp = pd.Series(X[var].value_counts() / np.float(len(X)))
			# frequent labels:
			self.frequent_lables[var] = list(temp[temp >= self.threshold].index)
		return self

	def transform(self, X):
		X = X.copy()

		for feature in self.variables:
			X[feature] = np.where(X[feature].isin(self.frequent_lables[feature]), X[feature], 'Rare')

		return X
