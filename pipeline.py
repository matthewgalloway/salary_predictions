import preprocessors
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImPipeline
import config.config as config
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


rf_pipeline = Pipeline(
    [
        (
            "EncodeNotInUniverse",
            preprocessors.EncodeNotInUniverse(variables=config.FEATURES),
        ),
        (
            "DropNaFeatures",
            preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        ),
        (
            "Fill_NA_encoder",
            preprocessors.FillNAEncoder(variables=config.CATEGORICAL_VALS),
        ),
        (
            "categorical_encoder",
            preprocessors.CategoricalEncoder(variables=config.CATEGORICAL_VALS[1:]),
        ),
         (
            "Ordinal Encoder",
            preprocessors.EducationEncoder(variables=config.CATEGORICAL_VALS[0]),
        ),
        (
            "Skewed_Numeric_Logger",
            preprocessors.NumericLogger(variables=config.SKEWED_NUMERIC_VARS),
        ),
        (
            "Ordinal_Numeric_scalar",
            preprocessors.OrdinalEncoder(variables=config.DISCRETE_NUMERIC_VARS+config.CONTINUOUS_NUMERIC_VARS),
        ),
        (
            "categorical_variable_scalar",
            preprocessors.OrdinalEncoder(variables=config.CATEGORICAL_VALS),
        ),
        (
            'rf', RandomForestClassifier(random_state=0)
        ),
    ]
        )

sm_rf_pipeline = ImPipeline(
    [
        (
            "EncodeNotInUniverse",
            preprocessors.EncodeNotInUniverse(variables=config.FEATURES),
        ),
        (
            "DropNaFeatures",
            preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        ),
        (
            "Fill_NA_encoder",
            preprocessors.FillNAEncoder(variables=config.CATEGORICAL_VALS),
        ),
        (
            "categorical_encoder",
            preprocessors.CategoricalEncoder(variables=config.CATEGORICAL_VALS[1:]),
        ),
         (
            "Ordinal Encoder",
            preprocessors.EducationEncoder(variables=config.CATEGORICAL_VALS[0]),
        ),
        (
            "Skewed_Numeric_Logger",
            preprocessors.NumericLogger(variables=config.SKEWED_NUMERIC_VARS),
        ),
        (
            "Ordinal_Numeric_scalar",
            preprocessors.OrdinalEncoder(variables=config.DISCRETE_NUMERIC_VARS+config.CONTINUOUS_NUMERIC_VARS),
        ),
        (
            "categorical_variable_scalar",
            preprocessors.OrdinalEncoder(variables=config.CATEGORICAL_VALS),
        ),
        (
            'smote', SMOTE(random_state=0)
        ),

        (
            'rf', RandomForestClassifier(random_state=0)
        ),
    ]
        )
lr_pipeline = Pipeline(
    [
        (
            "EncodeNotInUniverse",
            preprocessors.EncodeNotInUniverse(variables=config.FEATURES),
        ),
        (
            "DropNaFeatures",
            preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        ),
        (
            "Fill_NA_encoder",
            preprocessors.FillNAEncoder(variables=config.CATEGORICAL_VALS),
        ),
        (
            "categorical_encoder",
            preprocessors.CategoricalEncoder(variables=config.CATEGORICAL_VALS[1:]),
        ),
         (
            "Ordinal Encoder",
            preprocessors.EducationEncoder(variables=config.CATEGORICAL_VALS[0]),
        ),
        (
            "Skewed_Numeric_Logger",
            preprocessors.NumericLogger(variables=config.SKEWED_NUMERIC_VARS),
        ),
        (
            "Ordinal_Numeric_scalar",
            preprocessors.OrdinalEncoder(variables=config.DISCRETE_NUMERIC_VARS+config.CONTINUOUS_NUMERIC_VARS),
        ),
        (
            "categorical_variable_scalar",
            preprocessors.OrdinalEncoder(variables=config.CATEGORICAL_VALS),
        ),
        (
            'lr', LogisticRegression(random_state=0)
         ),
    ]
        )

sm_lr_pipeline = ImPipeline(
    [
        (
            "EncodeNotInUniverse",
            preprocessors.EncodeNotInUniverse(variables=config.FEATURES),
        ),
        (
            "DropNaFeatures",
            preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        ),
        (
            "Fill_NA_encoder",
            preprocessors.FillNAEncoder(variables=config.CATEGORICAL_VALS),
        ),
        (
            "categorical_encoder",
            preprocessors.CategoricalEncoder(variables=config.CATEGORICAL_VALS[1:]),
        ),
         (
            "Ordinal Encoder",
            preprocessors.EducationEncoder(variables=config.CATEGORICAL_VALS[0]),
        ),
        (
            "Skewed_Numeric_Logger",
            preprocessors.NumericLogger(variables=config.SKEWED_NUMERIC_VARS),
        ),
        (
            "Ordinal_Numeric_scalar",
            preprocessors.OrdinalEncoder(variables=config.DISCRETE_NUMERIC_VARS+config.CONTINUOUS_NUMERIC_VARS),
        ),
        (
            "categorical_variable_scalar",
            preprocessors.OrdinalEncoder(variables=config.CATEGORICAL_VALS),
        ),
        (
            'smote', SMOTE(random_state=0)
        ),
        (
            'lr', LogisticRegression(random_state=0)
         ),
    ]
        )

visualisation_pipeline = Pipeline(
    [
        (
            "EncodeNotInUniverse",
            preprocessors.EncodeNotInUniverse(variables=config.FEATURES),
        ),
        (
            "DropNaFeatures",
            preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        ),
        (
            "Fill_NA_encoder",
            preprocessors.FillNAEncoder(variables=config.CATEGORICAL_VALS),
        ),
        (
            "Education Encoder",
            preprocessors.EducationEncoder(variables='education'),
        ),
        (
            "categorical_encoder",
            preprocessors.CategoricalEncoder(variables=config.CATEGORICAL_VALS[1:]),
        ),
    ])

# #%% Modelling
#
#
#
# X_train_processed =income_pipeline.fit_transform(
#                                                     X_train[config.FEATURES], y_train)
#
# X_test_processed = income_pipeline.fit_transform(
#                                                     X_test[config.FEATURES], y_test)
#
# #%% Modeling
#
# corr_matric = X_train_processed.corr()
#
# corr_matric[(1>corr_matric)&(corr_matric>0.6)]
#
# #%% Modeling
#
# model = RandomForestClassifier(random_state=0)
# model.fit(X_train_processed, y_train)
# predicted_classes = model.predict(X_test_processed)
# print(accuracy_score(predicted_classes, y_test.values))
# print(confusion_matrix(predicted_classes, y_test.values))
# print(f1_score(predicted_classes, y_test.values))
