import preprocessors
from sklearn.pipeline import Pipeline
import config.config as config

income_pipeline = Pipeline(
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
            "Numeric_variable_scalar",
            preprocessors.ScaleNumeric(variables=config.NUMERIC_VALS),
        ),
        (
            "categorical_variable_scalar",
            preprocessors.ScaleCategoric(variables=config.CATEGORICAL_VALS),
        ),
    ]
        )


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
