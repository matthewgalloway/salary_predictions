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
            "CategoricalEncoder",
            preprocessors.CategoricalEncoder(variables=config.CATEGORICAL_VALS[1:]),
        ),
         (
            "EducationEncoder",
            preprocessors.EducationEncoder(variables=config.CATEGORICAL_VALS[0]),
        ),
        (
            "SkewedNumericLogger",
            preprocessors.NumericLogger(variables=config.SKEWED_NUMERIC_VARS),
        ),
        (
            "MinMaxScalar",
            preprocessors.Min_Max_Scalar(variables=config.DISCRETE_NUMERIC_VARS+config.CONTINUOUS_NUMERIC_VARS),
        ),
        (
            "CategoricalMinMaxScalar",
            preprocessors.Min_Max_Scalar(variables=config.CATEGORICAL_VALS),
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
            "CategoricalEncoder",
            preprocessors.CategoricalEncoder(variables=config.CATEGORICAL_VALS[1:]),
        ),
         (
            "EducationEncoder",
            preprocessors.EducationEncoder(variables=config.CATEGORICAL_VALS[0]),
        ),
        (
            "SkewedNumericLogger",
            preprocessors.NumericLogger(variables=config.SKEWED_NUMERIC_VARS),
        ),
        (
            "MinMaxScalar",
            preprocessors.Min_Max_Scalar(variables=config.DISCRETE_NUMERIC_VARS+config.CONTINUOUS_NUMERIC_VARS),
        ),
        (
            "CategoricalMinMaxScalar",
            preprocessors.Min_Max_Scalar(variables=config.CATEGORICAL_VALS),
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
            "CategoricalEncoder",
            preprocessors.CategoricalEncoder(variables=config.CATEGORICAL_VALS[1:]),
        ),
         (
            "EducationEncoder",
            preprocessors.EducationEncoder(variables=config.CATEGORICAL_VALS[0]),
        ),
        (
            "SkewedNumericLogger",
            preprocessors.NumericLogger(variables=config.SKEWED_NUMERIC_VARS),
        ),
        (
            "MinMaxScalar",
            preprocessors.Min_Max_Scalar(variables=config.DISCRETE_NUMERIC_VARS+config.CONTINUOUS_NUMERIC_VARS),
        ),
        (
            "CategoricalMinMaxScalar",
            preprocessors.Min_Max_Scalar(variables=config.CATEGORICAL_VALS),
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
            "CategoricalEncoder",
            preprocessors.CategoricalEncoder(variables=config.CATEGORICAL_VALS[1:]),
        ),
         (
            "EducationEncoder",
            preprocessors.EducationEncoder(variables=config.CATEGORICAL_VALS[0]),
        ),
        (
            "SkewedNumericLogger",
            preprocessors.NumericLogger(variables=config.SKEWED_NUMERIC_VARS),
        ),
        (
            "MinMaxScalar",
            preprocessors.Min_Max_Scalar(variables=config.DISCRETE_NUMERIC_VARS+config.CONTINUOUS_NUMERIC_VARS),
        ),
        (
            "CategoricalMinMaxScalar",
            preprocessors.Min_Max_Scalar(variables=config.CATEGORICAL_VALS),
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
            "FillNAencoder",
            preprocessors.FillNAEncoder(variables=config.CATEGORICAL_VALS),
        ),
        (
            "EducationEncoder",
            preprocessors.EducationEncoder(variables='education'),
        ),
        (
            "CategoricalMinMaxScalar",
            preprocessors.CategoricalEncoder(variables=config.CATEGORICAL_VALS[1:]),
        ),
    ])
