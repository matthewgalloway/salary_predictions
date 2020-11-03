import preprocessors
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImPipeline
import config.config as config
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier

mlp_pipeline = Pipeline(
    [
        (
            "EncodeNotInUniverse",
            preprocessors.EncodeNotInUniverse(variables=config.FEATURES),
        ),
        # (
        #     "DropNaFeatures",
        #     preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        # ),
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
            preprocessors.EducationEncoder(variables='education'),
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
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        ),
        (
            'MLP', MLPClassifier(hidden_layer_sizes=(100, 2), random_state=1, solver='adam')
        ),
    ]
        )

rf_pipeline = Pipeline(
    [
        (
            "EncodeNotInUniverse",
            preprocessors.EncodeNotInUniverse(variables=config.FEATURES),
        ),
        # (
        #     "DropNaFeatures",
        #     preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        # ),
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
            preprocessors.EducationEncoder(variables='education'),
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
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
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
        # (
        #     "DropNaFeatures",
        #     preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        # ),
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
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        ),
        (
            'smote', SMOTE(random_state=0)
        ),

        (
            'rf', RandomForestClassifier(random_state=0)
        ),
    ]
        )

sm_gb_pipeline = ImPipeline(
    [
        (
            "EncodeNotInUniverse",
            preprocessors.EncodeNotInUniverse(variables=config.FEATURES),
        ),
        # (
        #     "DropNaFeatures",
        #     preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        # ),
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
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        ),
        (
            'smote', SMOTE(random_state=0)
        ),

        (
            'rf', GradientBoostingClassifier(random_state=0)
        ),
    ]
        )

lr_pipeline = Pipeline(
    [
        (
            "EncodeNotInUniverse",
            preprocessors.EncodeNotInUniverse(variables=config.FEATURES),
        ),
        # (
        #     "DropNaFeatures",
        #     preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        # ),
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
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
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
        # (
        #     "DropNaFeatures",
        #     preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        # ),
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
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
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
            preprocessors.FillNAEncoder(variables=config.VIS_CATEGORICAL_VALS),
        ),
        (
            "EducationEncoder",
            preprocessors.EducationEncoder(variables='education'),
        ),

        (
            "CategoricalMinMaxScalar",
            preprocessors.CategoricalEncoder(variables=config.VIS_CATEGORICAL_VALS[1:]),
        ),
    ])
