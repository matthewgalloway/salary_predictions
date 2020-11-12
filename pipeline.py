import preprocessors
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImPipeline
import config.config as config
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

processed_pipeline = Pipeline(
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
            "RareEncoder",
            preprocessors.RareEncoder(variables=config.CATEGORICAL_VALS[1:], threshold=0.01),
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
            "Skewed2Cat",
            preprocessors.Skewed2Cat(variables=config.SKEWED_NUMERIC_VARS),
        ),
        (
            "MinMaxScalar",
            preprocessors.Min_Max_Scalar(variables=config.DISCRETE_NUMERIC_VARS+config.CONTINUOUS_NUMERIC_VARS),
        ),
        (
            "CategoricalMinMaxScalar",
            preprocessors.Min_Max_Scalar(variables=config.CATEGORICAL_VALS),
        ),
        # (
        #     "DropInstanceWeight",
        #     preprocessors.DropInstanceWeight(variables='instance weight'),
        # # ),
        # (
        #     "DropCorrelated",
        #     preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        # ),
        # (
        #     'rf', RandomForestClassifier(random_state=0)
        # ),
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
            "RareEncoder",
            preprocessors.RareEncoder(variables=config.CATEGORICAL_VALS[1:], threshold=0.01),
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
            "Skewed2Cat",
            preprocessors.Skewed2Cat(variables=config.SKEWED_NUMERIC_VARS),
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
            "DropInstanceWeight",
            preprocessors.DropInstanceWeight(variables='instance weight'),
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
            "RareEncoder",
            preprocessors.RareEncoder(variables=config.CATEGORICAL_VALS[1:], threshold=0.01),
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
            "Skewed2Cat",
            preprocessors.Skewed2Cat(variables=config.SKEWED_NUMERIC_VARS),
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
            "DropInstanceWeight",
            preprocessors.DropInstanceWeight(variables='instance weight', ),
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
best_sm_rf_pipeline = ImPipeline(
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
            "RareEncoder",
            preprocessors.RareEncoder(variables=config.CATEGORICAL_VALS[1:], threshold=0.01),
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
            "Skewed2Cat",
            preprocessors.Skewed2Cat(variables=config.SKEWED_NUMERIC_VARS),
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
            "DropInstanceWeight",
            preprocessors.DropInstanceWeight(variables='instance weight', ),
        ),
        (
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        ),
        (
            'smote', SMOTE(random_state=0)
        ),

        (
            'rf', RandomForestClassifier(max_depth=90, min_samples_leaf=4,
                                        n_estimators=50, random_state=0)
        ),
    ]
        )
rus_rf_pipeline = ImPipeline(
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
            "RareEncoder",
            preprocessors.RareEncoder(variables=config.CATEGORICAL_VALS[1:], threshold=0.01),
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
            "Skewed2Cat",
            preprocessors.Skewed2Cat(variables=config.SKEWED_NUMERIC_VARS),
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
            "DropInstanceWeight",
            preprocessors.DropInstanceWeight(variables='instance weight', ),
        ),
        (
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        ),
        (
            'smote', RandomUnderSampler(random_state=0)
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
        # (
        #     "DropNaFeatures",
        #     preprocessors.DropDuplicates(variables=config.DUPLICATE_VALS),
        # ),
        (
            "Fill_NA_encoder",
            preprocessors.FillNAEncoder(variables=config.CATEGORICAL_VALS),
        ),
        (
            "RareEncoder",
            preprocessors.RareEncoder(variables=config.CATEGORICAL_VALS[1:], threshold=0.01),
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
            "Skewed2Cat",
            preprocessors.Skewed2Cat(variables=config.SKEWED_NUMERIC_VARS),
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
            "DropInstanceWeight",
            preprocessors.DropInstanceWeight(variables='instance weight'),
        ),
        (
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        ),
        (
            'lr', LogisticRegression(class_weight="balanced", max_iter=1000)
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
            "RareEncoder",
            preprocessors.RareEncoder(variables=config.CATEGORICAL_VALS[1:], threshold=0.01),
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
            "Skewed2Cat",
            preprocessors.Skewed2Cat(variables=config.SKEWED_NUMERIC_VARS),
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
            "DropInstanceWeight",
            preprocessors.DropInstanceWeight(variables='instance weight'),
        ),
        (
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        ),

        (
            'smote', SMOTE(random_state=0)
        ),
        (
            'lr',
            LogisticRegression(class_weight="balanced", max_iter=1000)
         ),
    ]
        )

rus_lr_pipeline = ImPipeline(
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
            "RareEncoder",
            preprocessors.RareEncoder(variables=config.CATEGORICAL_VALS[1:], threshold=0.01),
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
            "Skewed2Cat",
            preprocessors.Skewed2Cat(variables=config.SKEWED_NUMERIC_VARS),
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
            "DropInstanceWeight",
            preprocessors.DropInstanceWeight(variables='instance weight'),
        ),
        (
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        ),

        (
            'RUS', RandomUnderSampler(random_state=0)
        ),
        (
            'lr',
            LogisticRegression(class_weight="balanced", max_iter=1000)
         ),
    ]
        )

visualisation_pipeline = Pipeline(
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
            "RareEncoder",
            preprocessors.RareEncoder(variables=config.RARE_VALS, threshold=0.01),
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
            "WeeksToMonths",
            preprocessors.WeeksToMonths(variables='weeks worked in year'),
        ),

        (
            "CategoricalMinMaxScalar",
            preprocessors.CategoricalEncoder(variables=config.VIS_CATEGORICAL_VALS[1:]),
        ),
    ])

ml_data_pipeline = ImPipeline(
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
            "RareEncoder",
            preprocessors.RareEncoder(variables=config.CATEGORICAL_VALS[1:], threshold=0.01),
        ),
        # (
        #     "CategoricalEncoder",
        #     preprocessors.CategoricalEncoder(variables=config.CATEGORICAL_VALS[1:]),
        # ),
         (
            "EducationEncoder",
            preprocessors.EducationEncoder(variables=config.CATEGORICAL_VALS[0]),
        ),
        # (
        #     "SkewedNumericLogger",
        #     preprocessors.NumericLogger(variables=config.SKEWED_NUMERIC_VARS),
        # ),
        # (
        #     "MinMaxScalar",
        #     preprocessors.Min_Max_Scalar(variables=config.DISCRETE_NUMERIC_VARS+config.CONTINUOUS_NUMERIC_VARS),
        # ),
        # (
        #     "CategoricalMinMaxScalar",
        #     preprocessors.Min_Max_Scalar(variables=config.CATEGORICAL_VALS),
        # ),
        # (
        #     "DropInstanceWeight",
        #     preprocessors.DropInstanceWeight(variables='instance weight'),
        # ),
        # (
        #     "DropCorrelated",
        #     preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        # )

        # (
        #     'lr', GaussianNB()
        #  ),
    ]
        )

