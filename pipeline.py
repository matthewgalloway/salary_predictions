import preprocessors
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImPipeline
import config.config as config
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

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
            "DropInstanceWeight",
            preprocessors.DropInstanceWeight(variables='instance weight'),
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
            "DropInstanceWeight",
            preprocessors.DropInstanceWeight(variables='instance weight'),
        ),
        (
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        ),
        (
            'lr', LogisticRegression(max_iter=1000)
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
            LogisticRegression(max_iter=1000)
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
            "DropInstanceWeight",
            preprocessors.DropInstanceWeight(variables='instance weight'),
        ),
        (
            "DropCorrelated",
            preprocessors.DropCorrelated(threshold=config.THRESHOLD),
        )

        # (
        #     'lr', GaussianNB()
        #  ),
    ]
        )

# def create_model(optimizer='adam', dropout=0.2):
#     input_layer = Input(shape=(X.shape[1],))
#     dense_layer_1 = Dense(15, activation='relu')(input_layer)
#     dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
#     output = Dense(y.shape[1], activation='softmax')(dense_layer_2)
#
#     model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
#
#     return model