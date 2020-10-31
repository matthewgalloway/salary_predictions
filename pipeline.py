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
            preprocessors.OrdinalEncoder(variables=config.CATEGORICAL_VALS[0]),
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
