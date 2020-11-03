# File paths

TRAIN_DIR = "data/census_income_learn.csv"

TEST_DIR = "data/census_income_test.csv"

META_DIR = "data/census_income_metadata.txt"

THRESHOLD = 0.6

# Features

TARGET = "Income"

TARGET_ENCODING = {
	" - 50000.": 0,
	" 50000+.": 1
}

FEATURES = [
	'age',
	'class of worker',
	'detailed industry recode',
	'detailed occupation recode',
	'education',
	'wage per hour',
	'enroll in edu inst last wk',
	'marital stat',
	'major industry code',
	'major occupation code',
	'race',
	'hispanic origin',
	'sex',
	'member of a labor union',
	'reason for unemployment',
	'full or part time employment stat',
	'capital gains',
	'capital losses',
	'dividends from stocks',
	'tax filer stat',
	'region of previous residence',
	'state of previous residence',
	'detailed household and family stat',
	'detailed household summary in household',
	'instance weight',
	'migration code-change in msa',
	'migration code-change in reg',
	'migration code-move within reg',
	'live in this house 1 year ago',
	'migration prev res in sunbelt',
	'num persons worked for employer',
	'family members under 18',
	'country of birth father',
	'country of birth mother',
	'country of birth self',
	'citizenship',
	'own business or self employed',
	"fill inc questionnaire for veteran's admin",
	'veterans benefits',
	'weeks worked in year',
	'year'
]

NUMERIC_VALS = [
	'age',
	'detailed industry recode',
	'detailed occupation recode',
	'wage per hour',
	'capital gains',
	'capital losses',
	'dividends from stocks',
	'instance weight',
	'num persons worked for employer',
	'own business or self employed',
	'veterans benefits',
	'weeks worked in year',
	'year'
]

DUPLICATE_VALS = [
	'enroll in edu inst last wk',  # info contained in 'education'
	# 'class of worker',  # info contained in 'detailed industry recode'
	'major occupation code',  # info contained in 'detailed industry recode'
	'major industry code',  # info contained in 'detailed occupation recode'
	'reason for unemployment',  # info contained in 'full or part time employment stat',
	'country of birth self',  # info contained in 'citizenship',
	'country of birth father',  # info contained in 'citizenship',
	'country of birth mother',  # info contained in 'citizenship',
	'region of previous residence',  # info contained in ''state of previous residence'	',
	'family members under 18',
	'detailed household and family stat',  # info contained in detailed household summary in household'
	"fill inc questionnaire for veteran's admin",  # info contained 'veterans benefits',
	'migration prev res in sunbelt',  # info contained in of 'migration code-move within reg'
	'migration code-change in msa',  # info contained in of 'migration code-move within reg'
	'migration code-change in reg',  # info contained in of 'migration code-move within reg'
]

CATEGORICAL_VALS = [
	'education',
	'sex',
	'race',
	'marital stat',
	'hispanic origin',
	'detailed household summary in household',
	'detailed household and family stat',
	'citizenship',
	'live in this house 1 year ago',
	'state of previous residence',
	'migration code-move within reg',
	'member of a labor union',
	'full or part time employment stat',
	'tax filer stat',
	'enroll in edu inst last wk',  # info contained in 'education'
	'class of worker',  # info contained in 'detailed industry recode'
	'major occupation code',  # info contained in 'detailed industry recode'
	'major industry code',  # info contained in 'detailed occupation recode'
	'reason for unemployment',  # info contained in 'full or part time employment stat',
	'family members under 18',
	'country of birth self',  # info contained in 'citizenship',
	'country of birth father',  # info contained in 'citizenship',
	'country of birth mother',  # info contained in 'citizenship',
	"fill inc questionnaire for veteran's admin",  # info contained 'veterans benefits',
	'migration prev res in sunbelt',  # info contained in of 'migration code-move within reg'
	'region of previous residence',  # info contained in ''state of previous residence'	',
	'migration code-change in msa',  # info contained in of 'migration code-move within reg'
	'migration code-change in reg',  # info contained in of 'migration code-move within reg'
]

VIS_CATEGORICAL_VALS = [
	'education',
	'sex',
	'race',
	'marital stat',
	'hispanic origin',
	'detailed household summary in household',
	'citizenship',
	'live in this house 1 year ago',
	'state of previous residence',
	'migration code-move within reg',
	'member of a labor union',
	'full or part time employment stat',
	'tax filer stat',
	'class of worker',
]

education_dict = {
	" Children": 0,
	" Less than 1st grade": 0,
	" 1st 2nd 3rd or 4th grade": 0,
	" 5th or 6th grade": 0,
	" 7th and 8th grade": 0,
	" 9th grade": 0,
	" 10th grade": 0,
	" 11th grade": 0,
	" 12th grade no diploma": 0,
	" High school graduate": 1,
	" Some college but no degree": 2,
	" Associates degree-occup /vocational": 3,
	" Associates degree-academic program": 3,
	" Bachelors degree(BA AB BS)": 4,
	" Masters degree(MA MS MEng MEd MSW MBA)": 5,
	" Doctorate degree(PhD EdD)": 6,
	" Prof school degree (MD DDS DVM LLB JD)": 7
}

vis_education_dict = {
	"Less than High School": 0,
	"High School": 1,
	"College": 2,
	"Associates degree": 3,
	'University Degree': 4,
	'Masters': 5,
	"Phd": 6,
	"Doctorate": 7
}

CONTAINS_NAN_VALS = [
	'country of birth self',
	'country of birth father',
	'country of birth mother'
]

vis_dict = {
	'race': [
		' Amer Indian Aleut or Eskimo',
		' Other',
		' Black',
		' White',
		' Asian or Pacific Islander'
	],
	'education': [
		"Less than High School",
		"High School",
		"College",
		"Associates degree",
		'University Degree',
		'Masters',
		"Phd",
		"Doctorate"
	],
	'citizenship': [
		' Foreign born- U S citizen by naturalization',
		' Native- Born in Puerto Rico or U S Outlying',
		' Foreign born- Not a citizen of U S ',
		' Native- Born abroad of American Parent(s)',
		' Native- Born in the United States',
	],
	'sex': [
		'Female',
		'Male'
	],
	'marital stat': [
		' Never married',
		' Married-A F spouse present',
		' Widowed',
		' Separated',
		' Married-spouse absent',
		' Divorced',
		' Married-civilian spouse present'
	],
	'detailed household summary in household':
		[
			' Child under 18 never married',
			' Child under 18 ever married',
			' Child 18 or older',
			' Other relative of householder',
			' Group Quarters- Secondary individual',
			' Nonrelative of householder',
			'Spouse of householder',
			' Householder'
		],
	"live in this house 1 year ago":
		[
			' No',
			' Yes',
			' Not in universe under 1 year old',

		],
	'full or part time employment stat':
		[
			' Not in labor force',
			' Unemployed part- time',
			' Unemployed full-time',
			' Children or Armed Forces',
			' PT for econ reasons usually PT',
			' PT for econ reasons usually PT',
			' PT for non-econ reasons usually FT',
			' Full-time schedules'
		],
	'class of worker': [
		' Never worked',
		' Without pay',
		' Missing',
		' Private',
		' Local government',
		' State government',
		' Self-employ-not inc',
		' Fed gov',
		' Self-employed-inc'
	]
}

SKEWED_NUMERIC_VARS = [
	'wage per hour',
	'capital gains',
	'capital losses',
	'dividends from stocks',
	'instance weight',
]

DISCRETE_NUMERIC_VARS = [
	'detailed industry recode',
	'detailed occupation recode',
	'num persons worked for employer',
	'own business or self employed',
	'veterans benefits',
	'weeks worked in year',
	'year'
]

DISCRETE_NOT_PLOTTED = [
	'detailed industry recode',
	'detailed occupation recode',
]

CATEGORICAL_VALS_NOT_PLOTTED = [
	'hispanic origin',
	'tax filer stat',
	'state of previous residence',
	'state of previous residence',
	'migration code-move within reg',
	'member of a labor union'
]

CONTINUOUS_NUMERIC_VARS = [
	'age',
	'wage per hour',
	'capital gains',
	'capital losses',
	'dividends from stocks',
	'instance weight',
]
