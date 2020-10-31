# filepaths

TRAIN_DIR = "data/census_income_learn.csv"

TEST_DIR = "data/census_income_test.csv"

META_DIR = "data/census_income_metadata.txt"

TARGET = "Income"

TARGET_ENCODING = {" - 50000.": 0, " 50000+.": 1}

FEATURES = ['age',
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


NUMERIC_VALS = ['age',
				'detailed industry recode',
				'detailed occupation recode',
				'wage per hour', 'capital gains',
				'capital losses',
				'dividends from stocks',
				'instance weight',
				'num persons worked for employer',
				'own business or self employed',
				'veterans benefits',
				'weeks worked in year',
				'year']

DROP_VALS = ['enroll in edu inst last wk',
			 'class of worker',
			 'major occupation code',
			 'member of a labor union',
			 'reason for unemployment',
			 'region of previous residence',
			 'state of previous residence',
			 'family members under 18',
			 "fill inc questionnaire for veteran's admin",
			 'migration prev res in sunbelt',
			 'migration code-change in msa',
			 'migration code-change in reg',
			 'migration code-move within reg',
			 ]

CATEGORICAL_VALS = ['education',
					'marital stat',
					'major industry code',
					'race',
					'hispanic origin',
					'sex',
					'full or part time employment stat',
					'tax filer stat',
					'detailed household and family stat',
					'detailed household summary in household',
					'live in this house 1 year ago',
					'citizenship',
					'country of birth self',
					 'country of birth father',
					 'country of birth mother'
					]

education_dict = {' Children': 0,
				  " Less than 1st grade": 0,
				  " 1st 2nd 3rd or 4th grade": 0,
				  " 5th or 6th grade": 0,
				  " 7th and 8th grade": 0,
				  " 9th grade": 0,
				  " 10th grade": 0,
				  " 11th grade": 0,
				  " 12th grade no diploma": 0,
				  " High school graduate": 1,
				  " Associates degree-occup /vocational": 2,
				  " Associates degree-academic program": 2,
				  " Some college but no degree": 3,
				  " Bachelors degree(BA AB BS)": 4,
				  " Masters degree(MA MS MEng MEd MSW MBA)": 5,
				  " Prof school degree (MD DDS DVM LLB JD)": 6,
				  " Doctorate degree(PhD EdD)": 7}

CONTAINS_NAN_VALS = ['country of birth self',
					 'country of birth father',
					 'country of birth mother']
