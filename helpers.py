import config.config as config
def get_meta_columns() -> list:
	"""loads the columns from the meta data files
		returns the fields as list"""

	columns_list = []

	with open(config.META_DIR, "r") as metadata_file:
		meta_data = metadata_file.readlines()
		meta_data = meta_data[81:121]

		line_count = 0
		for line in meta_data:
			columns_list.append(line[line.find("(") + 1:line.find(")")])
			if line_count == 23:
				columns_list.append('instance weight')
			line_count += 1

		columns_list.append('Income')

	return columns_list


# def get_meta_dict() -> dict:
# 	"""loads the meta data full list
# 		returns the fields as dictionary"""
#
# 	meta_dict = {}
#
# 	with open(config.META_DIR, "r") as metadata_file:
# 		meta_data = metadata_file.readlines()
# 		meta_data = meta_data[23:68]
#
# 		for i in meta_data:
# 			line = i.split('\t')
# 			meta_dict[line[-1].split('\n')[0]] = line[0][2:].rstrip()
