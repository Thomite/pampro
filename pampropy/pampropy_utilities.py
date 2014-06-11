def design_file_header(statistics):
	
	file_header = "id,timestamp"
	for k,v in statistics.items():
		for stat in v:
			if isinstance(stat, list):
				variable_name = str(k) + "_" + str(stat[0]) + "_" + str(stat[1])
			else:
				variable_name = str(k) + "_" + str(stat)
			
			file_header = file_header + "," + variable_name

	print file_header
	return file_header