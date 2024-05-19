import sys

# print(sys.argv)
path_to_file = sys.argv[1] + "/lm_eval/evaluator.py"
# print(path_to_file)

file = open(path_to_file, "r")
file_content = file.read()
file.close()

identifier = """check_integrity=False,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,"""
first_index = file_content.index(identifier) + len(identifier)

identifier = """"device": device"""
second_index = file_content.index(identifier) + len(identifier)

identifier = """output_base_path=output_base_path,"""
third_index = file_content.index(identifier) + len(identifier)

identifier = """"description_dict": description_dict,"""
fourth_index = file_content.index(identifier) + len(identifier)

identifier = """None,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,"""
fifth_index = file_content.index(identifier) + len(identifier)

# print(fifth_index)
file_content = file_content[:first_index] + """\n\ttrust_remote_code=True,""" + file_content[first_index:second_index] + """, "trust_remote_code": trust_remote_code""" + file_content[second_index:third_index] + """\n\t\ttrust_remote_code=trust_remote_code,""" + file_content[third_index:fourth_index] + """\n\t\t"trust_remote_code": trust_remote_code,""" + file_content[fourth_index:fifth_index] + """\n\ttrust_remote_code=True,""" + file_content[fifth_index:]
# print(file_content)

file = open(path_to_file, "w")
file.write(file_content)
file.close()