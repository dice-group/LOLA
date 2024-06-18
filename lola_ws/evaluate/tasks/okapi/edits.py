import sys

# print(sys.argv)
path_to_file = sys.argv[1] + "/lm_eval/models/huggingface.py"
# print(path_to_file)

file = open(path_to_file, "r")
file_content = file.read()
file.close()



# if(sys.argv[1] == '1'):
    
#     identifier = """check_integrity=False,
#     decontamination_ngrams_path=None,
#     write_out=False,
#     output_base_path="""
    
#     first_index = file_content.index(identifier) + len(identifier) 

#     identifier = """"device": device"""
#     second_index = file_content.index(identifier) + len(identifier)

#     identifier = """output_base_path=output_base_path,"""
#     third_index = file_content.index(identifier) + len(identifier)

#     identifier = """"description_dict": description_dict,"""
#     fourth_index = file_content.index(identifier) + len(identifier)

#     identifier = """None,
#     decontamination_ngrams_path=None,
#     write_out=False,
#     output_base_path="""
#     fifth_index = file_content.index(identifier) + len(identifier)

#     # print(fifth_index)
#     file_content = file_content[:first_index] + """CHANGEOUTPUTPATH,\n\ttrust_remote_code=True,""" + file_content[first_index + len("None,"):second_index] + """, "trust_remote_code": trust_remote_code""" + file_content[second_index:third_index] + """\n\t\ttrust_remote_code=trust_remote_code,""" + file_content[third_index:fourth_index] + """\n\t\t"trust_remote_code": trust_remote_code,""" + file_content[fourth_index:fifth_index] + """CHANGEOUTPUTPATH,\n\ttrust_remote_code=True,""" + file_content[fifth_index + len("None,"):]
#     # print(file_content)
# else:
#     result_path = sys.argv[3]
#     identifier = """CHANGEOUTPUTPATH"""
#     file_content_list = file_content.split(identifier)
#     file_content = file_content_list[0] + result_path + file_content_list[1] + result_path + file_content_list[2]


file_content_list = file_content.split("trust_remote_code: Optional[bool] = False")

file_content = file_content_list[0]
file_content_list.pop[0]

while(len(file_content_list) != 0):
    file_content += "trust_remote_code: Optional[bool] = True" + file_content_list[0]
    file_content_list.pop[0]



file = open(path_to_file, "w")
file.write(file_content)
file.close()

path_to_file = sys.argv[1] + "/scripts/run.sh"
# print(path_to_file)

file = open(path_to_file, "r")
file_content = file.read()
file.close()

file_content = """#!/bin/bash
lang=$1
model_path=$2
output_path=$3
tasks=arc_${lang},hellaswag_${lang},mmlu_${lang}
device=cuda

python main.py \
    --tasks=${tasks} \
    --model_args pretrained=${model_path} \
    --output_path=${output_path} \
    --device=${device}"""


file = open(path_to_file, "w")
file.write(file_content)
file.close()

