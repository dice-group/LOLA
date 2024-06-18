import sys

# print(sys.argv)
path_to_file = sys.argv[1] + "/lm_eval/models/huggingface.py"
# print(path_to_file)

file = open(path_to_file, "r")
file_content = file.read()
file.close()


file_content_list = file_content.split("trust_remote_code: Optional[bool] = False")

file_content = file_content_list[0]
del file_content_list[0]

while(len(file_content_list) != 0):
    file_content += "trust_remote_code: Optional[bool] = True" + file_content_list[0]
    del file_content_list[0]



file = open(path_to_file, "w")
file.write(file_content)
file.close()

path_to_file = sys.argv[1] + "/scripts/run.sh"

file_content = """#!/bin/bash
lang=$1
model_path=$2
output_path=$3
tasks=arc_${lang},hellaswag_${lang},mmlu_${lang}
device=cuda

python main.py \\
    --tasks=${tasks} \\
    --model_args pretrained=${model_path} \\
    --output_path=${output_path} \\
    --device=${device}"""


file = open(path_to_file, "w")
file.write(file_content)
file.close()

