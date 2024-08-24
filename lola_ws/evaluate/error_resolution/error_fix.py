import os
import yaml
import pandas as pd 
import subprocess

combinations = {}
commands = []

def preprocess_txt_to_dict(textfile_name):

    error_file = open(textfile_name,"r")
    error_file_content = error_file.read()

    models = [["dice-research__lola_v1","lola"],["FacebookAI__xlm-roberta-large", "xlmr-l"],["FacebookAI__xlm-roberta-xl", "xlmr-xl"],["google__mt5-large", "mt5-l"],["google__mt5-xl", "mt5-xl"],["facebook__m2m100_1.2B", "m2m"],["ai-forever__mGPT", "mgpt"],["bigscience__bloom-7b1", "bloom7b1"],["bigscience__bloom-1b1", "bloom1b1"],["bigscience__bloom-1b7", "bloom1b7"],["facebook__mbart-large-50", "mbart"],["Salesforce__xgen-7b-8k-base", "xgen"],["meta-llama__Llama-2-7b", "llama2"],["HuggingFaceH4__zephyr-7b-beta", "zephyr"],["Unbabel__TowerBase-7B-v0.1", "towerbase"],["SeaLLMs__SeaLLM-7B-v2", "seallm-7b-v2"],["SeaLLMs__SeaLLM-7B-v2.5", "seallm-7b-v2_5"],["SeaLLMs__SeaLLMs-v3-1.5B-Chat", "seallm-1_5b-v3"],["Henrychur__MMedLM2-1_8B", "mmedlm"],["mistralai__Mixtral-8x7B-v0.1", "mixtral"],["tiiuae__falcon-7b", "falcon"],["MediaTek-Research__Breeze-7B-Base-v1_0", "breeze"],["cis-lmu__glot500-base", "glot500"]]

    for i in models:
        error_file_content = error_file_content.replace(i[0],i[1])

    original_dicts = {}

    count = 0
    for i in error_file_content.split('\n')[1:]:
        original_dicts[count] = "{" + i.split('{')[1]
        count+=1

    count = 0
    for key1 in original_dicts:
        original_dicts[key1] = yaml.load(original_dicts[key1])
        for key2 in original_dicts[key1]:
            for value in original_dicts[key1][key2]:
                if('truth' in key2):
                    task = "truthfulqa_mc1"
                    lang = key2.split("_")[1]
                else:
                    task = key2.rsplit("_",1)[0]
                    lang = key2.rsplit("_",1)[1]
                if(value not in combinations):
                    combinations[value]={}
                if(task not in combinations[value]):
                    combinations[value][task] = []
                if(lang not in combinations[value][task]):
                    combinations[value][task].append(lang)
                    count += 1
    
    command_pt1 = "python3 noctua2_run_evaluation.py --models="
    command_pt2 = """ --tasks="lm-eval-harness:"""
    command_pt3 = """" --languages="""
    command_pt4 = """ --results_dir=./error_resolution/output"""
    counter = 0
    for model in combinations:
        for task in combinations[model]:
            langs = ""
            for lang in combinations[model][task]:
                langs += lang + ","
            langs = langs[:-1]
            commands.append(command_pt1 + model + command_pt2 + task + command_pt3 + langs + command_pt4)
            counter += 1


def create_commands_csv(csvfile_name):
    dict = {'Commands': commands, 'Execution status': ["" for x in commands], 'Success': ["" for x in commands]}  
    df = pd.DataFrame(dict) 
    df.to_csv(csvfile_name) 


def execute_commands():
    os.chdir(os.getcwd().rsplit("/",1)[0])
    temp_command = ""
    for command in commands:
        # print(command)
        temp_command = command
    sub_proc_arr = temp_command.replace('''"''','').split(" ")
    subprocess.run(sub_proc_arr)



preprocess_txt_to_dict("error_summary_new.txt")
# create_commands_csv('Commands.csv')
execute_commands()


