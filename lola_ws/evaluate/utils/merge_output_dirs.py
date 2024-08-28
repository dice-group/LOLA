import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# directories to take new outputs from
source_directories = ['../output-test', '../output']
# directory to merge into
target_dir = '../output-merge'

def copy_subtask(subtask, task_dir, target_task_dir):
    subtask_dir = os.path.join(task_dir, subtask)
    target_subtask_dir = os.path.join(target_task_dir, subtask)
    if os.path.isdir(subtask_dir):
        # Now we have the subtask directory, e.g. arc_ar
        # print(f'Subtask found {subtask}, directory: {subtask_dir}')
        # if the subtask directory exists for target
        if os.path.isdir(target_subtask_dir):
            # copy all the contents of subtask_dir into target_subtask_dir
            for item in os.listdir(subtask_dir):
                s_item = os.path.join(subtask_dir, item)
                t_item = os.path.join(target_subtask_dir, item)
                if os.path.isdir(s_item):
                    shutil.copytree(s_item, t_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(s_item, t_item)
        else:
            # copy the subtask_dir directly into the target_task_dir
            shutil.copytree(subtask_dir, target_subtask_dir)

def process_task(task, source_dir):
    task_dir = os.path.join(source_dir, task)
    target_task_dir = os.path.join(target_dir, task)
    # Check if this directory exists for both source and target
    if os.path.isdir(task_dir) and os.path.isdir(target_task_dir):
        # Now we have the task directory, e.g. lm-eval-harness
        # print(f'Task found {task}')
        # Loop through subtasks
        subtasks = os.listdir(task_dir)
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda subtask: copy_subtask(subtask, task_dir, target_task_dir), subtasks),
                      total=len(subtasks), desc=f"Processing {task_dir}"))
    else:
        print('No source/target task directories to merge. Please check if task directories exist for both source and target.')
        print(f'Expected: {task_dir} and {target_task_dir}')

# Main loop through source directories and tasks
for source_dir in source_directories:
    tasks = os.listdir(source_dir)
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda task: process_task(task, source_dir), tasks),
                  total=len(tasks), desc="Overall Progress"))
