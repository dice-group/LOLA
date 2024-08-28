import os
# directories to take new outputs from
source_directories = ['../output-test']
# directory to merge into
target_dir = '../output'

for source_dir in source_directories:
    # loop through tasks
    for task in os.listdir(source_dir):
        task_dir = os.path.join(source_dir, task)
        target_task_dir = os.path.join(target_dir, task)
        # Check if this directory exists for both source and target
        if os.path.isdir(task_dir) and os.path.isdir(target_task_dir):
            # Now we have the task directory, e.g. lm-eval-harness
            print(f'Task found {task}')
            # Loop through subtasks
            for subtask in os.listdir(task_dir):
                subtask_dir = os.path.join(task_dir, subtask)
                target_subtask_dir = os.path.join(target_task_dir, subtask)
                if os.path.isdir(subtask_dir):
                    # Now we have the subtask directory, e.g. arc_ar
                    print(f'Subtask found {subtask}, directory: {subtask_dir}')
                    # if the subtask directory exists for target
                    if os.path.isdir(target_subtask_dir):
                        # TODO: copy all the contents of subtask_dir into target_subtask_dir
                        pass
                    else:
                        # TODO: copy the subtask_dir directly into the target_task_dir
                        pass