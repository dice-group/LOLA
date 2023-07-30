"""
This python script is responsible for setting up and submitting the SLURM batch jobs for our scaling experiments.
In this experiment, we plan to test the scaling capability of the DeepSpeed's GPT MoE implementation.
"""

import subprocess
import json

# Output file
OUTPUT_FILE = 'scaling_exp_det.json'
# Script to execute
SCALE_SCRIPT = 'gpt3-moe-scaling-train.sh'
"""
Model configs to test, key represents the model size in billion(s) and the corresponding list has the values for 
the model shape  in the following order: [NUM_LAYERS, HIDDEN_SIZE, NUM_ATTN_HEADS]
"""
MODEL_CONFIGS = {
    '0.35': [12, 768, 12],
    '0.76': [24, 1536, 16],
    '1.3': [24, 2048, 16],
    '2.7': [32, 2560, 32],
    '6.7': [32, 4096, 32],
    '13': [40, 5120, 40]
}

BATCH_SIZE_VALUES = [1, 2, 4, 8, 16, 32]

"""
Hardware configs the list represents the following values in order: [NUM_NODES, NUM_GPU_PER_NODE, TIME]
"""
HARDWARE_CONFIGS = [
    [1, 1, '03:00:00'],
    [1, 4, '03:00:00'],
    [2, 4, '03:00:00'],
    [4, 4, '03:00:00']
]

EXP_NAME_TEMPLATE = 'GPT_%sB_MoE128_%dBATCH_%dGPU_%dNode'


def submit_slurm_job(sbatch_args, script_args):
    """
    Executes the given script along with its arguments and captures the slurm job ID

    :param script_args: list containing script to execute along with is arguments
    :param sbatch_args: list containing sbatch arguments
    :return: job_id: ID of the submitted slurm job
    """
    try:
        # Run sbatch command and capture its output
        process_list = ['sbatch']
        process_list.extend(sbatch_args)
        process_list.extend(script_args)
        print('Executing the following: %s' % ' '.join(map(str, process_list)))
        result = subprocess.run(process_list, capture_output=True, text=True, check=True)
        # Get the job ID from the output
        job_id = int(result.stdout.strip().split()[-1])
        print('Job ID: %s\n\n' % job_id)
        return job_id
    except subprocess.CalledProcessError as e:
        print("Error submitting the job:")
        print(e)
        return None


def compose_sbatch_args(hw_cfg, exp_name):
    return ['--job-name="%s"' % exp_name, '--nodes=%d' % hw_cfg[0], '--ntasks-per-node=1',
            '--gres=gpu:a100:%d' % hw_cfg[1], '--time=%s' % hw_cfg[2]]


def compose_script_args(mdl_cfg_key, hw_cfg, b_sz):
    # unpack the params
    mdl_cfg = MODEL_CONFIGS[mdl_cfg_key]
    mdl_num_layers = mdl_cfg[0]
    mdl_hidden_size = mdl_cfg[1]
    mdl_num_attn_heads = mdl_cfg[2]

    hw_num_nodes = hw_cfg[0]
    hw_num_gpu_per_nodes = hw_cfg[1]

    exp_name = EXP_NAME_TEMPLATE % (mdl_cfg_key, b_sz, (hw_num_nodes * hw_num_gpu_per_nodes), hw_num_nodes)

    return [SCALE_SCRIPT, exp_name, mdl_cfg_key, mdl_num_layers, mdl_hidden_size, mdl_num_attn_heads, b_sz]


# sbatch --job-name="${EXP_NAME}" --nodes=$NUM_NODES --ntasks-per-node=1 --gres=gpu:a100:$NUM_GPU_PER_NODE --time=$EXP_RUNTIME  gpt3-moe-scaling-train.sh

if __name__ == "__main__":
    res_dict = dict()
    # for each model_config
    for model_config_name in MODEL_CONFIGS:
        # for each hardware config
        for hw_config in HARDWARE_CONFIGS:
            # for each batch size
            for batch_size in BATCH_SIZE_VALUES:
                # Compose arguments
                scr_args = compose_script_args(model_config_name, hw_config, batch_size)
                ex_name = scr_args[0]
                sb_args = compose_sbatch_args(hw_config, ex_name)
                # Execute the experiment
                print('Submitting the the batch job for %s' % ex_name)
                res_dict[ex_name] = submit_slurm_job(sb_args, scr_args)

    # save the results
    with open('', 'w') as op_file:
        json.dumps(res_dict, indent=4)

    print('Finished submitting experiments!')
