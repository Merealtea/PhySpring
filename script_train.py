import glob
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default=None,
                    help="Device to train on, e.g. cuda:0 / cuda:1 (also accepts 0 / 1). "
                         "Overrides --local_rank.")
parser.add_argument("--local_rank", type=int, default=1,
                    help="Legacy way to pick the GPU index; use --device instead.")
args = parser.parse_args()


def parse_device_index(device_str):
    """Convert 'cuda:0' / 'cuda:1' / '0' / '1' into a GPU index."""
    s = str(device_str).strip().lower()
    if s.startswith("cuda:"):
        s = s[len("cuda:"):]
    elif s == "cuda":
        return 0
    if not s.isdigit():
        raise ValueError(
            f"Invalid --device '{device_str}'. Use e.g. cuda:0, cuda:1, 0 or 1."
        )
    return int(s)


if args.device is not None:
    local_rank = parse_device_index(args.device)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot use --device cuda:<idx>.")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"--device cuda:{local_rank} requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible "
            f"({', '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))})."
        )
    print(f"Using device: cuda:{local_rank} ({torch.cuda.get_device_name(local_rank)})")
else:
    local_rank = args.local_rank
    print(f"Using device: cuda:{local_rank} (from --local_rank default)")

case = "End2End_Reduction" # 'neural_spring_field' # "evospring" # End2End # End2End_Reduction
CONFIG_FILE=f"./configs/{case}"
# 0: train 1: local test 2: global
MODE=0
RESTART_EPOCH=-1

# Load config file (key=value format)
config = {}
with open(CONFIG_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            # Remove quotes if present
            value = value.strip().strip("'\"")
            config[key] = value

base_path = "./evomesh_optimization_outputs"
res_path = f"./res/{case}"
init_physical_settings = "./configs/initial_training_settings.json"
# load initial physical setting here
with open(init_physical_settings, "r") as f:
    physical_settings = json.load(f)

if not os.path.exists(res_path):
    os.makedirs(res_path)
if not os.path.exists(base_path):
    print(f"Base path {base_path} does not exist.")
    exit(1)

dir_names = os.listdir(base_path)
exclusive_cases = []
finished_cases = os.listdir(res_path)

dir_names = ['weird_package'] #, 'double_lift_zebra', 'rope_double_hand']

# Real2sim Cases
for idx, case_name in enumerate(dir_names):

    if case_name in exclusive_cases:
        continue
    
    print(f"Running case: {case_name}")

    # Merge this object's physical settings (lr, stiffness, ...) into the config
    case_config = dict(config)
    case_config.update(physical_settings.get(case_name, {}))

    os.system(
        f"python src/spring_mass_main.py \
            -case {case} -space_dim {case_config['space_dim']} -local_rank {local_rank} \
            -n_train {case_config['n_train']} -n_valid {case_config['n_valid']} -n_test {case_config['n_test']} -time_len {case_config['time_len']} \
            -noise_level {case_config['noise_level']} \
            -multi_mesh_layer {case_config['multi_mesh_layer']} -consist_mesh {case_config['consist_mesh']} \
            -num_epochs {case_config['num_epochs']} -batch {case_config['batch']} -lr {case_config['lr']} -gamma {case_config['gamma']} \
            -restart_epoch {RESTART_EPOCH} -mp_time {case_config['MP_time']} \
            -data_dir {case_config['data_dir']} -dump_dir {case_config['dump_dir']} -mode {MODE} -object_case {case_name} -reduction {case_config['reduction']} -init_spring_Y {case_config['init_spring_Y']}"
    )


