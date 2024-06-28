from pathlib import Path
import subprocess
import argparse
import yaml
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--overwrite', required=False, action='store_true', help='whether to overwrite existing testing logs')
args = parser.parse_args()

tasks_path = 'projects/retri/'
tasks = [
    'signclip_v1/baseline_sp_b768',
    'signclip_v1/baseline_sp_b768_aug',
    'signclip_v1/baseline_sp_b768_pre',
    'signclip_v1/baseline_sp_b768_pre_aug',
    'signclip_v1_1/baseline_layer',
    'signclip_v1_1/baseline_proj',
    'signclip_v1_1/baseline_anonym',
    'signclip_v1_1/baseline_anonym_no_pre',
]
notes = [
    'E1 pretrained with spreadthesign',
    'E1 + 2D augmentation',
    'E1 + SignVQ preprocessing',
    'E1 + 2D augmentation & SignVQ preprocessing',
    'E2: E1 + 6 -> 12 layers for video',
    'E3: E2 + projection',
    'E4: E2 + anonymization',
    'E4 - SignVQ preprocessing',
]

results = {}
for task in tasks:
    task_path = tasks_path + task + '.yaml'
    config = yaml.safe_load(Path(task_path).read_text())
    log_path = f"{config['eval']['save_path']}/test.log"
    command = f'python locallaunch.py {task_path} --jobtype local_predict'

    print(command)

    # if args.overwrite or (not Path(log_path).is_file()):
    #     Path(log_path).parent.mkdir(exist_ok=True, parents=True)
    #     with open(log_path, 'w') as f:
    #         subprocess.run(command, shell=True, stdout=f)

    result = {}
    with open(log_path, 'r') as f:
        prev_line = ''
        for line in f.readlines():
            line = line.strip()
            if prev_line.startswith('text to video'):
                prefix = 'T2V'
            elif prev_line.startswith('video to text'):
                prefix = 'V2T'
            else:
                prev_line = line
                continue

            metrics = line.split(' - ')
            for metric in metrics:
                name, value = metric.split(': ')
                result[f'{prefix}_{name}'] = value

            prev_line = line

    results[task] = result

df = pd.DataFrame.from_dict(results, orient='index')
df = df[['T2V_P@1', 'T2V_P@5', 'T2V_P@10', 'T2V_Median R', 'V2T_R@1', 'V2T_R@5', 'V2T_R@10', 'V2T_Median R']]
df['notes'] = notes

print(df)

df.to_csv('results_sign_clip_sp.csv')
    