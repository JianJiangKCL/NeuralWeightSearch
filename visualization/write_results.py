import os

import shutil
from pathlib import Path
import argparse
import pandas as pd

'''
read the last line of the file for the final result

move the directory to the recorded_results folder

'''


#   move the directory to the recorded_results folder
def move_dir(root, src_dir, des_dir):
    # print('moving directory to the recorded_results folder')
    # os.chdir(root)
    src_dir = os.path.join(root, src_dir)
    # os.chdir(DES_DIR)
    tgt_dir = os.path.join(root, des_dir)
    # os.rename(src_dir, tgt_dir)
    try:
        shutil.move(src_dir, tgt_dir)
        print(f'{src_dir}  moved')
    except:
        print(f'{src_dir}  not moved')
    # os.chdir(root)

def delete_dir(root, src_dir):
    src_dir = os.path.join(root, src_dir)
    shutil.rmtree(src_dir)
    print(f'{src_dir}  deleted')

#   read the last line of the file for the final result
def read_last_line(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return lines[-1]
# write results in list to file
# def write_results
def main(args):
    results = []
    settings = []
    accs = []
    dir_to_move = []
    dir_to_delete = []
    # mkdir
    root = args.root


    DES_DIR = os.path.join(root, args.target_dir)
    k = f'{root}_accuracy.md'
    if not os.path.exists(DES_DIR):
        os.mkdir(DES_DIR)
    for path in Path(root).rglob('*.txt'):

        setting = path.parent.name
        # avoid move the root dir
        if setting in root:
            continue
        print(setting)
        # avoid the recorded_results folder to be recorded
        if DES_DIR in path.parts:
            continue
        # if setting.find(DIR_START) != -1:  # -1 is not found
        last_line = read_last_line(path)
        if last_line.find('acc') != -1:
            # record the finished setting
            # this need to be changed for other applications
            accuracy = last_line.split('acc')[1].strip()
            accs.append(float(accuracy))
            setting = setting.split('task')[1].strip()
            # this need to be changed for other applications
            if args.save_fullname:
                settings.append(setting)
            else:
                settings.append(int(setting.split('_')[0][4:]))
            accuracy = setting + ' ' + accuracy
            # item = f"{setting.split('results_')[1]} : {accuracy}"

            results.append(accuracy)
            dir_to_move.append(setting)

        else:
            # delete those not finished
            dir_to_delete.append(setting)

    # for dir in dir_to_move:
    #
    #     move_dir(root, dir, DES_DIR)
    #
    # for dir in dir_to_delete:
    #     delete_dir(root, dir)
    avg_acc = sum(accs) / len(accs)
    setting = 100
    accs.append(avg_acc)
    settings.append(setting)
    if args.save_fullname:
        df = pd.DataFrame(data={'setting': settings, 'accuracy': accs})
        df['setting'] = pd.to_numeric(df['setting'])
        # import natsort
        # df = natsort.natsorted(df, key=lambda x: x['setting'])
        df.sort_values(by='setting', ascending=True, inplace=True)
        df.to_csv(os.path.join(root, 'accuracy.csv'), index=False)
    else:
        df = pd.DataFrame({'task_id': settings, 'accuracy': accs})
        df.sort_values(by='task_id', ascending=True, inplace=True)

        df.to_csv(os.path.join(root, 'accuracy.csv'), index=False)




if __name__ == "__main__":
    # only support one level root directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    # parser.add_argument("--src_dir", type=str, default="")
    parser.add_argument("--target_dir", type=str, default='recorded_results')
    parser.add_argument("--save_fullname", type=int, default=1)
    args = parser.parse_args()
    main(args)
# dir_name = 'results_lr0.01_e1_nemb256_ks2_nw5_kq10_tbs1_ttn20_gs1'
#
# move_dir(dir_name)
