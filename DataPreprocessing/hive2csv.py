import os

import pandas as pd
import pyarrow.parquet as pq


def read_pyarrow(path, use_threads=1):
    return pq.read_table(path, use_threads=use_threads).to_pandas()


def get_file_list(file_dir):
    L = []
    for root, _, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.parquet':
                L.append(os.path.join(root, file))
    return L


def get_csv(file_list):
    init_flag = 0
    for f in file_list:
        print('The current handling file is:\n', f)
        if init_flag == 0:
            init_df = read_pyarrow(f)
            init_flag = 1
        else:
            t_df = read_pyarrow(f)
            init_df = pd.concat([init_df, t_df])
    return init_df


path = '../data/raw_data'
dirs = os.listdir(path)
print(dirs)
for dir_path in dirs:
    file_list = get_file_list(path + "/" + dir_path)
    print(file_list)
    df = get_csv(file_list)
    filename = '../data/csv_data/' + dir_path + '.csv'
    print(filename)
    df.to_csv(filename, sep=',', index=False, mode='w', line_terminator='\n', encoding='utf-8')
