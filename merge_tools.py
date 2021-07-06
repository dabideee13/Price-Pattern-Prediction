import os
import subprocess
import pandas as pd


def get_max_files(expression: str,
                  path: str = None) -> int:
    numerics = list()
    for files in os.listdir(path):
        if expression in files:
            for num in files:
                if num.isdigit():
                    numerics.append(int(num))
    return max(numerics)

def create_new_file(file_num: int,
                    path: str = None):
    print("Creating new file: 'all_patterns{}.csv'".format(file_num + 1))
    subprocess.call('touch all_patterns{}.csv'.format(
        file_num + 1).split())

def create_folder_name(num: int, 
                       path: str = None):
    f_name = 'detect{}'
    return os.path.join(path, f_name).format(num)

def create_folder(path: str = None):
    if path is None:
        path = '/Users/d.e.magno/Datasets/stocks'

    try:
        max_ = get_max_files('detect', path=path)
        new_iter = max_ + 1
        print("Creating folder...")
        folder_name = create_folder_name(new_iter, path)
        os.mkdir(folder_name)
        print("Done.")
        print()
    except Exception as ex:
        new_iter = 1
        print("Creating folder...")
        folder_name = create_folder_name(new_iter, path)
        os.mkdir(folder_name)
        print("Done.")
        print()
    except:
        print("Skipped.")
        pass


if __name__ == '__main__':

    #create_new_file(get_max_files('all_patterns'))

    # To detect
    path = '/Users/d.e.magno/Datasets/stocks'

    # Create folder for storage of stocks to detect
    create_folder(path)
   
    # Done.
    print("Finished.")


    