import os
import shutil


def delete_different_files(folder1, folder2):
    # 获取两个文件夹中所有的文件名
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    for files in files1:
        files = files.split('_GT')[0]
        print(files)
    # 找到只在folder1中存在的文件名
    for files in files2:
        files = files.split('_hazy')[0]
        print(files)
    different_files = files1 - files2

    # 遍历这些文件名并删除对应的文件
    for file in different_files:
        file_path = os.path.join(folder1, file+'_hazy.png')
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

        # 使用你的文件夹路径替换这里的'path_to_folder1'和'path_to_folder2'

path_to_folder1 = ""
path_to_folder2 = ""
delete_different_files(path_to_folder1, path_to_folder2)