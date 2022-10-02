import os
path = './dataset'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join(["dataset_"+str(index), '.jpg'])))
