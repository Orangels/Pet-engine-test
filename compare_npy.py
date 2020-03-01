import numpy as np
import os

path_pet = 'P000014.npy'
file_name, extension_name = os.path.splitext(path_pet)
path_engine = file_name + '_engine' + extension_name


npy_pet = np.load(path_pet)
npy_engine = np.load(path_engine)
print((npy_pet==npy_engine).all())
compare = (npy_pet==npy_engine)
compare_coor = np.where(compare==False)
print(len(compare_coor[0]))
