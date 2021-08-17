import os
from tqdm import tqdm

folder_path = "C:/Users/Admin/Desktop/Face Recognition Pipeline/detection/dataset_Widerface"
folders = os.listdir(folder_path)

result = []
for folder in tqdm(folders):
    image_folder = os.path.join(folder_path + "/", folder)
    image_names = [os.path.join(folder + "/", f) for f in os.listdir(image_folder)]
    result.extend(image_names)

f = open(os.path.join(folder_path, "wider_val.txt"), "w")
for res in result:
    f.write(res + "\n")

f.close()