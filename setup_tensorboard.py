import os
import shutil

modeldirs = os.listdir("models")
modeldirs = [os.path.join("models", x) for x in modeldirs]
modeldirs = [x for x in modeldirs if os.path.isdir(x)]

summaries_dir = [os.path.join(x, "summaries") for x in modeldirs]
summaries_dir = [x for x in summaries_dir if os.path.isdir(x)]

tensorboard_path = "tensorboard_summaries"
if os.path.isdir(tensorboard_path):
    shutil.rmtree(tensorboard_path)

os.makedirs(tensorboard_path)
abspath = os.path.abspath(".")
for sdir in summaries_dir:
    
    model_name = sdir.split(os.sep)
    model_name = model_name[-2]
    sdir = os.path.join(abspath, sdir)
    new_dir = os.path.join(abspath, tensorboard_path, model_name)
    os.symlink(sdir, new_dir, target_is_directory=True)
    print(f"Creating symlink: \n\t\tsrc: {sdir}, \n\t\tdst: {new_dir}")


