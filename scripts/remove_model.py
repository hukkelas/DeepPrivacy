import sys
import shutil
import os 
names = sys.argv[1:]
print(names)
answer = input("Are you sure you want to delete model \"{}\"?".format(names)).strip()

if answer != "y" and "answer" != "yes":
    exit(0)
for name in names:
    to_remove = [
        "models/{}/checkpoints".format(name),
        "models/{}/summaries".format(name),
        "models/{}/generated_data".format(name),
        "models/{}/transition_checkpoints".format(name)
    ]
    for folder in to_remove:
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            print("Folder already removed:", folder)
        print("Removed:", folder)

    print(os.system("docker rm haakohu_{}".format(name)))
