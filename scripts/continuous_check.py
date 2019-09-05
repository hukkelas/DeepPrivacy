import time
import subprocess

filepath = [
    "fid_values.csv",
    "fid_values_no_pose",
    "fid_values_large.csv"
]

while True:
    cmds = [
        "python3 deep_privacy/metrics/fid_official/start_calculation.py 14 models/isvc/config.yml",
        "python3 deep_privacy/metrics/fid_official/start_calculation.py 14 models/isvc_no_pose/config.yml",
        "python3 deep_privacy/metrics/fid_official/start_calculation.py 14 models/isvc_large/config.yml"
    ]
    for i in range(3):
        cmd = cmds[i]
        process = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE)

        out, err = process.communicate()

        for line in str(out).split("\\n"):
            line = line.strip()
            if line.startswith("[*] "):
                print(i, line)
                with open(filepath[i], "a") as fp:
                    fp.write(line.strip())
            if line.startswith("FID:  "):
                print(i, line)
                with open(filepath[i], "a") as fp:
                    fp.write(f", {line.strip()}\n")
    time.sleep(60*60)
