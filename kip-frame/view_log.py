import os

check_dir = "log_dir/check"

filelist = os.listdir(check_dir)

for filename in filelist:
    if not filename.endswith(".log"):
        continue
    history_acc_ = []
    current_acc_ = []
    with open(f"{check_dir}/{filename}",encoding="utf-8") as f:
        # with open(f"{filename}") as f:
        lines = f.readlines()
        for line in lines:
            if "woprompt" in line:
                continue

            if line.startswith("history test acc "):
                line.strip()
                offset = len("history test acc ")
                acc_list = eval(line[offset:])
                if len(acc_list) == 10:
                    history_acc_.append(acc_list)
            if line.startswith("current test acc "):
                line.strip()
                offset = len("current test acc ")
                acc_list = eval(line[offset:])
                if len(acc_list) == 10:
                    current_acc_.append(acc_list)

    N = len(history_acc_)
    M = len(current_acc_)
    #     N=1
    #     M=1

    import numpy as np

    y = []
    z = []
    for i in range(10):
        res = [history_acc_[j][i] for j in range(N)]
        res2 = [current_acc_[j][i] for j in range(M)]
        yi = np.mean(res)
        y.append(yi)

        zi = np.mean(res2)
        z.append(zi)

    res = ""
    for item in y:
        res += f"{100 * item:.4}, "
    if ".log" in filename:
        print(f"'{filename}':\n{y}")
        #         print(f"current acc:\n{z}\nmean:{np.mean(z)}")
        print(res)
        if N < 5:
            print(f"Warnning： not upto 5/{N} rounds")
        print("\n")