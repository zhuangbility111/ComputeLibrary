import os
import re

path = "./log/"
files= os.listdir(path)

for file in files:
    p_read = path + "/" + file
    if not os.path.isdir(p_read):
        f_read = open(p_read)
        f_write = open(path + "/csv/" + file + ".csv", 'w')
        res = 0.0
        for line in f_read:
            line = line.replace('\n', '').replace('\r', '')
            if len(line) > 0 and len(re.findall(r"\d+\.?\d*", line)) > 0:
                tmp_str = re.findall(r"\d+\.?\d*", line)[0]
                if line[0] == 'k' or line[0] == 'i' or line[0] == 'g' or line[0] == 'o':
                    f_write.write(tmp_str + ',')
                    res += float(tmp_str)
                if line[0] == 'A':
                    f_write.write(str(res) + ',')
                    f_write.write(tmp_str + '\r\n')
                    res = 0.0
        f_read.close()
        f_write.close()

