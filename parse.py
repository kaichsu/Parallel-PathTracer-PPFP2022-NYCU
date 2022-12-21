import re

f = open("./test/view/log.txt", "r");

data=dict();

version = ""
baseline = 0.1;
for line in f.readlines():
    line = line.rstrip()
    target = re.findall("/([A-z]*).png", line);
    if len(target) > 0:
        version = target[0]
        continue
    target = re.findall("Work took ([0-9.]+) seconds", line);
    if len(target) > 0:
        time = round(float(target[0]), 2)
        if version == "serial":
            baseline = time
        speedup = round(baseline / time,2)
        if version not in data.keys():
            data[version] = list()
        data[version].append(time)
        data[version].append(speedup)

for key, value in data.items():
    print("|", end="")
    print(key, end="")
    for v in value:
        print(f"|{v}", end="")
    print("|")

