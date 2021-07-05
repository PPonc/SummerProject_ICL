import sys
import argparse

def to_loc(str, old = ',', new = '.'):
    return str.replace(old, new)

def parse_file(filename):
    names = []
    file = open(filename)
    samples = []
    v = []
    l = 0
    s = 0

    while(line := file.readline()):
        line = line.replace("\n", "")
        if line == "start":
            v = []
            l = 0
            continue
        elif line == "finish":
            samples.append(v)
            s += 1
            continue
        else:
            l += 1
            vals = line.split(";")
            if l == 1 or vals[0] == "duration":
                for i,val in enumerate(vals):
                    if i % 2 == 0 and s == 0:
                        names.append(val)
                    elif i % 2 == 1:
                        v.append(to_loc(val))
            else:
                if s == 0:
                    names.append(vals[2])
                v.append(to_loc(vals[0]))

    file.close()
    return names,samples

def print_samples(samples):
    for s in samples:
        print(*s, sep=";")

def run(filename, name, desc):
    names,samples = parse_file(filename)
    print("name;{};desc;{};".format(name, desc))
    print(*names, sep=';')
    print_samples(samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate perf measurements with energy")
    parser.add_argument('-f', '--filename', type=str, default="measures.csv")
    parser.add_argument('-n', '--name', type=str, default='unknown', help='Name of the measurement')
    parser.add_argument('-d', '--desc', type=str, default=None, help='Description for measurement')
    args = parser.parse_args()
    run(args.filename, args.name, args.desc)
