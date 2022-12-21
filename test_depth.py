import subprocess

def log(str, file):
    print(str)
    file.write(str);
    file.write('\n');

name = "depth"
file = open(f"./test/{name}/log.txt", "w");
for var in [10, 25, 50]:
    width = 600
    height = 400
    samples = 50
    depth = var

    # serial
    threads = 1
    img_path = f"./test/{name}/{depth}/serial.png"
    serial = f"./multithreads/raytrace -w {width} -h {height} -s {samples} -d {depth} -t {threads} -o {img_path}";
    log(serial, file);
    process = subprocess.run(serial.split(), stderr=subprocess.PIPE, universal_newlines=True);
    log(process.stderr, file)

    # 8 threads
    threads = 8
    img_path = f"./test/{name}/{depth}/multithreads.png"
    multhreads = f"./multithreads/raytrace -w {width} -h {height} -s {samples} -d {depth} -t {threads} -o {img_path}";
    log(multhreads, file);
    process = subprocess.run(multhreads.split(), stderr=subprocess.PIPE, universal_newlines=True);
    log(process.stderr, file)

    # cuda
    img_path = f"./test/{name}/{depth}/cuda.png"
    cuda = f"./cuda/raytrace -w {width} -h {height} -s {samples} -d {depth} -o {img_path}";
    log(cuda, file);
    process = subprocess.run(cuda.split(), stderr=subprocess.PIPE, universal_newlines=True);
    log(process.stderr, file)

    

