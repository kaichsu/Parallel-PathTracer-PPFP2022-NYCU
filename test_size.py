import subprocess

def log(str, file):
    print(str)
    file.write(str);
    file.write('\n');

name = "size"
file = open(f"./test/{name}/log.txt", "w");
for var in [[300, 200], [600, 400], [1200, 800]]:
    width = var[0]
    height = var[1]
    samples = 50
    depth = 50

    # serial
    threads = 1
    img_path = f"./test/{name}/{width}x{height}/serial.png"
    serial = f"./multithreads/raytrace -w {width} -h {height} -s {samples} -d {depth} -t {threads} -o {img_path}";
    log(serial, file);
    process = subprocess.run(serial.split(), stderr=subprocess.PIPE, universal_newlines=True);
    log(process.stderr, file)

    # 8 threads
    threads = 8
    img_path = f"./test/{name}/{width}x{height}/multithreads.png"
    multhreads = f"./multithreads/raytrace -w {width} -h {height} -s {samples} -d {depth} -t {threads} -o {img_path}";
    log(multhreads, file);
    process = subprocess.run(multhreads.split(), stderr=subprocess.PIPE, universal_newlines=True);
    log(process.stderr, file)

    # cuda
    img_path = f"./test/{name}/{width}x{height}/cuda.png"
    cuda = f"./cuda/raytrace -w {width} -h {height} -s {samples} -d {depth} -o {img_path}";
    log(cuda, file);
    process = subprocess.run(cuda.split(), stderr=subprocess.PIPE, universal_newlines=True);
    log(process.stderr, file)

    

