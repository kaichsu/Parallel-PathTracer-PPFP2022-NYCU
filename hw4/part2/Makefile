all: matmul

matmul: matrix.cc main.cc
	mpicxx -o matmul main.cc matrix.cc

clean:
	rm -f *.o matmul

.PHONY: move test
move:
	parallel-scp -A -h hosts -r ~/HW4 ~/
	
test:
	mpirun -np 4 --hostfile hosts matmul < data0 