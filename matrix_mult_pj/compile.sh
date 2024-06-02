nvcc -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -c matrix_mult_p.cu -o matrix_mult_p.o
mpicxx -g matrix_mult_p.o -o matrix_mult_p -L/usr/local/cuda-11.6/targets/x86_64-linux/lib -lcudart
mpirun --allow-run-as-root -n 1 matrix_mult_p 2000 2000 2000 2000