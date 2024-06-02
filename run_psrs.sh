num_thread=("2" "4" "8" "16")
size=("1000" "5000" "10000" "100000")

for ((i=0;i<4;i+=1))
do
    for ((j=0;j<4;j+=1))
    do
    mpiexec --allow-run-as-root -n ${num_thread[i]} python mpi_psrs.py -s ${size[j]}
    done
done    
