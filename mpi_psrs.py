from mpi4py import MPI
import random
import bisect
import argparse

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--data_size', type=int, required=True, help='Size of the array')
    args = parser.parse_args()
    data_size = args.data_size

    array1 = [random.randint(0, 10000000) for _ in range(data_size)]
    array2 = [i for i in array1]

    start_time_np = MPI.Wtime()
    array1.sort()
    end_time_np = MPI.Wtime()

    start_time_p = MPI.Wtime()
    # 划分
    group_size = data_size // size
    sub_array = [array2[i:i+group_size] for i in range(0, data_size, group_size)]
    if data_size % size != 0:
        last = sub_array.pop()
        sub_array[-1].extend(last)
else:
    sub_array = None

# 排序
sub_array = comm.scatter(sub_array, root=0)
sub_array.sort()

# 选择样本
step = max(len(sub_array) // size, 1)
samples = [sub_array[i] for i in range(0, len(sub_array), step)][:size]
samples = comm.gather(samples, root=0)

# 样本排序
if rank == 0:
    samples_array = [i for sample in samples for i in sample]
    samples_array.sort()
    
    # 主元
    pivots = [samples_array[i] for i in range(size, len(samples_array), size)][:size-1]
else:
    pivots = None

# 主元划分
pivots = comm.bcast(pivots, root=0)
data_cut = []
s_index = 0
for pivot in pivots:
    index = bisect.bisect_left(sub_array, pivot, s_index)
    data_cut.append(sub_array[s_index:index])
    s_index = index
data_cut.append(sub_array[s_index:])

# 全局交换
data_exchange = [data_cut[rank]]
for i in range(size):
    if i != rank:
        data_exchange_ = comm.sendrecv(sendobj=data_cut[i], dest=i, source=i)
        data_exchange.append(data_exchange_)

data_exchange = [i for sublist in data_exchange for i in sublist]

# 归并排序
data_exchange.sort()
data_exchange = comm.gather(data_exchange, root=0)
if rank == 0:
    array2_ = [i for sublist in data_exchange for i in sublist]
    end_time_p = MPI.Wtime()
    # print(array1 == array2_)
    print(f'Speed up Rate:{(end_time_np-start_time_np)/(end_time_p-start_time_p)}')


    





