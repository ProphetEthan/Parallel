#include<stdio.h>
#include<omp.h>
#include<time.h>
#include<stdlib.h>


// 划分数据
int partition(int* data, int start, int end){
    int tmp = data[start];
    while (start < end){
        while (start < end && data[end] >= tmp) end--;
        data[start] = data[end];
        
        while (start < end && data[start] <= tmp) start++;
        data[end] = data[start];
    }
    data[start] = tmp;
    return start;
}

//快速排序
void quickSort(int* data, int start, int end){
    if (start < end){
        int pos = partition(data, start, end);

        //并行
        #pragma omp parallel sections
        {
            #pragma omp section
                quickSort(data, start, pos-1); 
            #pragma omp section
                quickSort(data, pos+1, end);
        }
    }
}

int main(int argc, char* argv[]){
    int n_thread = atoi(argv[2]), data_size = atoi(argv[1]);
    int* data = (int*)malloc(sizeof(int) * data_size);
    int* data2 = (int*)malloc(sizeof(int) * data_size);

    srand(time(NULL) + rand()); 
    for (int i = 0; i < data_size; i++) {
        data[i] = rand();
        data2[i] = data[i];
    }

    double start_time_p = omp_get_wtime();
    omp_set_num_threads(n_thread);
    quickSort(data, 0, data_size-1);
    double end_time_p = omp_get_wtime();

    double start_time = omp_get_wtime();
    omp_set_num_threads(1);
    quickSort(data2, 0, data_size-1);
    double end_time = omp_get_wtime();


    printf("\nSpeed up Rate: %lf\n", (end_time - start_time) / (end_time_p - start_time_p));
    return 0;
}