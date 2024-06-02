#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <mpi.h>
#include <chrono>
#include <string>
#include <iomanip>
#include <sstream>

#define BLOCKSIZE 16

int IntializingMatrixVectors(float **, float **, float **, int , int , int , int);
int CheckCudaDevice(int );
void printMatrix(float *, int , int, const char*);
float abs(float, float);

#define SAFE(call)                                                         \
            do{                                                                      \
                 cudaError_t err = call;                                             \
                 if(err != cudaSuccess)                                              \
                 {                                                                   \
                        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                         __FILE__, __LINE__, cudaGetErrorString( err) );             \
                         exit(1);                                                    \
                 }                                                                   \
               } while (0)                                                           \

__global__ void MatrixMultiplication(float *MA, float *MB, float *Res, int r1, int c2, int cut_size, int id) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < r1 && col < c2) {
        float sum = 0.0;
        for (int k = 0; k < cut_size; ++k) {
            sum += MA[row * cut_size + k] * MB[k * c2 + col];
        }
        Res[row * c2 + col] = sum;
    }
}

int IntializingMatrixVectors(float **MA, float **MB, float **ResultM, int R1, int C1, int R2, int C2){
    float *TempMA, *TempResultM, *TempMB;
    int Status = 1;

    TempMA = (float *)malloc(R1 * C1 * sizeof(float));
    if(TempMA == NULL) Status = 0;

    TempMB = (float *)malloc(R2 * C2 * sizeof(float));
    if(TempMB == NULL) Status = 0;

    TempResultM = (float *)malloc(R1 * C2 * sizeof(float));
    if(TempResultM == NULL) Status = 0;

    int limit = 10;

    for(int i = 0; i < R1 * C1; i++)
        TempMA[i] = (float)rand() / (float)(RAND_MAX / limit);

    for(int i = 0; i < R2 * C2; i++)
        TempMB[i] = (float)rand() / (float)(RAND_MAX / limit);

    for(int i = 0; i < R1 * C2; i++)
        TempResultM[i] = 0.0f;

    *MA = TempMA;
    *MB = TempMB;
    *ResultM = TempResultM;

    return Status;
}

int CheckCudaDevice(int id) {
    int DeviceCount, Device;
    struct cudaDeviceProp Properties;

    cudaGetDeviceCount(&DeviceCount);
    if(DeviceCount >= 1) {
        cudaGetDevice(&Device);
        cudaGetDeviceProperties(&Properties, Device);
        printf("Processor with rank %d has the Device by name %s and computation is done on this device \n", id, Properties.name);
    } else {
        printf("Processor with rank %d found no CUDA device.\n", id);
        return 0;
    }
    return DeviceCount;
}

void printMatrix(float *M, int R, int C, const char *filename) {
    FILE *file = fopen(filename, "a");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            fprintf(file, "%f", M[i * C + j]);
            if (j < C - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

float abs(float a, float b) {
    return (a >= b) ? a - b : b - a;
}

std::string getCurrentTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm;
    localtime_r(&now_time_t, &now_tm);
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

double calculate_mult_time(float *MA, float *MB, float *RM, int R1, int C1, int R2, int C2) {
    double s_time = MPI_Wtime();
    for (int i = 0; i < R1; i++) {
        for (int j = 0; j < C2; j++) {
            float sum = 0.0;
            for (int k = 0; k < R2; k++) sum += MA[i * C1 + k] * MB[k * C2 + j];
            RM[i * C2 + j] = sum;
        }
    }
    double e_time = MPI_Wtime();
    return e_time - s_time;
}

int main(int argc, char **argv) {
    int id, num_proc;
    int Root = 0, Status = 1;
    float *MatrixA, *MatrixB, *ResultM, *RM;
    float *Cut_MatrixA, *Cut_ResultM, *Cut_MatrixB;
    float *CudaMA, *CudaRM, *CudaMB;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    if (argc < 4) {
        if (id == Root) printf("Invalid input args\n");
        MPI_Finalize();
        exit(-1);
    }

    int R1 = atoi(argv[1]), C1 = atoi(argv[2]), R2 = atoi(argv[3]), C2 = atoi(argv[4]);

    if (C1 != R2) {
        if (id == Root) std::cout << "Entered wrong input, Number of columns of matrix 1 should be equal to number of rows of matrix 2 " << std::endl;
        MPI_Finalize();
        exit(-1);
    }

    if (R1 < num_proc) {
        if (id == Root) std::cout << "Given number of Rows of the matrix should be more than number of processors" << std::endl;
        MPI_Finalize();
        exit(-1);
    }

    if (R1 % num_proc != 0 || C2 % num_proc != 0) {
        if (id == Root) std::cout << "The Rows of the matrixA or the columns of the matrixB cannot be distributed evenly among processors " << std::endl;
        MPI_Finalize();
        exit(-1);
    }

    if (id == Root) Status = IntializingMatrixVectors(&MatrixA, &MatrixB, &ResultM, R1, C1, R2, C2);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&Status, 1, MPI_INT, Root, MPI_COMM_WORLD);

    if (id != Root) MatrixB = (float *)malloc(R2 * C2 * sizeof(float));

    int cut_size = C1 / num_proc;

    Cut_MatrixA = (float *)malloc(cut_size * R1 * sizeof(float));
    Cut_MatrixB = (float *)malloc(cut_size * C2 * sizeof(float));
    if (Cut_MatrixA == NULL) Status = 0;
    if (Cut_MatrixB == NULL) Status = 0;

    Cut_ResultM = (float *)malloc(R1 * C2 * sizeof(float));
    if (Cut_ResultM == NULL) Status = 0;

    MPI_Scatter(MatrixB, cut_size * C2, MPI_FLOAT, Cut_MatrixB, cut_size * C2, MPI_FLOAT, Root, MPI_COMM_WORLD);

    int *sendcounts = (int *)malloc(num_proc * sizeof(int));
	int *displacements = (int *)malloc(num_proc * sizeof(int));
	
	if (id == 0) {
        for (int i = 0; i < num_proc; i++) {
            sendcounts[i] = cut_size;
            displacements[i] = i * cut_size;
        }
    }

    for(int i = 0; i < R1; i++)
		MPI_Scatterv(MatrixA+i*C1, sendcounts, displacements, MPI_FLOAT, Cut_MatrixA+i*cut_size, cut_size*num_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (CheckCudaDevice(id) == 0) {
        std::cout << "No CUDA device is found! Using CPU only" << std::endl;

        for (int i = 0; i < cut_size; i++) {
            Cut_ResultM[i] = 0;
            int indexs = i * C1;
            for (int col = 0; col < C1; col++) {
                Cut_ResultM[i] += (Cut_MatrixA[indexs] * MatrixB[col]);
                indexs++;
            }
        }
    } else {
        cudaSetDevice(id);
        SAFE(cudaMalloc((void **)&CudaMA, cut_size * R1 * sizeof(float)));
        SAFE(cudaMalloc((void **)&CudaMB, cut_size * C2 * sizeof(float)));
        SAFE(cudaMalloc((void **)&CudaRM, R1 * C2 * sizeof(float)));

        SAFE(cudaMemcpy((void *)CudaMA, (void *)Cut_MatrixA, cut_size * R1 * sizeof(float), cudaMemcpyHostToDevice));
        SAFE(cudaMemcpy((void *)CudaMB, (void *)Cut_MatrixB, cut_size * C2 * sizeof(float), cudaMemcpyHostToDevice));

        double s_time = MPI_Wtime();

        dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
        dim3 gridSize((C2 + BLOCKSIZE - 1) / BLOCKSIZE, (R1 + BLOCKSIZE - 1) / BLOCKSIZE);
        MatrixMultiplication<<<gridSize, blockSize>>>(CudaMA, CudaMB, CudaRM, R1, C2, cut_size, id);
        SAFE(cudaDeviceSynchronize());

        SAFE(cudaMemcpy(Cut_ResultM, CudaRM, R1 * C2 * sizeof(float), cudaMemcpyDeviceToHost));

        MPI_Barrier(MPI_COMM_WORLD);
	    MPI_Reduce(Cut_ResultM, ResultM, R1 * C2, MPI_FLOAT, MPI_SUM, Root, MPI_COMM_WORLD);
        double e_time = MPI_Wtime();

        if (id == Root) {
            RM = (float *)malloc(R1 * C2 * sizeof(float));
            double cpu_time = calculate_mult_time(MatrixA, MatrixB, RM, R1, C1, R2, C2);

            bool IfTrue = true;
            for (int i = 0; i < R1 * C2; i++) {
                if (abs(RM[i], ResultM[i]) > 0.01) {
                    IfTrue = false;
                    printf("%f, %f", RM[i], ResultM[i]);
                    break;
                }
            }
            printf("Result are %s\n", IfTrue ? "true" : "false");

            // FILE *file = fopen("./matrix_record.txt", "a");
            // std::string currentTime = getCurrentTime();
            // fprintf(file, "Current Time: %s\n", currentTime.c_str());
            // fprintf(file, "Matrix A:\n");
            // printMatrix(MatrixA, R1, C1, "./matrix_record.txt");
            // fflush(stdout); 
            // fprintf(file, "Matrix B:\n");
            // printMatrix(MatrixB, R2, C2, "./matrix_record.txt");
            // fflush(stdout); 
            // fprintf(file, "CPU Matrix R:\n");
            // printMatrix(RM, R1, C2, "./matrix_record.txt");
            // fflush(stdout); 
            // fprintf(file, "CUDA Matrix R:\n");
            // printMatrix(ResultM, R1, C2, "./matrix_record.txt");
            // fflush(stdout); 
            std::cout << "Time for CPU:" << cpu_time << std::endl;
            std::cout << "Time for GPU:" << e_time - s_time << std::endl;
            std::cout << "Speed up rate:" << cpu_time / (e_time - s_time) << std::endl;
            // fclose(file);
        }
    }

    free(Cut_MatrixA);
    free(Cut_MatrixB);
    free(Cut_ResultM);

    SAFE(cudaFree(CudaMA));
    SAFE(cudaFree(CudaMB));
    SAFE(cudaFree(CudaRM));

    MPI_Finalize();
    return 0;
}
