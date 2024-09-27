

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BSIZE 1024

__global__ 
void vecSumKernel1(unsigned int *d_in, unsigned int *d_out, int size){
    int idx = threadIdx.x;

    if (idx < size){
      
        d_out += d_in[idx] ;
        __syncthreads();
        
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: <filename>\n");
        exit(-1);
    }
    unsigned int log2size, size;
    unsigned int *vec;
    FILE *f = fopen(argv[1], "r");
    fscanf(f, "%d\n", &log2size);
    if (log2size > 10) {
        printf("Size (%u) is too large: size is limited to 2^10\n", log2size);
        exit(-1);
    }
    size = 1 << log2size;
    unsigned int bytes = size * sizeof(unsigned int);
    vec = (unsigned int *) malloc(bytes);
    assert(vec);
    for (unsigned int i = 0; i < size; i++) {
        fscanf(f, "%u\n", &(vec[i]));
    }
    fclose(f);

    int dimgrid = 1;
    int dimblock = 1024;
    unsigned int *d_in;
    unsigned int *d_out;
    // int bytes = size * sizeof(unsigned int);

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, vec, bytes, cudaMemcpyHostToDevice);
    
    vecSumKernel1<<<dimgrid, dimblock>>>(d_in, d_out, size);

    cudaMemcpy(vec, d_out, bytes, cudaMemcpyDeviceToHost);

    printf("GPU= %u\n", vec[0]);
    cudaFree(d_in);
    cudaFree(d_out);
}
