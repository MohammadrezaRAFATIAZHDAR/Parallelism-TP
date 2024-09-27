void vecAdd(int *h_in, int *h_out, int n){
    
    int dimgrid = 1;
    int dimblock = 1024;
    int *d_in;
    int *d_out;
    int size = n * sizeof(int)

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    vecSumKernel1<<<dimgrid, dimblock>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

--global--
void vecSumKernel1(int *d_in, int d_out, int n){
    int idx = threadIdx.x;

    if (idx < n){
      
        d_out[idx] += d_in[idx];
        
    }
}