#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void
upsweep_parallel_for(int* input, int N, int* result, int two_d) {
    // figure out thread index, block index
    int two_dplus1 = 2*two_d;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    index = index * two_dplus1;

    // printf("value at index: %d equals: %d \n", index, result[index]);
    // printf("value at index: %d equals: %d \n", index+1, result[index+1]);

    result[index + two_dplus1 - 1] += result[index + two_d - 1];
    // printf("post-upsweep: value at index: %d equals: %d \n", index + two_dplus1 - 1, result[index + two_dplus1 - 1]);
}

__global__ void
downsweep_parallel_for(int* input, int N, int* result, int two_d) {
    // figure out thread index, block index
    int two_dplus1 = 2*two_d;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    index = index * two_dplus1;

    // printf("initial downsweep: value at index: %d equals: %d \n", index, result[index]);
    int t = result[index+two_d-1];
    result[index+two_d-1] = result[index+two_dplus1-1];
    result[index+two_dplus1-1] += t;
    // if (index == 0) {
    //     printf("two_d: %d, set index: %d to: %d \n", two_d, index+two_d-1, result[index+two_d-1]);
    // }
}


// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result

void exclusive_scan(int* input, int N, int* result)
{
    int rounded_length = nextPow2(N);
    cudaMemset(&result[N], 0, sizeof(float) * (rounded_length - N) );

    N = rounded_length;

    const int threadsPerBlock = min(N, 512);

    // upsweep
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        int totalThreads = (N + two_dplus1 - 1) / two_dplus1;
        int blocks_iter = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
        upsweep_parallel_for<<<blocks_iter, min(totalThreads, threadsPerBlock)>>>(input, N, result, two_d);
        cudaDeviceSynchronize();
    }

    cudaMemset(&result[N-1], 0, sizeof(float));

    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        int totalThreads = (N + two_dplus1 - 1) / two_dplus1;
        int blocks_iter = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
        // printf("starting a new iteration with two_d %d, thread per block %d, block iters %d \n", two_d, threadsPerBlock, blocks_iter);
        downsweep_parallel_for<<<blocks_iter, min(totalThreads, threadsPerBlock)>>>(input, N, result, two_d);
        cudaDeviceSynchronize();
    }
}



//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}

__global__ void
add_repeated_entry_to_array(int* differenceResult, int* scanResult, int N, int* result, int* size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nextIndexIsDuplicate = differenceResult[index];
    if (index > (N - 1)) {
        return;
    }
    if (nextIndexIsDuplicate == 1) {
        // printf("value at scan result index: %d equals: %d \n", scanResult[index+1], index);
        result[scanResult[index+1] - 1] = index;
    }
    if (index == (N-1)) {
        *size = scanResult[index];
        // printf("set size to: %d \n", *size);
    }
    return;
}

__global__ void
set_one_if_next_equal(int* input, int N, int* result, int* result2) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == N-1) {
        result[index] = 0;
        result2[index] = 0;
    }
    else {
        if (input[index] == input[index+1]) {
            result[index] = 1;
            result2[index] = 1;
        }
        else {
            result[index] = 0;
            result2[index] = 0;
        }
    }
}

void printCudaArray(int* d_array, int n) {
    // Allocate memory on the host
    int* h_array = new int[n];

    // Copy data from device (GPU) to host (CPU)
    cudaMemcpy(h_array, d_array, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the array
    for (int i = 0; i < n; ++i) {
        std::cout << "index: " << i << ", value: " << h_array[i] << "\n";
    }
    std::cout << std::endl;

    // Free host memory
    delete[] h_array;
}./render -r cpuref snow

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    // {0,1,1,3,3,5,5,4} --> take the differences
    // {0,1,0,1,0,1,0} --> reduce these differences to output
    // {0,0,1,1,2,2,3} --> exclusive scan : name this scan_res
    // allocate array of size == scan_res[N-1]
    // kick off job to write to the relevant index 

    // result to return:
    // { 1, 3, 5}

    int* diff_result;
    int* scan_result;
    int* size;
    int sizeToReturn;

    // printCudaArray(device_input, length);

    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&scan_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&diff_result, sizeof(int) * rounded_length);
    cudaMalloc(&size, sizeof(int));

    const int threadsPerBlock = min(length, 512);
    const int numBlocks = (length + threadsPerBlock - 1) / threadsPerBlock;

    set_one_if_next_equal<<<numBlocks, threadsPerBlock>>>(device_input, length, diff_result, scan_result);

    // printCudaArray(diff_result, length);

    cudaDeviceSynchronize();
    exclusive_scan(diff_result, length, scan_result);
    cudaDeviceSynchronize();
    add_repeated_entry_to_array<<<numBlocks, threadsPerBlock>>>(diff_result, scan_result, length, device_output, size);
    cudaDeviceSynchronize();
    cudaMemcpy(&sizeToReturn, size, sizeof(int), cudaMemcpyDeviceToHost);

    // printCudaArray(device_output, sizeToReturn);
    return sizeToReturn;
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
