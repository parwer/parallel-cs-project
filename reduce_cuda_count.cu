#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>

#define MAX_TAGS 50000
#define MAX_TAGS_LEN 64

using namespace std;

__device__ bool isValidHashtagChar(char c)
{
    return (c >= 'a' && c <= 'z') || 
           (c >= 'A' && c <= 'Z') || 
           (c >= '0' && c <= '9') || 
           c == '_' || c == '#';
}

__global__ void find_hashtag_positions(char* d_buffer, int buffer_size, int* d_hashtag_positions, int *d_hashtag_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < buffer_size && d_buffer[idx] == '#')
    {
        int cur_len_tag = atomicAdd(d_hashtag_count, 1);
        d_hashtag_positions[cur_len_tag] = idx;
    }
}


__global__ void parallel_hashtag_extracting(char* d_buffer, int buffer_size,
                                            int *d_hashtag_positions, 
                                            int hashtag_count, 
                                            char (*d_hashtags)[MAX_TAGS_LEN])
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < hashtag_count)
    {
        int start_position = d_hashtag_positions[idx];
        int tag_len = 0;

        for (int i = start_position; i < buffer_size && tag_len < MAX_TAGS_LEN-1; i++)
        {
            if (!isValidHashtagChar(d_buffer[i]) || d_buffer[i] == '\n' || (d_buffer[i] == '#' && i != start_position))
            {
                break;
            }
            d_hashtags[idx][tag_len++] = d_buffer[i];
        }
        d_hashtags[idx][tag_len] = '\0';
    }
}

__global__ void unique_count_hashtags(char (*d_hashtags)[MAX_TAGS_LEN], int hashtag_count, 
                                        char (*d_unique_hashtags)[MAX_TAGS_LEN],
                                        int *d_unique_counts,
                                        int *d_unique_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < hashtag_count)
    {
        char *hashtag = d_hashtags[idx];
        int tag_len = 0;
        while (hashtag[tag_len] != '\0' && tag_len < MAX_TAGS_LEN)
        {
            tag_len++;
        }

        if (tag_len <= 1)
        {
            return;
        }

        // tag exist?
        bool tag_exist = false;
        for (int i = 0; i < *d_unique_count; i++)
        {
            bool match = true;
            for (int j = 0; j < tag_len; j++)
            {
                if (hashtag[j] != d_unique_hashtags[i][j])
                {
                    match = false;
                    break;
                }
            }

            if (match && d_unique_hashtags[i][tag_len] == '\0')
            {
                atomicAdd(&d_unique_counts[i], 1);
                tag_exist = true;
                break;
            }
        }

        // init new hashtag
        if (!tag_exist)
        {
            int cur_len_tag = atomicAdd(d_unique_count, 1);
            if (cur_len_tag < MAX_TAGS)
            {
                for (int j = 0; j < tag_len; j++)
                {
                    d_unique_hashtags[cur_len_tag][j] = hashtag[j];
                }
            }
            d_unique_hashtags[cur_len_tag][tag_len] = '\0';
            d_unique_counts[cur_len_tag] = 1;
        }
    }
}

int main(int argc, char* argv[]) 
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    string filename = "content.txt";
    if (argc > 1)
    {
        filename = argv[1];
    }
    else
    {
        cout << "Usage: " << argv[0] << " <filename>" << endl;
        cout << "Using default file: " << filename << endl;
    }

    ifstream inputFile(filename);
    vector<string> str;

    if (inputFile.is_open())
    {
        string line;
        while (getline(inputFile, line))
        {
            str.push_back(line);
        }

        inputFile.close();
    }
    else
    {
        cerr << "cant open file" << endl;
    }

    // convert vector to array
    int numstr = str.size();

    size_t total_size = 0;
    for (int i = 0; i < numstr; i++)
    {
        total_size += str[i].length() + 1;
    }

    char *h_buffer = new char[total_size];
    int *h_len = new int[numstr];
    int *h_offsets = new int[numstr];
    int h_hashtag_count = 0;

    size_t offset = 0;
    for (int i = 0; i < str.size(); i++)
    {
        strcpy(h_buffer + offset, str[i].c_str());
        offset += str[i].length();
        h_buffer[offset++] = '\n';
    }
    h_buffer[total_size-1] = '\0';

    cout << "Total buffer size: " << total_size << " bytes" << endl;



    // Deivce Memory
    char *d_buffer;
    int *d_hashtag_positions;
    int *d_hashtag_count;
    char (*d_hashtags)[MAX_TAGS_LEN];
    char (*d_unique_hashtags)[MAX_TAGS_LEN];
    int *d_unique_counts;
    int *d_unique_count;

    cudaMalloc(&d_buffer, total_size);
    cudaMalloc(&d_hashtag_positions, MAX_TAGS * sizeof(int));
    cudaMalloc(&d_hashtag_count, sizeof(int));
    cudaMalloc(&d_hashtags, MAX_TAGS * MAX_TAGS_LEN * sizeof(char));
    cudaMalloc(&d_unique_hashtags, MAX_TAGS * MAX_TAGS_LEN * sizeof(char));
    cudaMalloc(&d_unique_counts, MAX_TAGS * sizeof(int));
    cudaMalloc(&d_unique_count, sizeof(int));
    
    cudaMemcpy(d_buffer, h_buffer, total_size, cudaMemcpyHostToDevice);
    cudaMemset(d_hashtag_count, 0, sizeof(int));
    cudaMemset(d_unique_count, 0, sizeof(int));
    cudaMemset(d_unique_counts, 0, MAX_TAGS * sizeof(int));
    
    int threadsPerBlock = 512;
    
    cudaEventRecord(start);

    int blocksPerGrid1 = (total_size + threadsPerBlock - 1) / threadsPerBlock;
    find_hashtag_positions<<<blocksPerGrid1, threadsPerBlock>>>(d_buffer, total_size, d_hashtag_positions, d_hashtag_count);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_hashtag_count, d_hashtag_count, sizeof(int), cudaMemcpyDeviceToHost);

    int blocksPerGrid2 = (h_hashtag_count + threadsPerBlock - 1) / threadsPerBlock;
    parallel_hashtag_extracting<<<blocksPerGrid2, threadsPerBlock>>>(d_buffer, total_size, d_hashtag_positions, h_hashtag_count, d_hashtags);
    cudaDeviceSynchronize();


    unique_count_hashtags<<<blocksPerGrid2, threadsPerBlock>>>(d_hashtags, h_hashtag_count, d_unique_hashtags, d_unique_counts, d_unique_count);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Kernel execution time: " << ms << " ms\n";

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        cerr << "CUDA error: " << cudaGetErrorString(error) << endl;
    }


    char h_unique_hashtags[MAX_TAGS][MAX_TAGS_LEN];
    int h_unique_counts[MAX_TAGS];
    int h_unique_count = 0;

    cudaMemcpy(h_unique_hashtags, d_unique_hashtags, MAX_TAGS * MAX_TAGS_LEN * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_unique_counts, d_unique_counts, MAX_TAGS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_unique_count, d_unique_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < h_unique_count && i < 20; i++)
    {
        cout << h_unique_counts[i] << "\t: " << h_unique_hashtags[i] << endl;
    }

    // // Cleanup
    delete[] h_buffer;
    cudaFree(d_buffer);
    cudaFree(d_hashtag_positions);
    cudaFree(d_hashtag_count);
    cudaFree(d_hashtags);
    cudaFree(d_unique_hashtags);
    cudaFree(d_unique_counts);
    cudaFree(d_unique_count);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}