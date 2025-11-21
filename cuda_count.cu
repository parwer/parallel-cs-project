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

__global__ void parallel_hashtag_count(char **d_str, int *d_len, int numstr, char (*d_hashtags)[MAX_TAGS_LEN], int *d_tag_count, int *d_tags_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numstr)
    {
        char* s = d_str[idx];
        
        for (int i = 0; s[i] != '\0'; i++)
        {
            if (s[i] == '#')
            {
                char hashtag[MAX_TAGS_LEN];
                int tag_len = 0;
                for (int j = i+1; s[j] != '\0'; j++)
                {
                    if (!isValidHashtagChar(s[j]) || tag_len >= MAX_TAGS_LEN-1)
                    {
                        break;
                    }

                    hashtag[tag_len++] = s[j];
                }
                hashtag[tag_len] = '\0';

                // counting
                bool tag_exist = false;
                int existing_pos = -1;
                
                for (int attempt = 0; attempt < 2 && !tag_exist; attempt++)
                {
                    int current_count = *d_tags_count;
                    
                    for (int k = 0; k < current_count; k++)
                    {
                        // str cmp
                        bool match = true;
                        for (int l = 0; l < tag_len; l++)
                        {
                            if (hashtag[l] != d_hashtags[k][l])
                            {
                                match = false;
                                break;
                            }
                        }

                        if (match && d_hashtags[k][tag_len] == '\0')
                        {
                            existing_pos = k;
                            tag_exist = true;
                            break;
                        }
                    }
                    
                    if (!tag_exist && attempt == 0)
                    {
                        __threadfence();
                    }
                }

                if (tag_exist)
                {
                    atomicAdd(&d_tag_count[existing_pos], 1);
                }
                else
                {
                    int cur_len_tags = atomicAdd(d_tags_count, 1);
                    if (cur_len_tags < MAX_TAGS)
                    {
                        for (int l = 0; l < tag_len; l++)
                        {
                            d_hashtags[cur_len_tags][l] = hashtag[l];
                        }
                        d_hashtags[cur_len_tags][tag_len] = '\0';
                        
                        __threadfence();
                        d_tag_count[cur_len_tags] = 1;
                    }
                }
            }
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
    char **h_str = new char *[numstr];
    int *h_len = new int[numstr];
    int *h_offsets = new int[numstr];

    size_t offset = 0;
    for (int i = 0; i < numstr; i++)
    {
        h_len[i] = str[i].length();
        h_offsets[i] = offset;
        strcpy(h_buffer + offset, str[i].c_str());
        offset += h_len[i] + 1;
    }
    char h_hashtags[MAX_TAGS][MAX_TAGS_LEN];
    int h_tag_count[MAX_TAGS] = {0};
    int h_tags_count = 0;

    // Deivce Memory
    char *d_buffer;
    char **d_str;
    int *d_len;

    char (*d_hashtags)[MAX_TAGS_LEN];
    int *d_tag_count;
    int *d_tags_count;

    cudaMalloc(&d_buffer, total_size);
    cudaMalloc(&d_str, numstr * sizeof(char *));
    cudaMalloc(&d_len, numstr * sizeof(int));
    
    cudaMalloc(&d_hashtags, MAX_TAGS * MAX_TAGS_LEN * sizeof(char));
    cudaMalloc(&d_tag_count, MAX_TAGS * sizeof(int));
    cudaMalloc(&d_tags_count, sizeof(int));
    
    cudaMemcpy(d_buffer, h_buffer, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, h_len, numstr * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tag_count, h_tag_count, MAX_TAGS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tags_count, &h_tags_count, sizeof(int), cudaMemcpyHostToDevice);
    
    
    // measure time str cpy
    auto string_copy_start = chrono::high_resolution_clock::now();
    
    char **temp_str = new char*[numstr];
    for (int i = 0; i < numstr; i++)
    {
        temp_str[i] = d_buffer + h_offsets[i];
    }
    cudaMemcpy(d_str, temp_str, numstr * sizeof(char*), cudaMemcpyHostToDevice);

    auto string_copy_end = chrono::high_resolution_clock::now();
    cout << "String copy time: " << chrono::duration_cast<chrono::milliseconds>(string_copy_end - string_copy_start).count() << " ms\n";


    // kernel
    int threadsPerBlock = 512;
    int blocksPerGrid = (numstr + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    parallel_hashtag_count<<<blocksPerGrid, threadsPerBlock>>>(d_str, d_len, numstr, d_hashtags, d_tag_count, d_tags_count);
    // test_parallel<<<blocksPerGrid, threadsPerBlock>>>();
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

    cudaMemcpy(h_hashtags, d_hashtags, MAX_TAGS * MAX_TAGS_LEN * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tag_count, d_tag_count, MAX_TAGS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tags_count, d_tags_count, sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_tags_count && i < 20; i++)
    {
        cout << h_tag_count[i] << "\t:" << h_hashtags[i] << endl;
    }

    // // Cleanup
    delete[] h_buffer;
    delete[] h_str;
    delete[] h_len;
    delete[] h_offsets;
    delete[] temp_str;
    cudaFree(d_buffer);
    cudaFree(d_str);
    cudaFree(d_len);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}