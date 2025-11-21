#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cstring>
#include <unordered_map>
#include <sstream>
#include <chrono>


#define MAX_TAGS 50000
#define MAX_TAGS_LEN 64

using namespace std;

bool isValidHashtagChar(char c)
{
    return (c >= 'a' && c <= 'z') || 
           (c >= 'A' && c <= 'Z') || 
           (c >= '0' && c <= '9') || 
           c == '_' || c == '#';
}

void count_hashtag(char** h_str, int* h_len, int numstr, char hashtags[MAX_TAGS][MAX_TAGS_LEN], int tag_count[MAX_TAGS], int* tags_count)
{    
    int cht = 0;
    for (int i = 0; i < numstr; i++)
    {
        char* s = h_str[i];
        for (int j = 0; s[j] != '\0'; j++)
        {
            if (s[j] == '#')
            {
                char hashtag[MAX_TAGS_LEN];
                int tag_len = 0;
                for (int k = j+1; s[k] != '\0'; k++)
                {
                    if (!isValidHashtagChar(s[k]) || tag_len >= MAX_TAGS_LEN-1)
                    {
                        break;
                    }
                    hashtag[tag_len++] = s[k];
                }
                hashtag[tag_len] = '\0';

                // counting
                bool tag_exist = false;
                for(int l = 0; l < (*tags_count); l++)
                {
                    if (strcmp(hashtag, hashtags[l]) == 0)
                    {
                        tag_count[l]++;
                        tag_exist = true;
                        break;
                    }
                }

                // init new hashtag
                if (!tag_exist && (*tags_count) < MAX_TAGS)
                {
                    strcpy(hashtags[(*tags_count)], hashtag);
                    tag_count[(*tags_count)] = 1;
                    (*tags_count)++;
                }
            }
        }
    }
}


int main(int argc, char* argv[]) 
{
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


    int numstr = str.size();
    char** h_str = new char*[numstr];
    int* h_len = new int[numstr];

    for (int i = 0; i < numstr; i++)
    {
        h_len[i] = str[i].length();
        h_str[i] = new char[h_len[i]+1];
        strcpy(h_str[i], str[i].c_str());
    }

    char hashtags[MAX_TAGS][MAX_TAGS_LEN];
    int tag_count[MAX_TAGS] = {0};
    int tags_count = 0;

    auto start = chrono::high_resolution_clock::now();
    count_hashtag(h_str, h_len, numstr, hashtags, tag_count, &tags_count);
    auto end = chrono::high_resolution_clock::now();

    for (int i = 0; i < 20; i++)
    {
        cout << tag_count[i] << "\t:" << hashtags[i] << endl;
    }

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Time: " << elapsed.count() << " ms\n";
}