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

void count_hashtag(char** h_str, int* h_len, int numstr)
{    
    int cht = 0;
    for (int i = 0; i < numstr; i++)
    {
        char* s = h_str[i];
        for (int j = 0; s[j] != '\0'; j++)
        {
            if (s[j] == '#')
            {
                cht++;
            }
        }
    }

    cout << "All hashtags count: " << cht << endl;

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


    auto start = chrono::high_resolution_clock::now();
    count_hashtag(h_str, h_len, numstr);
    auto end = chrono::high_resolution_clock::now();


    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Time: " << elapsed.count() << " ms\n";
}