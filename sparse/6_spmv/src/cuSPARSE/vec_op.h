#ifndef VEC_OP_H
#define VEC_OP_H

#include <iostream>
#include <fstream>

#define ERROR_THRESHOLD 1e-6

using namespace std;

// write a vector to file
template <typename T>
void vec_write_file(T *vec, int n, char *filename)
{
    ofstream destFile(filename, ios::out);
    for (size_t i = 0; i < n; i++)
    {
        destFile << vec[i] << endl;
    }
    destFile.close();
}

// read a vector from file
template <typename T>
void vec_read_file(T *vec, int n, char *filename)
{
    ifstream srcFile(filename, ios::in);
    for (size_t i = 0; i < n; i++)
    {
        srcFile >> vec[i];
    }
    srcFile.close();
}

// compare two vector, return 1 if they are equal, else return 0
template <typename T1,typename T2>
int vec_compare(T1 *vec_a, T2 *vec_b, int n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (vec_a[i] - vec_b[i] > ERROR_THRESHOLD)
        {
            return 0;
        }
    }
    return 1;
}

#endif