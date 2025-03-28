
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <cstdint>
#include <ap_fixed.h>
#include <hls_math.h>
#include <hls_vector.h>
#include <hls_stream.h>
#include <stdlib.h>
#include <cstdint>

// typedef ap_fixed<16, 5> data_t;
#define DATA_WIDTH 16
#define DATA_INT 5

typedef ap_fixed<DATA_WIDTH, DATA_INT> data_t;
typedef ap_uint<16> index_t;


#define N 64  // Rows of A and C
#define M 64  // Columns of A and Rows of B
#define K 64  // Columns of B and C

// #if defined(SPARSE_1)
#define SPARSITY 0.1
// #endif

struct C_entry
{
    data_t to_sum;
    index_t ith_index;
    index_t jth_index;
};

struct A_entries
{
    data_t A_val;
    index_t A_col_index;
};

// hls::stream<A_entries, 1> 

// #if defined(SPARSE_2)
// #define SPARSITY 0.5
// #endif

// #if defined(SPARSE_3)
// #define SPARSITY 0.8
// #endif


// #define STRINGIFY(x) #x
// #define TO_STRING(x) STRINGIFY(x)

// #pragma message("sparsity: " TO_STRING(SPARSITY)) 

// void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
//                              data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K]);


void sparse_matrix_multiply_HLS(hls::vector<data_t, N*M>* values_A, hls::vector<index_t, N*M>* column_indices_A, hls::vector<index_t, N + 1>* row_ptr_A,
                                hls::vector<data_t, M*K>* values_B, hls::vector<index_t, M*K>* row_indices_B, hls::vector<index_t, M + 1>* col_ptr_B,
                                hls::vector<data_t, K> C[N]);

