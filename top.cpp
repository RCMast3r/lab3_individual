#include "dcl.h"


// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K]) 
{
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2

#pragma HLS interface m_axi port=C offset=slave bundle=mem3

#pragma HLS interface s_axilite port=return

    data_t local_values_A[N * M];
    data_t local_values_B[M * K];

    int local_column_indices_A[N * M];
    int local_row_indices_B[M * K];

    static_assert(((N*M) == (M*K))); 
    for(uint16_t i = 0; i<(N*M); i++) {
    #pragma HLS pipeline
        local_values_A[i] = values_A[i];
        local_values_B[i] = values_B[i];
        local_column_indices_A[i] = column_indices_A[i];
        local_row_indices_B[i] = row_indices_B[i];   
    }
    data_t local_C[N][K];
    // data_t local_C[N][K];

    // Perform Sparse x Sparse Multiplication
    for (int i = 0; i < N; i++) {
        #pragma HLS unroll
        for (int idx_A = row_ptr_A[i]; idx_A < row_ptr_A[i + 1]; idx_A++) {
            int k = local_column_indices_A[idx_A]; // Column index of A
            data_t value_A = local_values_A[idx_A];

            // Iterate over columns of B corresponding to row k
            for (int idx_B = col_ptr_B[k]; idx_B < col_ptr_B[k + 1]; idx_B++) {
                int j = local_row_indices_B[idx_B]; // Row index of B
                data_t value_B = local_values_B[idx_B];

                // Accumulate the product into C[i][j]
                local_C[i][j] += value_A * value_B;
            }
        }
    }

    for(uint16_t i = 0; i<N; i++) {
        for(uint16_t j=0; j<K; j++) {
        #pragma HLS pipeline
            C[i][j] = local_C[i][j];
        }
    } 
}