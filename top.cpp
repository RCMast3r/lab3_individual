#include "dcl.h"
#include "hls_print.h"

void send_values(int i, int k, data_t value_A, const int col_ptr_B[M + 1], const int row_indices_B[M * K], const data_t values_B[M * K], hls::stream<PartialSum, N> partial_sums[N])
{
    for (int idx_B = col_ptr_B[k]; idx_B < col_ptr_B[k + 1]; idx_B++) {
    #pragma HLS pipeline
        int j = row_indices_B[idx_B]; // Row index of B
        data_t value_B = values_B[idx_B];

        // Create a PartialSum object with the column and value
        PartialSum ps;
        ps.col = j;                   // Target column in C
        ps.value = value_A * value_B; // Partial product

        // Push the partial product to the corresponding row stream
        partial_sums[i].write(ps);
    }
}

void handle_inner_loop(int i, const int row_ptr_A[N + 1], const int column_indices_A[N * M],
                              const data_t values_A[N * M], const int col_ptr_B[M + 1],
                              const int row_indices_B[M * K], const data_t values_B[M * K],
                              hls::stream<PartialSum, N> partial_sums[N])
{
    #pragma HLS dataflow

    for (int idx_A = row_ptr_A[i]; idx_A < row_ptr_A[i + 1]; idx_A++) {
        int k = column_indices_A[idx_A]; // Column index of A
        data_t value_A = values_A[idx_A];
        send_values(i, k, value_A, col_ptr_B, row_indices_B, values_B, partial_sums);
    }
}

void compute_partial_products(const int row_ptr_A[N + 1], const int column_indices_A[N * M],
                              const data_t values_A[N * M], const int col_ptr_B[M + 1],
                              const int row_indices_B[M * K], const data_t values_B[M * K],
                              hls::stream<PartialSum, N> partial_sums[N]) {

    // #pragma HLS INLINE off

    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL
        handle_inner_loop(i, row_ptr_A, column_indices_A, values_A, col_ptr_B, row_indices_B, values_B, partial_sums);
    }
}

size_t get_read_length(int i, const int row_ptr_A[N + 1], const int column_indices_A[N * M], const int col_ptr_B[M + 1])
{
    size_t read_length = 0;
    for (int idx_A = row_ptr_A[i]; idx_A < row_ptr_A[i + 1]; idx_A++) {
        int k = column_indices_A[idx_A]; // Column index of A
        for (int idx_B = col_ptr_B[k]; idx_B < col_ptr_B[k + 1]; idx_B++) {
        #pragma HLS pipeline
            read_length++;
        }
    }
    return read_length;
}

void accumulate_C_row(size_t i, size_t read_length, hls::stream<PartialSum, N> partial_sums[N], data_t C[N][K])
{
    for(size_t j=0; j < read_length; j++)
    {
        PartialSum ps = partial_sums[i].read();

        hls::print("row %d ", i);
        hls::print("col %d ", ps.col);
        hls::print("val %f\n", ps.value.to_float());
        C[i][ps.col] += ps.value; // Accumulate to the correct column
    }
}

// Function to accumulate results from streams into output matrix C
void accumulate_results(const int row_ptr_A[N + 1], const int column_indices_A[N * M], const int col_ptr_B[M + 1], hls::stream<PartialSum, N> partial_sums[N], data_t C[N][K]) {
    // #pragma HLS INLINE off
    #pragma HLS array partition variable=C type=complete dim=1 // for the unroll
    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL
        auto length = get_read_length(i, row_ptr_A, column_indices_A, col_ptr_B); 
        accumulate_C_row(i, length, partial_sums, C);
    }
}

void sparse_mult(const int row_ptr_A[N + 1], const int column_indices_A[N*M],
                 const data_t values_A[N*M], const int col_ptr_B[M+1],
                 const int row_indices_B[M*K], const data_t values_B[M*K],
                 data_t C[N][K]) {

    hls::stream<PartialSum, N> partial_sums[N];
    #pragma HLS ARRAY_PARTITION variable=partial_sums complete dim=1
    #pragma HLS DATAFLOW

    // Compute partial products and store in streams
    compute_partial_products(row_ptr_A, column_indices_A, values_A,
                             col_ptr_B, row_indices_B, values_B,
                             partial_sums);

    // Accumulate results from streams into C
    accumulate_results(row_ptr_A, column_indices_A, col_ptr_B, partial_sums, C);
}



void load_values_A(data_t* values_A, data_t* local_values_A) {
#pragma HLS inline
    for(uint16_t i = 0; i < (N * M); i++) {
#pragma HLS pipeline
        local_values_A[i] = values_A[i];
    }
}

void load_values_B(data_t* values_B, data_t* local_values_B) {
#pragma HLS inline
    for(uint16_t i = 0; i < (M * K); i++) {
#pragma HLS pipeline
        local_values_B[i] = values_B[i];
    }
}

void load_column_indices_A(int* column_indices_A, int* local_column_indices_A) {
#pragma HLS inline
    for(uint16_t i = 0; i < (N * M); i++) {
#pragma HLS pipeline
        local_column_indices_A[i] = column_indices_A[i];
    }
}

void load_row_indices_B(int* row_indices_B, int* local_row_indices_B) {
#pragma HLS inline
    for(uint16_t i = 0; i < (M * K); i++) {
#pragma HLS pipeline
        local_row_indices_B[i] = row_indices_B[i];
    }
}

void load_ptrs(int* row_ptr_A, int* col_ptr_B, int* local_row_ptr_A, int* local_col_ptr_B) {
#pragma HLS inline
    for(uint16_t i = 0; i < (N + 1); i++) {
#pragma HLS pipeline II=2
        local_row_ptr_A[i] = row_ptr_A[i];
        local_col_ptr_B[i] = col_ptr_B[i];
    }
}

void load_all(
    data_t* values_A, data_t* values_B,
    int* column_indices_A, int* row_indices_B,
    int* row_ptr_A, int* col_ptr_B,
    data_t local_values_A[N * M], data_t local_values_B[M * K],
    int local_column_indices_A[N * M], int local_row_indices_B[M * K],
    int local_row_ptr_A[N + 1], int local_col_ptr_B[M + 1]) {

#pragma HLS dataflow

    load_values_A(values_A, local_values_A);
    load_values_B(values_B, local_values_B);
    load_column_indices_A(column_indices_A, local_column_indices_A);
    load_row_indices_B(row_indices_B, local_row_indices_B);
    load_ptrs(row_ptr_A, col_ptr_B, local_row_ptr_A, local_col_ptr_B);
}


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

    data_t local_C[N][K] = {0};    

    data_t local_values_A[N * M] = {0};
    data_t local_values_B[M * K] = {0};

    int local_column_indices_A[N * M] = {0};
    int local_row_indices_B[M * K] = {0};
    int local_row_ptr_A[N+1];
    int local_col_ptr_B[M+1];
    // static_assert(((N*M) == (M*K))); 
    load_all(values_A, values_B, column_indices_A, row_indices_B,
             row_ptr_A, col_ptr_B,
             local_values_A, local_values_B,
             local_column_indices_A, local_row_indices_B,
             local_row_ptr_A, local_col_ptr_B);
    // for(uint16_t i = 0; i<(N*M); i++) {
    // #pragma HLS pipeline II=4
    //     local_values_A[i] = values_A[i];
    //     local_values_B[i] = values_B[i];
    //     local_column_indices_A[i] = column_indices_A[i];
    //     local_row_indices_B[i] = row_indices_B[i];   
    // }


    // for(uint16_t i=0; i<(N+1); i++)
    // {
    //     #pragma HLS pipeline II=2
    //     local_row_ptr_A[i]=row_ptr_A[i];
    //     local_col_ptr_B[i]=col_ptr_B[i];
    // }

    sparse_mult(local_row_ptr_A, local_column_indices_A, local_values_A, local_col_ptr_B, local_row_indices_B, local_values_B, local_C);



    // hls_thread_local hls::task agg_C(aggregate_C, ind_stream, local_C);
    
    for(uint16_t i = 0; i<N; i++) {
    #pragma HLS pipeline
        for(uint16_t j=0; j<K; j++) {
        #pragma HLS pipeline
            C[i][j] = local_C[i][j];
        }
    } 
}