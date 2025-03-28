#include "dcl.h"

using A_data_stream_type = hls::stream<data_t, 1>;
using A_index_stream_type = hls::stream<index_t, 1>;

using A_ent_stream_type = hls::stream<A_entries, 1>;


using B_data_stream_type = hls::stream<data_t, 1>;
using B_index_stream_type = hls::stream<index_t, 1>;

template <typename stream_type, typename vec_type, size_t stream_size_outer, size_t stream_size_inner>
void write_buffer(stream_type &dest, vec_type *input_vector) {
// #pragma HLS inline
    for (size_t i = 0; i < stream_size_outer; i++) {
        for(size_t j=0; j < stream_size_inner; j++)
        {
            dest.write(input_vector[i][j]);
        }
    }
}

template <typename des_type, typename vec_type, size_t stream_size_outer, size_t stream_size_inner>
void copy_buffer(des_type& dest, vec_type *input_vector) {
// #pragma HLS inline
    for (size_t i = 0; i < stream_size_outer; i++) {
        for(size_t j=0; j < stream_size_inner; j++)
        {
            dest[i][j] = input_vector[i][j];
        }
    }
}

template <typename stream_type, typename vec_type, size_t size>
void write_vec(stream_type &dest, vec_type *input_vector) {
// #pragma HLS inline
    for (size_t i = 0; i < size; i++) {
        dest.write(input_vector[i]);
    }
}

template <typename des_type, typename vec_type, size_t size>
void copy_vec(des_type* dest, vec_type *input_vector) {
// #pragma HLS inline
    for (size_t i = 0; i < size; i++) {
        dest[i]=input_vector[i];
    }
}

template <typename data_stream_type, typename index_stream_type, typename output_stream_type>
void accumulate_C(data_t value_A, data_stream_type& B_vals, index_stream_type& row_indices_stream, output_stream_type &output, index_t ith_ind)
{
    auto jth_ind = row_indices_stream.read();
    auto val_B = B_vals.read();
    auto out = value_A * val_B;
    output.write({out, ith_ind, jth_ind});
}

template <typename data_stream_type>
void sum_C(index_t inner_start, index_t inner_end, data_stream_type& C_data, data_t C[N][K])
{
    for(index_t i=inner_start; i < inner_end; i++)
    {
        auto entry = C_data.read();
        C[entry.ith_index][entry.jth_index] += entry.to_sum;
    }

}

// void read_A_cols()
// first we stream out the row pointers to index through the column indices and values of A
void read_A_rows(hls::vector<index_t, N+1>* local_row_ptr_A, hls::vector<data_t, N*M>* A_vals, hls::vector<index_t, N*M>* col_inds_A, A_ent_stream_type& stream_out)
{
    for(size_t i=0; i<N; i++)
    {
        for (index_t idx_A = (*local_row_ptr_A)[i]; idx_A < (*local_row_ptr_A)[i + 1]; idx_A++) {
            A_entries a = {(*A_vals)[idx_A], (*col_inds_A)[idx_A]};
            stream_out.write(a);
        }
    }
}


// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(hls::vector<data_t, N*M>* values_A, hls::vector<index_t, N*M>* column_indices_A, hls::vector<index_t, N + 1>* row_ptr_A,
                                hls::vector<data_t, M*K>* values_B, hls::vector<index_t, M*K>* row_indices_B, hls::vector<index_t, M + 1>* col_ptr_B,
                                hls::vector<data_t, K> C[N])
{
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1 depth=32
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1 depth=32
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1 depth=32

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2 depth=32

#pragma HLS interface m_axi port=C offset=slave bundle=mem3

#pragma HLS interface s_axilite port=return

    hls::vector<data_t, N*M> local_values_A;
    hls::vector<index_t, N*M> local_column_indices_A;

    A_ent_stream_type A_entry_stream;
    // A_index_stream_type local_column_indices_A;


    hls::vector<index_t, N+1> local_row_ptr_A;

    // B_data_stream_type local_values_B;
    // B_index_stream_type local_row_indices_B;
    // B_index_stream_type local_col_ptr_B;

    copy_vec<hls::vector<data_t, N*M>, hls::vector<data_t, N*M>, (N*M)>(&local_values_A, values_A);
    copy_vec<hls::vector<index_t, N*M>, hls::vector<index_t, N*M>, (N*M)>(&local_column_indices_A, column_indices_A);
    copy_vec<hls::vector<index_t, N+1>, hls::vector<index_t, N + 1>, (N+1)>(&local_row_ptr_A, row_ptr_A);

    read_A_rows(&local_row_ptr_A, &local_values_A, &local_column_indices_A, A_entry_stream);

    // copy_buffer<B_data_stream_type, hls::vector<data_t, K>, K, M>(local_values_B, values_B);
    // copy_buffer<B_index_stream_type, hls::vector<index_t, K>, K, M>(local_row_indices_B, row_indices_B);
    // copy_vec<B_index_stream_type, hls::vector<index_t, M + 1>, (M+1)>(local_col_ptr_B, col_ptr_B);

    data_t local_C[N][K] = {};

    // Perform Sparse x Sparse Multiplication
    // for (int i = 0; i < N; i++) {
    // // #pragma HLS unroll
    //     for (int idx_A = local_row_ptr_A[i]; idx_A < local_row_ptr_A[i + 1]; idx_A++) {
    //         int k = local_column_indices_A[idx_A]; // Column index of A
    //         data_t value_A = local_values_A[idx_A];

    //         // Iterate over columns of B corresponding to row k
    //         for (int idx_B = local_col_ptr_B[k]; idx_B < local_col_ptr_B[k + 1]; idx_B++) {
    //             int j = local_row_indices_B[idx_B]; // Row index of B
    //             data_t value_B = local_values_B[idx_B];

    //             // Accumulate the product into C[i][j]
    //             local_C[i][j] += value_A * value_B;
    //         }
    //     }
    // }

    // for(uint16_t i = 0; i<N; i++) {
    //     for(uint16_t j=0; j<K; j++) {
    //     #pragma HLS pipeline
    //         C[i][j] = local_C[i][j];
    //     }
    // }
}