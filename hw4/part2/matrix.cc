#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define MASTER 0

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int n, m , l;
    if(world_rank == MASTER){
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
        n = *n_ptr;
        m = *m_ptr;
        l = *l_ptr;
        *a_mat_ptr = (int *)malloc(sizeof(int) * n * m);
        *b_mat_ptr = (int *)malloc(sizeof(int) * m * l);
        for(int y=0; y<n; ++y){
            for(int x=0; x<m; ++x){
                scanf("%d", *a_mat_ptr + y * m + x);
            }
        }
        for(int y=0; y<m; ++y){
            for(int x=0; x<l; ++x){
                scanf("%d", *b_mat_ptr + y * l + x);
            }
        }
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&l, 1, MPI_INT, 0, MPI_COMM_WORLD);
    *n_ptr = n;
    *m_ptr = m;
    *l_ptr = l;
    if(world_rank != MASTER){
        *a_mat_ptr = (int *)malloc(sizeof(int) * n * m );
        *b_mat_ptr = (int *)malloc(sizeof(int) * m * l );
    }
    MPI_Bcast(*a_mat_ptr, n*m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, m*l, MPI_INT, 0, MPI_COMM_WORLD);
}

// Just matrix multiplication (your should output the result in this function)
//
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    for(int y=0; y<n; ++y){
        int local[l];
        int ans[l];
        for(int x=0; x<l; ++x){
            local[x] = 0;
            if(x % world_size == world_rank){
                for(int i=0; i<m; ++i){
                    local[x] += a_mat[y*m + i] * b_mat[i*l + x];
                }
            }
        }

        MPI_Reduce(local, ans, l, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(world_rank == MASTER){
            for(int i=0; i<l; ++i){
                printf("%d ", ans[i]);
            }
            printf("\n");
        }
    }
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    int world_size, world_rank; 
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    free(a_mat);
    free(b_mat);
}