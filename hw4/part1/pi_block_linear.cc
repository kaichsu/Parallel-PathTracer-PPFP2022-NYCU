#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

long long Monto_Carlo_Pi(int tosses, int rank){
    unsigned int seed = rank;
    long long  samples = tosses;
    long long  in_circle_cnt = 0;
    for(long long  i = 0; i < samples; ++i){
        double x = 2.0 * rand_r(&seed) / (RAND_MAX + 1.0) - 1.0;
        double y = 2.0 * rand_r(&seed) / (RAND_MAX + 1.0) - 1.0;
        in_circle_cnt += (x*x + y*y <= 1);
    }
    return in_circle_cnt;
}


int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: init MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int tag = 0;
    long long total_count = 0;

    if (world_rank > 0)
    {
        // TODO: handle workers
        long long count = Monto_Carlo_Pi(tosses/world_size, world_rank);
        MPI_Send(&count, 1, MPI_LONG_LONG, 0, tag, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: master
        total_count = Monto_Carlo_Pi(tosses/world_size, world_rank);
        for(int i=0; i<world_size-1; ++i){
            long long count = 0;
            MPI_Status status;
            MPI_Recv(&count, 1, MPI_LONG_LONG, i+1, tag, MPI_COMM_WORLD, &status);
            total_count += count;
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        pi_result = (double) total_count * 4.0 / (double) tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
