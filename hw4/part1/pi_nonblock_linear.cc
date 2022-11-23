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
    long long total_count = Monto_Carlo_Pi(tosses/world_size, world_rank);

    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Request req;
        MPI_Isend(&total_count, 1, MPI_LONG_LONG, 0, tag, MPI_COMM_WORLD, &req);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size-1];
        MPI_Status status[world_size-1];
        long long count[world_size-1];
        for(int i=0;i<world_size-1; ++i){
            MPI_Irecv(&count[i], 1, MPI_LONG_LONG, i+1, tag, MPI_COMM_WORLD, &requests[i]);
        }
        
        MPI_Waitall(world_size-1, requests, status);
        for(int i=0;i<world_size-1; ++i){
            total_count += count[i];
        }
    }

    if (world_rank == 0)
    {
        // TODO: PI result
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
