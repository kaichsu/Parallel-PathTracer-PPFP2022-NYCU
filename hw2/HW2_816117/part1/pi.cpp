#include <iostream>
#include <string>
#include <pthread.h>
#include <random>

using namespace std;
int thread_n = 1;
long long int tosses  = 1e8;
long long int in_circle_cnt = 0;
void* ret;

void* Monto_Carlo_Pi(void* rank);

int main(int argc, char *argv[]){
    try{
        thread_n = stoi(argv[1]);
        tosses = stoll(argv[2]);
    } catch (...){
        cerr << "Usage: ./pi.out <Number of Threads> <Tosses>\n";
        exit(-1);
    }

    pthread_t tid[thread_n];

    for(int i=0; i<thread_n; ++i)
        pthread_create(&tid[i], NULL, Monto_Carlo_Pi, (void*) i);


    for(int i=0; i<thread_n; ++i){
        pthread_join(tid[i], &ret);
        in_circle_cnt += (long long int) ret;
    }

    cout << (double) in_circle_cnt * 4.0 / (double) tosses << "\n";

    return 0;
}

void* Monto_Carlo_Pi(void* rank){
    random_device rd;
    default_random_engine generator(rd());
    uniform_real_distribution<double> unir(-1.0, 1.0);
    
    long long samples = tosses / thread_n;
    long long in_circle_cnt = 0;

    for(long long i = 0; i < samples; ++i){
        double x = unir(generator);
        double y = unir(generator);
        double distance = x*x + y*y;
        if(distance <= 1){
            ++in_circle_cnt;
        }
    }
    return (void*) in_circle_cnt;
}
