#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

const double tolerance = 0.000001f;

double Monto_Carlo_Pi(){
    unsigned long long sample_cnt = 0;
    unsigned long long in_circle_cnt = 0;
    double estimate_pi;

    srand( time(NULL) );
    for(sample_cnt = 0; sample_cnt < 5000000; ++sample_cnt)
    {
        double x = (double) rand() * 2 / RAND_MAX - 1;
        double y = (double) rand() * 2 / RAND_MAX - 1;
        double distance = x*x + y*y;
        if(distance <= 1){
            ++in_circle_cnt;
        }    
    }
    estimate_pi = (double) in_circle_cnt * 4.0 / (double) sample_cnt;
    return estimate_pi;
}


int main(){
    // for(int i=0; i<10; ++i) {
    //     cout << (double) rand() * 2 / RAND_MAX - 1 << "\n";
    // }
    cout << Monto_Carlo_Pi() << endl;
    return 0;
}