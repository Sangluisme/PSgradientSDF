//============================================================================
// Name        : Timer.h
// Author      : Christiane Sommer
// Date        : 07/2018
// License     : GNU General Public License
// Description : simple class to time processes on the CPU
//============================================================================

#ifndef TIMER_H_
#define TIMER_H_

// includes
#include <iostream>
#include <ctime>
#include <string>
#include <omp.h>

class Timer {

private:
    double start_time;
    double end_time;
    double elapsed;

public:
    Timer() : start_time(0.), end_time(0.), elapsed(0.) {}
    ~Timer() {}
    
    void tic() {
        start_time = omp_get_wtime();
    }
    
    double toc(std::string s = "Time elapsed") {
        if (start_time!=0) {
            end_time = omp_get_wtime();
            elapsed = end_time-start_time;
            print_time(s);
        }
        else
            std::cout << "Timer was not started, no time could be measured." << std::endl;
        return elapsed;
    }
    
    void print_time(std::string s = "Time elapsed") {        
        if (elapsed<1.)
            std::cout << "---------- " << s << ": " << 1000.*elapsed << "ms." << std::endl;
        else
            std::cout << "---------- " << s << ": " << elapsed << "s." << std::endl;
    }

};

#endif // TIMER_H_
