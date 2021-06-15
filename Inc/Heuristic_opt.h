//
// Created by 翰樵 on 2021/5/7.
//

#ifndef OPERATIONRESEARCH_HEURISTIC_OPT_H
#define OPERATIONRESEARCH_HEURISTIC_OPT_H
#include "OptCore.h"
using namespace autodiff;
using namespace Eigen;
namespace heuOpt{
    class simulated_Annealing: public Optimizer{
    private:
        int iteration_max;
        double T_START;//Initialize the temperature;
        double Tmp;//Current temperature;
        double des;//decide the rate of descending;
        double k0;
        double k1;
        int cnt ;//record stop times;
        double warmUp_rate;
        double T_END;
        int stayTimes;
        int fireUp_times;
        double y, tempY;

        bool trapped_in_Local() const;

        inline void coolDown();
        inline double warmUp();
        inline void wander(double tmp);


    public:
        simulated_Annealing(double(*_fct)(VectorXd &), VectorXd &initialX, int it_max=200, double DescentRate=0.995);
        void Forward();
        void Stay();
        bool Metropolis(double n);
        double Evaluate(VectorXd &tempX);
        bool solve();

};
}


#endif //OPERATIONRESEARCH_HEURISTIC_OPT_H
