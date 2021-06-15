//
// Created by 翰樵 on 2021/5/7.
//
#include "Inc/Heuristic_opt.h"
#define EPS 1e-9
#define FOR_DEBUG 0
using namespace heuOpt;

simulated_Annealing::simulated_Annealing(double(*_fct)(VectorXd &), VectorXd &initialX, int it_max, double DescentRate){
    Obj_Fct = _fct;
    X = initialX;
    X_next = X;
    y = (*Obj_Fct)(X);
    tempY = 0;
    dimension = X.rows();
    it_counter = 0;

    iteration_max = it_max;
    des = DescentRate;
    T_START=1;//Initialize the temperature;
    T_END=1e-20;
    Tmp = T_START;

    k0 =1e-23;
    k1 =10;//larger k helps to get out of trap.
    fireUp_times = 0;
    cnt=0;//record stop times
    warmUp_rate=4.5;//lift the temperature when wander so as to get out of trap;
    stayTimes=5000;//To judge whether it is get trapped;
}

void simulated_Annealing::Forward()
{
    cnt=0;
    y = tempY;
    X = X_next;
}

void simulated_Annealing::Stay(){
    X_next = X;
    cnt++;
}

bool simulated_Annealing::Metropolis(double n)
{
    if(n<0) return true;
    double m = rand()/((double)RAND_MAX);
    double p = trapped_in_Local()? exp(-n/Tmp/k1):exp(-n/Tmp/k0);
    return p > m;
}

double simulated_Annealing::Evaluate(VectorXd &tempX) {
    tempY = (*Obj_Fct)(tempX);
    double delta = tempY-y;
    return delta;
}

inline void simulated_Annealing::coolDown(){
    Tmp *=des;
}

bool simulated_Annealing::trapped_in_Local() const{
    return (cnt>stayTimes&&y>EPS);
}

inline double simulated_Annealing::warmUp(){
    return trapped_in_Local() ? warmUp_rate * Tmp : 0;
}

inline void simulated_Annealing::wander(double tmp){
    for(int i=0;i<dimension;i++)
        X_next[i]+=tmp*((rand()/((double)RAND_MAX))*2-1);
}

bool simulated_Annealing::solve(){
    bool flag = false;
    fireUp_times = 0;

    do{
        Tmp=T_START;
        srand(std::time(nullptr));
        fireUp_times++;
        while(Tmp>=T_END)
        {
            for(int i=0;i<iteration_max;i++, it_counter++)
            {
                wander((Tmp+warmUp()));
                if(Metropolis(Evaluate(X_next))){
                    Forward();
#if FOR_DEBUG
                    std::cout<<X.transpose()<<std::endl;
#endif
                }
                else Stay();
            }
            if(y<EPS) break;
            coolDown();
        }
    }while(y>EPS&&fireUp_times<2);
    X = X_next;
    if(y<EPS) {
        flag = true;
        std::cout<<std::endl<<"Global Minimum Arrived!"<<std::endl
                 << "# Global Minimum Pos:    [" << X.transpose() << "]'" << std::endl
                 << "# Global Minimum Value:  " << y<< std::endl
                 << "# Iteration times:       " << it_counter<< std::endl;
    }else{
        std::cout<<std::endl<<"Fail to arrive at the global Minimum!"<<std::endl
                 << "# Global Minimum Pos:    [" << X.transpose() << "]'" << std::endl
                 << "# Global Minimum Value:  " << y<< std::endl;
    }
    return flag;
}


