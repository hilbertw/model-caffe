#pragma once
#include "systemc.h"

template <int N>
SC_MODULE(sc_and)
{
   SC_CTOR(sc_and):out("out"),in("in",N)
   {
       out(result);
       SC_THREAD(run);
       for(int i=0;i<N;i++)
           sensitive << in[i];
   }

   void run()
   {
        bool r=true;
        for(int i=0;i<N;i++) r&=in[i].read();
        result.write(r);
   }

   void debug()
   {

        std::cout << name() << std::endl;

        for(int i=0;i<N;i++) std::cout <<i <<':'<<in[i] << ";";
        std:cout <<std::endl << result << std::endl;
       
   }
   sc_out<bool> out;
   sc_vector<sc_in<bool>> in;
   sc_signal<bool> result;
};
