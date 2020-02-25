#pragma once
#include "systemc.h"

template <int N>
SC_MODULE(sc_and)
{
   SC_CTOR(sc_and)
   {
       SC_THREAD(run);
       for(int i=0;i<N;i++)
           sensitive << in[i];
   }

   void run()
   {
        bool r=true;
        for(int i=0;i<N;i++) r&=-in[i].read();
        out.write(r);
   }

   sc_out<bool> out;
   sc_in<bool> in[N];
};
