#pragma once
#include "systemc.h"

template <int N>
SC_MODULE(sc_gated_and)
{
   SC_CTOR(sc_gated_and):clk("clk"),reset("reset"),out("out"),in("in",N)
   {
       out(result);
       SC_METHOD(run);
       sensitive << clk.pos()<<reset;
       for(int i=0;i<N;i++)
           sensitive << in[i];
   }

   void run()
   {
       bool r=true;
       for(int i=0;i<N;i++) r=r&&in[i].read();
       bool s=false;
       for(int i=0;i<N;i++) s=s||in[i].read();
       if(!reset.read())
       {
          result.write(r);
            
       }
       else if(clk.posedge())
       {
          if(!s)result.write(false);
          else if(r)result.write(true);
       }
   }

   void debug()
   {

        std::cout << name() << ":";

        for(int i=0;i<N;i++) std::cout <<"in" << i <<':'<<in[i] << ";";
        std:cout <<"out:" << result << std::endl;
       
   }
   sc_in<bool> clk;
   sc_in<bool> reset;
   sc_out<bool> out;
   sc_vector<sc_in<bool>> in;
   sc_signal<bool> result;
};
