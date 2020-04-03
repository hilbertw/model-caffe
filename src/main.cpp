
#include <sstream>
#include <iostream>
#include <string.h>
#include <new>

#include "systemc.h"
#include "sc_net.h"
#include "cv-bridge.h"

using namespace std;


// sc_main in top level function like in C++ main
int sc_main(int argc, char* argv[]) {
        if( argc<2)
        {
             std::cout <<"Usage:./sim <img>" << std::endl;
             return 0;
        }
        cout << "Loading processor..." << endl;
	sleep(1);
try{
	sc_net net("net");
        net.dump();
        bridge::init(net.input_blobs[0],net.output_blobs[0]);	
	sc_signal<bool> clock;
	sc_signal<bool> reset;
	sc_signal<bool> input_filled;
	sc_signal<bool> output_empty;
	net.input_filled(input_filled);
	net.output_empty(output_empty);

	net.clk(clock);
        net.reset(reset);
        net.setup_wires();

        std::cout << "resetting..." << endl;
        reset.write(false);
        input_filled.write(false);
        output_empty.write(true);
	sc_start(1, SC_NS);
        reset.write(true);
        cout << "running..." << endl;
	int numberCycles = 0;
        bool input_enable=true;
	while (numberCycles<150 && not sc_end_of_simulation_invoked()) {
                std::cout <<"No:" << numberCycles <<":"<<std::endl;
//                std::cout <<"input enable:" << input_enable <<":"<<std::endl;
//                std::cout <<"input empty:" << net.input_empty.read() <<":"<<std::endl;
//                std::cout <<"output empty:" << output_empty.read() <<":"<<std::endl;
//                std::cout <<"output filled:" << net.output_filled.read() <<":"<<std::endl;
		clock = 0;
                net.debug();
                if(input_enable&&net.input_empty.read())
                {
                       bridge::read_in_image(argv[1]);
                       input_filled.write(true);
                       input_enable=false;
                }else input_filled.write(false);
		sc_start(1, SC_NS);
		clock = 1;
		sc_start(1, SC_NS);
		numberCycles++;
                if(net.output_filled.read())
                {
                       output_empty.write(false);
                       bridge::read_out_result();
                       output_empty.write(true);
                       std::cout <<"ended.\n";
                       break;
                }
	}
	
	cout << "\nFinished after " << numberCycles - 4 << " cycles. Final state:\n" << endl;	
}
  catch (std::bad_alloc& ba)
  {
    std::cerr << "bad_alloc caught: " << ba.what() << '\n';
  }
 
}

