
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
        output_empty.write(false);
	sc_start(1, SC_NS);
        reset.write(true);
        cout << "running..." << endl;
	int numberCycles = 0;

	while (numberCycles<1000 && not sc_end_of_simulation_invoked()) {
		clock = 0;
                net.debug();
                if(net.input_empty.read())
                {
                       bridge::read_in_image();
                       input_filled.write(true);
                }else input_filled.write(false);
		sc_start(1, SC_NS);
		clock = 1;
		sc_start(1, SC_NS);
		numberCycles++;
                if(net.output_filled.read())
                {
                       bridge::read_out_result();
                       output_empty.write(true);
                }else   output_empty.write(false);
	}
	
	cout << "\nFinished after " << numberCycles - 4 << " cicles. Final state:\n" << endl;	
}
  catch (std::bad_alloc& ba)
  {
    std::cerr << "bad_alloc caught: " << ba.what() << '\n';
  }
 
}

