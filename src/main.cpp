
#include <sstream>
#include <iostream>
#include <string.h>

#include "systemc.h"
#include "sc_net.h"
#include "cv-bridge.h"

using namespace std;


// sc_main in top level function like in C++ main
int sc_main(int argc, char* argv[]) {
        cout << "Loading processor..." << endl;
	sleep(1);

	sc_net net("net");

        bridge::init(net.input_blobs[0],net.output_blobs[0]);	
	sc_signal<bool> clock;

	net.clk(clock);

	int numberCycles = 0;

	while (not sc_end_of_simulation_invoked()) {
		clock = 0;
		sc_start(1, SC_NS);
		clock = 1;
		sc_start(1, SC_NS);
		numberCycles++;
	}
	
	cout << "\nFinished after " << numberCycles - 4 << " cicles. Final state:\n" << endl;	

}

