#include "sc_net.h"

void sc_net::init()
{
    load_weights();
    create_blobs();
    setup_blobs();
    setup_layers();
    setup_wires();
}
