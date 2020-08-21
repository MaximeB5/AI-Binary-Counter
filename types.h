#ifndef TYPES_H
#define TYPES_H

// This file must not be included by the class Neuron since it needs it to define the Layer type.
// Class Neuron has to define itself the Layer type.

// Headers
#include <vector>

// Class Headers
#include "neuron.h"

typedef std::vector<Neuron> Layer;

#endif // TYPES_H
