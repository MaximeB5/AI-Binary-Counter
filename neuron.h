#ifndef NEURON_H
#define NEURON_H

// Headers
#include <vector>
#include <cstdlib>

// Types def :
class Neuron;
typedef std::vector<Neuron> Layer;


struct Connection { // contains the weight
    double weight;
    double deltaWeight;
};


class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta; // only class neuron needs it [0.0...1.0]. Overall net training rate.
    static double alpha; // only class neuron needs it [0.0...n]. Multiplier of last weight change (momentum).
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x); // for back propagate learning
    static double randomWeight(void) { return ( rand() / double(RAND_MAX) ); }
    double sumDow(const Layer &nextLayer) const;
    double m_outputVal;
    std::vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

#endif // NEURON_H
