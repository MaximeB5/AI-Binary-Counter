// Headers
#include "neuron.h"
#include <cmath>

//////////////////////////////////////////////////////////////////////

double Neuron::eta = 0.15; // overall bet learning rate.
double Neuron::alpha = 0.5; // momentum (multiplier of the last deltaWeight).

//////////////////////////////////////////////////////////////////////

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    // c for connection
    for( unsigned c = 0; c < numOutputs; ++c ) {
        m_outputWeights.push_back(Connection());

        // Let's put random values
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

//////////////////////////////////////////////////////////////////////

void Neuron::feedForward(const Layer &prevLayer) {
    // Activation function : output = f( sum of all (input * weight) )
    double sum = 0.0;

    // Sum the previous layer's output (which are our inputs)
    // include the bias node from the previous layer
    for( unsigned n = 0; n < prevLayer.size(); ++n ) {
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    // Activation function (also nammed Transfer function)
    m_outputVal = Neuron::transferFunction(sum);
}

//////////////////////////////////////////////////////////////////////

double Neuron::transferFunction(double x) {
    // It can be step function (trigger 0->1 e.g.),
    // or ramp function with linear part.
    // But we need to derivate it, so we need something with a curve
    // like some kind of exponential, or sigmoid curve.
    // We'll use hyperbolic tangent function which gives an output
    // in the range : tanh : [-1.0...+1.0]

    // Any of these functions work.
    // The only consideration is when we're setting up our training data,
    // just scale our output, so that the output values are always within
    // the range of what the transfer function is able to make it.

    // tanh : [-1.0...+1.0]
    return tanh(x);
}

//////////////////////////////////////////////////////////////////////

double Neuron::transferFunctionDerivative(double x) {
    // tanh derivative
    // we use a little approximation over the interval we need to cover in.
    return (1.0 - x * x);
}

//////////////////////////////////////////////////////////////////////

void Neuron::calcOutputGradients(double targetVal) {
    // Following lines are just one of the several ways to calculate gradients
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

//////////////////////////////////////////////////////////////////////

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    // Unlike output, we don't know how to do the delta (no data to).
    double dow = sumDow(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

//////////////////////////////////////////////////////////////////////

double Neuron::sumDow(const Layer &nextLayer) const {
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed in the next layer.
    // Without including the bias ( -> -1)
    for( unsigned n = 0; n < nextLayer.size() - 1; ++n ) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

//////////////////////////////////////////////////////////////////////

void Neuron::updateInputWeights(Layer &prevLayer) {
    // The weights to be updated are in the connection container
    // in the neurons in the preceding layer.
    // Including the bias !
    for( unsigned n = 0; n < prevLayer.size(); ++n ) {
        // neuron intialized with the current neuron
        // that were modifying in the previous layer
        // so the neuron is the other neuron in the previous layer we were updating
        Neuron &neuron = prevLayer[n];

        // We need to remember that other neuron connection weights
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate.
                // eta = overall training rate.
                // eta = 0.0 : slow learner ; 0.2 : medium learner ; 1.0 : reckless learner.
                eta
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum (alpha) that is a fraction of the previous delta weight.
                // Momentum = Multiplier of last weight change.
                // alpha = 0.0 : no momentum ; 0.5 : moderate momentum
                + alpha
                * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}
