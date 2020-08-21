// Headers
#include "net.h"
#include <iostream>
#include <cassert>
#include <cmath>

//////////////////////////////////////////////////////////////////////

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

//////////////////////////////////////////////////////////////////////

Net::Net(const std::vector<unsigned> &topology) {
    unsigned numLayers = topology.size();

    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        std::cout << "Made a Layer !" << std::endl;

        // Output layer has no further output (= 0), otherwise we got the number of outputs for the next hidden layer
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        // We have a new Layer, now fill it with neurons,
        // and add a bias neuron to the layer.
        // A bias neuron is a neuron without output whose
        // the output is set to 1.0 only.
        // the bias is added thanks to the operator '<=' and not only '<'
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));	// back give the most recent element in the container. So we add neurons in the layer.
            std::cout << "Made a Neuron !" << std::endl;
        }

        // Force the bias node's output value to 1.0. It's the last neuron created above.
        m_layers.back().back().setOutputVal(1.0);
    }
}

//////////////////////////////////////////////////////////////////////

void Net::feedForward(const std::vector<double> &inputVals) {
    // Check if the number of neurons is the same at the next layer, otherwise there is no real reason to feed forward.
    assert( inputVals.size() == m_layers[0].size() - 1 ); // if statement not true, error msg during runtime

    // Assign (latch) the input values into the input neurons.
    for( unsigned i = 0; i < inputVals.size(); ++i ) {
        // element 0 is the input layer and element i is the i neuron of the input
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagate

    // Propagation means in this case a looping through each layer,
    // and then inside their looping through each neuron in the layer,
    // and then telling each individual neuron 'please feedforward'

    // Begins from 1 because the inputs are already set.
    // We're starting with the first hidden layer.
    // We want to go through, and including the ouput layer.
    for( unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum ) {
        // See comment (*) below for the following line :
        Layer &prevLayer = m_layers[layerNum - 1];

        // We go through each neuron in that layer.
        // We minus 1 because of the bias neuron that we do not want.
        for( unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n ) {
            // Inside, we want to tell each neuron to do the feedforward to address
            // individual neurons to say n layers.
            // The 1st index is the layer number, the next index is the neuron number.
            // Then, we call the feed forward method in Neuron that has the nasty mathematical stuff inside that updates its output value.

            // When a class net is asked to feed forward, it's going to need to add up all of its input values,
            // and then function to it in order to upgrade its output value.
            // To do this, it needs to ask to the neurons in the preceding layer what is alpha values are.
            // So it needs to go through ahh the neurons in the previous layer.
            // We could make them friends, but what we need is not so many information.
            // We just need a hand or a pointer to the neurons in the previous layer, and that's all it needs.

            // (*)
            // Let's say at this point it will have a reference to the previous layer.
            // Which is a container of neurons that we can pass here.

            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

//////////////////////////////////////////////////////////////////////

void Net::backProp(const std::vector<double> &targetVals) {
    // We have to do several calculations :
    // -> Calculate the overall net error (RMS (Root Mean Square Error) of output neuron errors)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    // - 1 because we do not include the bias
    for( unsigned n = 0; n < outputLayer.size() - 1; ++n ) {
        // delta between expected value and the actual value
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }

    m_error /=  outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error);           // RMS

    // (*) Implement a recent average measurement :
    // it has nothing to do with the neural network,
    // but it will help us to has an error indication
    // to know how the net is going over several training samples.
    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / ( m_recentAverageSmoothingFactor + 1.0 );

    // -> Calculate output layer gradients
    // -1 because we do not want the bias
    for( unsigned n = 0; n < outputLayer.size() - 1; ++n ) {
        // It needs to have its target value past tense will pass to that target value for that neuron.
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // -> Calculate gradients on hidden layers
    // -2 because we'll loop through all hidden layers from the right side until finding the last one before the input layer.
    for( unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum  ) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer   = m_layers[layerNum + 1];

        // Now we have hidden and next layers, we'll loop through all neurons in the hidden layer.
        for( unsigned n = 0; n < hiddenLayer.size(); ++n  ) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // -> For all layers from outputs to first hidden layer, update connection weights
    // We need to go through all the layers. We begin by the rightmost layer (which is the number of layer minus one).
    // We don't need the input layer because there is no way to coming into yet so.
    for( unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum ) {
        Layer &layer     = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        // For each neuron, we index that individual neuron in our layer to update its wieghts.
        for( unsigned n = 0; n < layer.size() - 1; ++n ) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

//////////////////////////////////////////////////////////////////////

void Net::getResults(std::vector<double> &resultVals) const {
    resultVals.clear();

    for( unsigned n = 0; n < m_layers.back().size() - 1; ++n ) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

//////////////////////////////////////////////////////////////////////
