// Neural Network C++, binary counter test by Belaval Maxime 04/10/2019

// Headers
#include "types.h"
#include "net.h"
#include "neuron.h"
#include "data.h"
#include <iostream>
#include <cassert>

//////////////////////////////////////////////////////////////////////

void showVectorVals(std::string label, std::vector<double> &v)
{
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }

    std::cout << std::endl;
}

//////////////////////////////////////////////////////////////////////

int main() {

    /*********** TRAINING **********/

    Data trainData("..\\TrainingData\\trainingCounter_25000.txt");

    // Topolgy determines the number of layer
    // e.g. { 3, 2, 1 } :
    // 		input layer : 3 neurons
    // 		hidden layer : 2 neurons
    // 		output layer : 1 neuron
    std::vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    std::vector<double> inputVals;
    std::vector<double> targetVals;
    std::vector<double> resultVals;
    int trainingPass = 0;

    while( !trainData.isEof() ) {
        ++trainingPass;

        std::cout << std::endl << "Pass " << trainingPass;

        // Get new input data and feed it forward :
        if( trainData.getNextInputs(inputVals) != topology[0] ) {
            break;
        }

        showVectorVals(": Inputs : ", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results :
        myNet.getResults(resultVals);
        showVectorVals("Outputs : ", resultVals);

        // Train the net what the output should have been
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets : ", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals); // learning

        // Report how well the training is working, average over recent samples :
        std::cout << "Net recent average error : "
                  << myNet.getRecentAverageError()
                  << std::endl;
    } // end while

    std::cout << std::endl << "Work done - Learning complete" << std::endl;


    /******* REAL USE TESTING ******/
    inputVals.clear();  inputVals.shrink_to_fit();
    targetVals.clear(); targetVals.shrink_to_fit();
    resultVals.clear(); resultVals.shrink_to_fit();
    inputVals.reserve(8);
    targetVals.reserve(4);
    resultVals.reserve(1);

    std::cout << std::endl << "Now testing with real data :" << std::endl;

    Data realUseData("..\\RealUseData\\dataCounter.txt");
    trainingPass = 0;

    while( !realUseData.isEof() ) {
        ++trainingPass;

        std::cout << std::endl << "Pass " << trainingPass;

        // Get new input data and feed it forward :
        if( realUseData.getNextInputs(inputVals) != topology[0] ) {
            break;
        }

        showVectorVals(": Inputs : ", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results :
        myNet.getResults(resultVals);
        showVectorVals("Outputs : ", resultVals);

        // Train the net what the output should have been
        realUseData.getTargetOutputs(targetVals);
        showVectorVals("Targets : ", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals); // learning

        // Report how well the training is working, average over recent samples :
        std::cout << "Net recent average error : "
                  << myNet.getRecentAverageError()
                  << std::endl;
    } // end while


    // TODO : save weights and neural network config into a file like a XML or just a text file maybe ?
}
