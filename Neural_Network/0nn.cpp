#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

struct Connection{
    double weight;
    double deltaWeight;
};

class Neuron{
    public:
        Neuron(unsigned nOutputs, unsigned index);
        void setOutputVal(double val){
            m_outputVal = val;
        }
        double getOutputVal(void) const{
            return m_outputVal;
        }
        static double randomWeight(void){
            return rand() / double(RAND_MAX);
        }
        void feedForward(const Layer &prevLayer);
        double transferFunction(double x);
        void calcOutputGradients(double targetValues);
        double transferFunctionDerivative(double x);
        void calcHiddenGradients(Layer &nextLayer);
        double sumDOW(Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);
        double getWeight(double w);

    private:
        static double eta;
        static double alpha;
        double m_outputVal;
        vector<Connection> m_outputWeights;
        unsigned m_index;
        double m_gradient;
};

double Neuron::eta = 0.18;
double Neuron::alpha = 0.5;
Neuron::Neuron(unsigned nOutputs, unsigned index){
    for(unsigned c = 0; c < nOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_index = index;
}
double Neuron::transferFunctionDerivative(double x){
    return 1.0 - x * x;
}
double Neuron::transferFunction(double x){
    return tanh(x);
}
void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;
    for(int n = 0; n < prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_index].weight;
    }
    m_outputVal = Neuron::transferFunction(sum);
}
void Neuron::calcOutputGradients(double targetValues){
    double delta = targetValues - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}
void Neuron::calcHiddenGradients(Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * transferFunctionDerivative(m_outputVal);
}
double Neuron::sumDOW(Layer &nextLayer){
    double sum = 0.0;
    for(int n = 0; n < nextLayer.size() - 1; n++){
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}
void Neuron::updateInputWeights(Layer &prevLayer){
    for(int n = 0; n < prevLayer.size(); n++){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
        neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_index].weight += newDeltaWeight;
    }
}
double Neuron::getWeight(double w){
    return m_outputWeights[w].weight;
}

class GetData{
    public:
        GetData(const string filename);
        void getTopology(vector<int>& topology, int &argc, char** &argv);
        int getNextInputs(vector<double> &inputValues);
        int getTargetOutputs(vector<double> &targetOutputValues);
        bool isEof(void){
            return dataFile.eof();
        }
    private:
        ifstream dataFile;
};

void GetData::getTopology(vector<int>& topology, int &argc, char** &argv){
    for(int i = 1; i < argc; i++){
        topology.push_back(atoi(argv[i]));
    }
    return;
}
GetData::GetData(const string filename){
    dataFile.open(filename.c_str());
}
int GetData::getNextInputs(vector<double> &inputValues){
    string line;
    getline(dataFile, line);
    stringstream ss(line);
    string label;
    ss >> label;
    if(label.compare("in:") == 0){
        inputValues.clear();
        double oneValue;
        while(ss >> oneValue){
            inputValues.push_back(oneValue);
        }
    }
    return inputValues.size();
}

int GetData::getTargetOutputs(vector<double> &targetOutputValues){
    string line;
    getline(dataFile, line);
    stringstream ss(line);
    string label;
    ss >> label;
    if(label.compare("out:") == 0){
        targetOutputValues.clear();
        double oneValue;
        while(ss >> oneValue){
            targetOutputValues.push_back(oneValue);
        }
    }
    return targetOutputValues.size();
}

class Net{
    public:
        Net(const vector<int> &topology);
        void feedForward(const vector<double> &inputValues);
        void getResults(vector<double> &resultValues);
        void backProp(vector<double> &targetValues);
        void printWeights(void);
        double getRecentAverageError(void) const {
            return m_recentAverageError;
        }
    private:
        vector<Layer> m_layers;
        double m_error;
        double m_recentAverageError;
        double m_recentAverageSmoothingFactor;
};
Net::Net(const vector<int> &topology){
    int nLayers = topology.size();
    for (int layerNum = 0; layerNum < nLayers; ++layerNum) {
        m_layers.push_back(Layer());
        int nOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        cout<<endl;
        for (int neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(nOutputs, neuronNum));
            cout << "Made Neuron " << neuronNum+1 << " in Layer " << layerNum+1<< endl;
        }
        m_layers.back().back().setOutputVal(1.0);
    }
}
void Net::printWeights(){
    for(int layerNum = 0; layerNum < m_layers.size(); layerNum++){
        cout<<endl;
        cout<<"layer " << layerNum <<":"<<endl;
        cout<<"-----------------"<<endl;
        for(int n = 0; n < m_layers[layerNum].size() - 1; n++){
            cout<<"Neuron "<<n<<":= ";
            for(int w = 0; w < m_layers[layerNum + 1].size(); w++){
                cout<<"Weight "<<m_layers[layerNum][n].getWeight(w)<<",";
            }cout<<endl;   
        }
    }
}
void Net::feedForward(const vector<double> &inputValues){
    assert(inputValues.size() == m_layers[0].size() - 1);
    for(int i = 0; i < inputValues.size(); i++){
        m_layers[0][i].setOutputVal(inputValues[i]);
    }
    for(int layerNum = 1; layerNum < m_layers.size(); ++layerNum){
        Layer &prevLayer = m_layers[layerNum - 1];
        for(int n = 0; n < m_layers[layerNum].size() - 1; ++n){
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}
void Net::getResults(vector<double> &resultValues){
    resultValues.clear();
    for(int n = 0; n < m_layers.back().size() - 1; ++n){
        resultValues.push_back(m_layers.back()[n].getOutputVal());
    }
}
void Net::backProp(vector<double> &targetValues){
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    for(int n = 0; n < outputLayer.size() - 1; ++n){
        double delta = targetValues[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);
    for(int n = 0; n < outputLayer.size() - 1; n++){
        outputLayer[n].calcOutputGradients(targetValues[n]);
    }
    for(int layerNum = m_layers.size() - 2; layerNum > 0; layerNum--){
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
        for(int n = 0; n < hiddenLayer.size(); n++){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    for(int layerNum = m_layers.size() - 1; layerNum > 0; layerNum--){
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];
        for(int n = 0; n < layer.size() - 1; n++){
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void showVectorValues(string label, vector<double> &v){
    cout<<label<<" ";
    for(int i = 0; i < v.size(); i++){
        cout<<v[i]<<" ";
    }cout<<endl;
}

int main(int argc, char** argv) {
    GetData getData("samples.txt");
    vector<int> topology;
    getData.getTopology(topology, argc, argv);
    Net network(topology);
    vector<double> inputValues, targetValues, resultValues;

    int count = 0;
    while(!getData.isEof()){
        count++;
        cout<<"feed forward input number: "<<count<<" to NN"<<endl;
        if(getData.getNextInputs(inputValues) != topology[0]){
            break;
        }
        assert(inputValues.size() == topology[0]);
        showVectorValues("Inputs:", inputValues);
        network.feedForward(inputValues);
        network.getResults(resultValues);
        showVectorValues("Outputs:", resultValues);
        getData.getTargetOutputs(targetValues);
        showVectorValues("Targets:", targetValues);
        assert(targetValues.size() == topology.back());
        network.backProp(targetValues);
        cout<<"Net recent average error: "<<network.getRecentAverageError()<<endl;
    }
    network.printWeights();
    cout<<endl<<"The End!"<<endl;

    cout<<"Testing"<<endl;
    inputValues.clear();
    inputValues.push_back(100);
    inputValues.push_back(120);
    inputValues.push_back(140);
    int targetValue = 0.63*100 + 0.21*120 + 0.14*140;
    showVectorValues("Inputs:", inputValues);
    network.feedForward(inputValues);
    network.getResults(resultValues);
    showVectorValues("Outputs:", resultValues);
    cout<<"Target: "<<targetValue<<" Result is "<<resultValues[0]<<endl;
}