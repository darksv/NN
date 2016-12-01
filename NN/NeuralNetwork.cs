using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace NN
{
    /// <summary>
    /// Simple neural network.
    /// </summary>
    public class NeuralNetwork
    {
        /// <summary>
        /// Activation function for neurons.
        /// </summary>
        private readonly IActivationFunction _activationFunction;

        /// <summary>
        /// Collection of layers of the network.
        /// </summary>
        private readonly List<int> _layers = new List<int>();

        /// <summary>
        /// Collection of matrices with weights for connections between layers.
        /// </summary>
        private readonly List<Matrix<double>> _weights = new List<Matrix<double>>();

        /// <summary>
        /// Collection of vectors of biases for every neuron in each layer.
        /// </summary>
        private readonly List<Vector<double>> _biases = new List<Vector<double>>();

        /// <summary>
        /// Number of layers.
        /// </summary>
        public int LayerCount => _layers.Count;

        /// <summary>
        /// Whether network has any layer.
        /// </summary>
        public bool HasLayers => _layers.Count > 0;

        /// <summary>
        /// Number of inputs.
        /// </summary>
        public int InputCount => _layers.FirstOrDefault();

        /// <summary>
        /// Number of outputs
        /// </summary>
        public int OutputCount => _layers.LastOrDefault();

        /// <summary>
        /// Initializes new neural network with given activation function.
        /// </summary>
        /// <param name="activationFunction"></param>
        public NeuralNetwork(IActivationFunction activationFunction)
        {
            _activationFunction = activationFunction;
        }

        /// <summary>
        /// Add a new layer with a given number of neurons.
        /// </summary>
        /// <param name="numNeurons">number of neurons in the layer</param>
        public void AddLayer(int numNeurons)
        {
            if (HasLayers)
            {
                int previousLayerSize = _layers.Last();
                var distribution = new ContinuousUniform(-1.0, 1.0);

                _weights.Add(Matrix<double>.Build.Random(numNeurons, previousLayerSize, distribution));
                _biases.Add(Vector<double>.Build.Random(numNeurons, distribution));
            }

            _layers.Add(numNeurons);
        }
        
        /// <summary>
        /// Calculate output of the network for a given input.
        /// </summary>
        /// <param name="input">vector of inputs</param>
        /// <returns>output of the network</returns>
        public Vector<double> GetOutput(Vector<double> input)
        {
            if (input.Count != InputCount)
                throw new ArgumentException($"Network has {InputCount} inputs, but got {input.Count} values");

            var layerOutput = input;
            for (int i = 0; i < _layers.Count - 1; i++)
            {
                var weightedInput = _weights[i] * layerOutput + _biases[i];
                layerOutput = weightedInput.Map(_activationFunction.CalculateValue);
            }
            
            return layerOutput;
        }

        /// <summary>
        /// Perform backpropagation algorithm for a single training example.
        /// </summary>
        /// <param name="input">features of training example</param>
        /// <param name="output">expected response from the network</param>
        /// <returns>gradients for biases and weights</returns>
        private Tuple<Vector<double>[], Matrix<double>[]> BackpropagateOne(Vector<double> input, Vector<double> output)
        {
            if (input.Count != InputCount)
                throw new Exception($"Number of components in the input vector ({input.Count}) " +
                                    $"does not match to number of network's inputs ({InputCount})");

            if (output.Count != OutputCount)
                throw new Exception($"Number of components in the output vector ({output.Count}) " +
                                    $"does not match to number of network's outputs ({OutputCount})");

            var layerOutputs = new Vector<double>[LayerCount - 1];
            var weightedInputs = new Vector<double>[LayerCount];
            weightedInputs[0] = input;
            
            // Feedforward
            var layerOutput = input;
            for (int i = 0; i < LayerCount - 1; i++)
            {
                var weightedInput = _weights[i] * layerOutput + _biases[i];
                weightedInputs[i + 1] = weightedInput;

                layerOutput = weightedInput.Map(_activationFunction.CalculateValue);
                layerOutputs[i] = layerOutput;
            }

            var biasesGrad = new Vector<double>[LayerCount - 1];
            var weightsGrad = new Matrix<double>[LayerCount - 1];
            
            // Backpropagation
            Vector<double> delta = layerOutputs.Last() - output;
            biasesGrad[LayerCount - 2] = delta;
            weightsGrad[LayerCount - 2] = delta.ToColumnMatrix() * weightedInputs[LayerCount - 2].ToRowMatrix();

            for (int i = LayerCount - 3; i >= 0; --i)
            {
                delta = (_weights[i + 1].Transpose() * delta)
                    .PointwiseMultiply(layerOutputs[i].Map(_activationFunction.CalculatePrimeValue));

                biasesGrad[i] = delta;
                weightsGrad[i] = delta.ToColumnMatrix() * weightedInputs[i].ToRowMatrix();
            }
            
            return Tuple.Create(biasesGrad, weightsGrad);
        }

        /// <summary>
        /// Perform update of network parameters using a minibatch.
        /// </summary>
        /// <param name="inputs">collection of inputs</param>
        /// <param name="outputs">collection of outputs</param>
        /// <param name="learningRate">network learining rate parameter</param>
        public void Minibatch(Vector<double>[] inputs, Vector<double>[] outputs, double learningRate)
        {
            if (inputs.Length != outputs.Length)
                throw new ArgumentException($"Invalid training data. Number of inputs {inputs.Length} samples must be equal to number of output samples {outputs.Length}.");

            int numSamples = inputs.Length;

            // Create empty matrices for gradients
            Vector<double>[] biasesGrad = _biases
                .Select(x => Vector<double>.Build.Dense(x.Count))
                .ToArray();

            Matrix<double>[] weightsGrad = _weights
                .Select(x => Matrix<double>.Build.Dense(x.RowCount, x.ColumnCount))
                .ToArray();

            double alpha = learningRate / numSamples;
            
            for (int i = 0; i < numSamples; ++i)
            {
                var grad = BackpropagateOne(inputs[i], outputs[i]);

                for (int j = 0; j < LayerCount - 1; ++j)
                {
                    biasesGrad[j] += alpha * grad.Item1[j];
                    weightsGrad[j] += alpha * grad.Item2[j];
                }
            }

            // Perform update of parameters
            for (int j = 0; j < LayerCount - 1; ++j)
            {
                _weights[j] -= weightsGrad[j];
                _biases[j] -= biasesGrad[j];
            }
        }
    }
}
