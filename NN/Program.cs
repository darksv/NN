using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace NN
{
    internal class Program
    {
        private static IEnumerable<Vector<double>> LoadFromCsv(string path)
        {
            try
            {
                return File.ReadAllLines(path)
                   .Select(line =>
                           Vector<double>.Build.DenseOfEnumerable(
                               line
                               .Split(',')
                               .Select(x => double.Parse(x, CultureInfo.InvariantCulture))
                           )
                   );
            }
            catch (IOException)
            {
                Console.WriteLine($"Could not open file {path}");

                return Enumerable.Empty<Vector<Double>>();
            }
        }

        public static Vector<double> ConvertOutput(Vector<double> output)
        {
            var v = Vector<double>.Build.Dense(10);
            int index = (int)output[0] - 1;
            v[index] = 1.0;
            return v;
        }
        

        private static void Main(string[] args)
        {
            Console.WriteLine("Loading samples from files...");

            var inputs = LoadFromCsv("X.csv").ToArray();
            var outputs = LoadFromCsv("y.csv").Select(ConvertOutput).ToArray();

            if (inputs.Length == 0)
                return;
            
            var net = new NeuralNetwork(new SigmoidFunction());
            net.AddLayer(400);
            net.AddLayer(25);
            net.AddLayer(10);
            
            const int numEpochs = 100;
            const int batchSize = 50;
            const double learningSpeed = 0.21;

            int numSamples = inputs.Length;
            int numBatches = numSamples / batchSize;

            Console.WriteLine($"Started learning network ({numEpochs} epochs, {numBatches} minibatches, {batchSize} samples each)");
            
            var sw = new Stopwatch();

            for (int i = 0; i < numEpochs; ++i)
            {
                sw.Start();

                var shuffledBatch = inputs
                    .Zip(outputs, Tuple.Create)
                    .OrderBy(x => Guid.NewGuid())
                    .ToArray();

                var shuffledInputs = shuffledBatch
                    .Select(x => x.Item1)
                    .ToArray();

                var shuffledOutputs = shuffledBatch
                    .Select(x => x.Item2)
                    .ToArray();
                
                // Perform Stochastic Gradient Descent
                for (int j = 0; j < numBatches; ++j)
                {
                    var batchInputs = shuffledInputs
                        .Skip(j * batchSize)
                        .Take(batchSize)
                        .ToArray();

                    var batchOutputs = shuffledOutputs
                        .Skip(j * batchSize)
                        .Take(batchSize)
                        .ToArray();
                    
                    net.UpdateMultipleSamples(batchInputs, batchOutputs, learningSpeed);

                    Console.Write('.');
                }

                Console.WriteLine();


                int numCorrect = inputs
                    .Select(input => net.GetOutput(input))
                    .Where((output, j) => outputs[j][output.MaximumIndex()].Equals(1.0))
                    .Count();
                
                int numInvalid = numSamples - numCorrect;
                double ratio = (double) numInvalid / numSamples;
                Console.WriteLine($"Epoch: {i + 1} / {numEpochs}; Error: {numInvalid} / {numSamples} [{ratio}]; Time: {sw.Elapsed}");

                sw.Reset();
            }
        }
    }
}