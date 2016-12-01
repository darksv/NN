using System;

namespace NN
{
    class SigmoidFunction : IActivationFunction
    {
        public double CalculatePrimeValue(double x)
        {
            return CalculateValue(x) * (1.0 - CalculateValue(x));
        }

        public double CalculateValue(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}
