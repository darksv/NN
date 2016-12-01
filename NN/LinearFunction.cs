using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    class LinearFunction : IActivationFunction
    {
        public double CalculateValue(double x)
        {
            return x;
        }

        public double CalculatePrimeValue(double x)
        {
            return 1.0;
        }
    }
}
