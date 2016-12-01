using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    public interface IActivationFunction
    {
        double CalculateValue(double x);
        double CalculatePrimeValue(double x);
    }
}
