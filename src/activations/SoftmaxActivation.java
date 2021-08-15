package activations;

import activations.Activation;

import java.lang.Math;

public class SoftmaxActivation extends Activation {

    public double[] forwardPass(double[] inputs) {
        double[] probabs = new double[inputs.length];
        double denominator = 0;

        for(int inI = 0; inI < inputs.length; inI++){
            denominator += Math.pow(Math.E, inputs[inI]);
        }

        for(int inI = 0; inI < inputs.length; inI++){
            double numerator = Math.pow(Math.E, inputs[inI]);
            probabs[inI] = numerator/denominator;
        }

        return probabs;
    }

    //TODO: backward prop
    public double[] backPass(double[] gradients) {
        double[] computedGradients = new double[gradients.length];

    }
}
