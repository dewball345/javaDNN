package activations;

import activations.Activation;

public class ReluActivation extends Activation {
    //forward pass
    public double[] forwardPass(double[] inputs){
        double[] outputs = new double[inputs.length];

        //if item in input is less then zero, make it zero
        for(int inputI = 0; inputI < inputs.length; inputI++){
            if(inputs[inputI] > 0){
                outputs[inputI] = inputs[inputI];
            } else {
                outputs[inputI] = 0;
            }
        }

        return outputs;
    }

    //backprop
    public double[] backPass(double[] gradients){
        double[] computedGradients = new double[gradients.length];

        //if gradients less than zero, set gradients to zero, else multiply upstream gradient with inner gradient(0)
        for(int gradientI = 0; gradientI < gradients.length; gradientI++){
            if(gradients[gradientI] > 0){
                computedGradients[gradientI] = 1 * gradients[gradientI];
            } else {
                computedGradients[gradientI] = 0;
            }
        }

        return computedGradients;
    }
}
