package activations;

import activations.Activation;

public class LeakyReluActivation extends Activation {
    //see relu comments for understanding...

    public double[] forwardPass(double[] inputs){
        double[] outputs = new double[inputs.length];
        for(int inputI = 0; inputI < inputs.length; inputI++){
            if(inputs[inputI] > 0){
                outputs[inputI] = inputs[inputI];
            } else {
                outputs[inputI] = 0.2 * inputs[inputI];
            }
        }

        return outputs;
    }

    public double[] backPass(double[] gradients){
        double[] computedGradients = new double[gradients.length];

        for(int gradientI = 0; gradientI < gradients.length; gradientI++){
            if(gradients[gradientI] > 0){
                computedGradients[gradientI] = 1 * gradients[gradientI];
            } else {
                computedGradients[gradientI] = 0.2 * gradients[gradientI];
            }
        }

        return computedGradients;
    }
}
