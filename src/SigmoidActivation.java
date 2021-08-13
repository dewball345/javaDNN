import java.lang.Math;

public class SigmoidActivation extends Activation {
    //forward pass implementation
    public double[] forwardPass(double[] inputs){
        double[] outputs = new double[inputs.length];
        for(int inputI = 0; inputI < inputs.length; inputI++){
            //equation of a sigmoid
            outputs[inputI] = 1/(1 + Math.pow(Math.E, inputs[inputI] * -1));
        }

        return outputs;
    }

    //backprop implementation
    public double[] backPass(double[] gradients){
        double[] computedGradients = new double[gradients.length];


        //retrieving local gradients
        double[] sigmoidForward = this.forwardPass(gradients);
        double[] localGradient = new double[gradients.length];

        for(int sfI = 0; sfI < gradients.length; sfI++){
            //sigmoid gradient is sigmoid(gradient) * (1 - sigmoid(gradient))
            localGradient[sfI] = sigmoidForward[sfI] * (1-sigmoidForward[sfI]);
        }

        //multiply with upstream gradients
        for(int gradientI = 0; gradientI < gradients.length; gradientI++){
            computedGradients[gradientI] = localGradient[gradientI] * gradients[gradientI];
        }

        return computedGradients;
    }
}