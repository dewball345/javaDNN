import activations.SoftmaxActivation;

import java.util.Arrays;

public class softmaxtest {
    public static void main(String[] args){
        SoftmaxActivation a = new SoftmaxActivation();

        System.out.println(Arrays.toString(a.forwardPass(new double[]{0.1, 0.6, -0.2})));
    }
}
