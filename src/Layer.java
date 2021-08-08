public abstract class Layer {
    public abstract double[] forwardPass(double[] inputs);
    public abstract double[] backPass(double[] gradients);
}
