import java.util.Arrays;

public class Model {
    private DenseLayer[] layers;
    private MSELoss loss;

    public Model(DenseLayer[] layers, double lr, MSELoss loss){
        this.layers = layers;
        this.loss = loss;

        for(DenseLayer layerI: layers){
            layerI.setLr(lr);
        }
    }

    public void train(double[][] xs, double[][] ys, int epochs){
        for(int epochI = 0; epochI < epochs; epochI++){
            for(int xsJ = 0; xsJ < xs.length; xsJ++){
//                System.out.println(" ");

                double[] currentState = this.layers[0].forwardPass(xs[xsJ]);
                for(int layerK = 1; layerK < this.layers.length; layerK++){
                    currentState = this.layers[layerK].forwardPass(currentState);
                }

//                System.out.println(Arrays.toString(currentState));

                //TODO: loss metric
                double[] loss = this.loss.loss(currentState, ys[xsJ]);
                double[] lossGradient = this.loss.lossGradient(currentState, ys[xsJ]);

                System.out.println("LOSS: " +
                        Arrays.toString(loss) +
                        " OUTPUT: " +
                        Arrays.toString(currentState) +
                        " ACTUAL: " +
                        Arrays.toString(ys[xsJ]));

                currentState = this.layers[this.layers.length-1].backPass(lossGradient);


                for(int layerK = this.layers.length-2; layerK >= 0; layerK++){
                    currentState = this.layers[layerK].backPass(currentState);
                }
            }
        }
    }
}
