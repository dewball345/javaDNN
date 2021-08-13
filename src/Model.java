import java.util.Arrays;

public class Model {
    private Layer[] layers;
    private MSELoss loss;

    public Model(Layer[] layers, double lr, MSELoss loss){
        this.layers = layers;
        this.loss = loss;

        //Sets learning rate for layers and not activations
        for(Layer layerI: layers){
            if(!(layerI instanceof Activation)) {
                ((DenseLayer) layerI).setLr(lr);
            }
        }
    }

    //method for training model
    public void train(double[][] xs, double[][] ys, int epochs, int batchSize, boolean showActualPred){
        //iterates epoch times
        for(int epochI = 0; epochI < epochs; epochI++){
            //splits training data into batches
            for(int batchI = 0; batchI < xs.length; batchI += batchSize){
                //retrieves batch of data
                double[][] batchX = Arrays.copyOfRange(xs, batchI, batchI+batchSize);
                double[][] batchY = Arrays.copyOfRange(ys, batchI, batchI+batchSize);

                //array for batch predictions
                double[][] batchPreds = new double[batchY.length][];
                //state of layer(for getting prev layer output, feeding to next, etc.)
                double[] currentState = new double[batchY[0].length];

                //iterate through the batch of inputs
                for(int xsJ = 0; xsJ < batchX.length; xsJ++){
//                System.out.println(" ");
                    //forward pass through all layers
                    currentState = this.layers[0].forwardPass(batchX[xsJ]);
                    for(int layerK = 1; layerK < this.layers.length; layerK++){
                        currentState = this.layers[layerK].forwardPass(currentState);
                    }

                    //adds model output to batch predictions
                    batchPreds[xsJ] = currentState.clone();
//                System.out.println(Arrays.toString(currentState));

                }

                //calculates loss and gradient of yTrue, yPred
                double loss = this.loss.loss(batchPreds, batchY);
                double[] lossGradient = this.loss.lossGradient(batchPreds, batchY);

                //displays metrics to user
                if(showActualPred){
                    System.out.println(
                            "EPOCH: " + epochI +
                                    " STEP: " + batchI/batchSize +
                                    " LOSS: " +
                                    loss +
                                    " OUTPUT: " +
                                    Arrays.deepToString(batchPreds) +
                                    " ACTUAL: " +
                                    Arrays.deepToString(batchY) +
                                    " GRADIENT: " +
                                    Arrays.toString(lossGradient));
                } else {
                    System.out.println(
                            "EPOCH: " + epochI +
                                    " STEP: " + batchI/batchSize +
                                    " LOSS: " +
                                    loss);
                }


                //backpropagate loss to last layer
                Layer layerAt = this.layers[this.layers.length-1];
                currentState = layerAt.backPass(lossGradient);

//                System.out.println("GRADIENT");
//                System.out.println(Arrays.toString(currentState));
//                System.out.println("WEIGHT");
//                System.out.println(Arrays.deepToString(((DenseLayer) layerAt).getWeights()));
//                System.out.println("BIAS");
//                System.out.println(Arrays.toString(((DenseLayer) layerAt).getBiases()));
                //backpropagate gradients through rest of model
                for(int layerK = this.layers.length-2; layerK >= 0; layerK--){
                    layerAt = this.layers[layerK];
//                    if(layerAt instanceof DenseLayer){
//                        System.out.println(((DenseLayer) layerAt).getName());
//                    }
                    currentState = layerAt.backPass(currentState);

                }
            }

        }
    }
}
