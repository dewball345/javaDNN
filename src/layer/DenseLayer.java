package layer;


import java.util.Random;

public class DenseLayer extends Layer {
    private double[][] weights;
    public double[][]  getWeights() {
        return weights;
    }

    private double[] biases;
    public double[] getBiases() {
        return biases;
    }

    private int startNeurons;
    private int endNeurons;

    private String name;
    public String getName() {
        return name;
    }


    private double[] outputs;
    private double[] inputs;

    private double lr;
    public void setLr(double lr){
        this.lr = lr;
    }

    Random random = new Random();

    //for syntactic sugar(let intellij know which comments to block)
    public void c_c(){}

    public DenseLayer(int startNeurons, int endNeurons, String name){
        this.startNeurons = startNeurons;
        this.endNeurons = endNeurons;
        this.name = name;
        //weight initialization
        this.weights = new double[startNeurons][endNeurons];
        for(int i = 0; i<startNeurons; i++){
            for(int j = 0; j<endNeurons; j++) {
                this.weights[i][j] = random.nextDouble();
            }
        }

        //bias initialization
        this.biases = new double[endNeurons];
        for(int j = 0; j<endNeurons; j++){
            this.biases[j] = random.nextDouble();
        }

        this.outputs = new double[endNeurons];
    }

    public double[] forwardPass(double[] inputs){
//        double[] outputs = new double[endNeurons];
        this.outputs = new double[endNeurons];
        //matrix multiplication
        //output[0] = input[0] * weight_matrix[0][0] + bias[0] + input[1] * weight+matrix[1][0] + bias[0]...
        //i * wT + b?
        this.inputs = inputs;

        for(int endI = 0; endI < endNeurons; endI++){
            for(int startJ = 0; startJ < startNeurons; startJ++){
                double input = inputs[startJ];
                double weight = this.weights[startJ][endI];
                double bias = this.biases[endI];
                //Replaced += with =
                this.outputs[endI] += input * weight + bias;
            }
        }

//        System.out.println(Arrays.toString(this.outputs));
        return this.outputs;
        //removed init of this.next
//        System.out.println(Arrays.toString(this.outputs));
//        if(this.next != null) {
//            this.next.forwardPass(this.outputs);
//        }
    }

    public double[] backPass(double[] gradients){
//        System.out.println(" " + Arrays.toString(gradients));
//        System.out.println(" " + Arrays.deepToString(this.weights));
        //empty array for gradients
        double[] createdWeightGradients = new double[this.startNeurons];
        //iterate through upstream gradients
        for(int gradientJ = 0; gradientJ < gradients.length; gradientJ++){
            //this is fc layer, so all inputs will have access to gradient
            for(int inputI = 0; inputI < this.startNeurons; inputI++){
                //weight gradient = upstream gradient(gradients[gradientJ]) * inner gradient(inputs[inputI])
                //aggregate the gradients by adding the new gradient to the createdGradients arra
                double currentGradient = gradients[gradientJ] * this.inputs[inputI];
                createdWeightGradients[inputI] += currentGradient;
                this.weights[inputI][gradientJ] -= currentGradient * this.lr;
                this.biases[gradientJ] -= 1 * gradients[gradientJ] * this.lr;
            }
        }

//        for(int gradientJ = 0; gradientJ < gradients.length; gradientJ++){
//            //this is fc layer, so all inputs will have access to gradient
//            for(int inputI = 0; inputI < this.startNeurons; inputI++){
//                System.out.println(" WEIGHT: "
//                        + inputI
//                        + " "
//                        + gradientJ);
//
//                System.out.println(
//                        " GRADIENT for weight "
//                                + createdWeightGradients[gradientJ]
////                                + " NAME: "
////                                + this.name
//                            );
//
//                System.out.println(" GRADIENT LENGTH: "
//                        + gradients.length
//                        + " CREATED GRADIENTS LENGTH: "
//                        + createdWeightGradients.length);
//
//                //updating gradients of weight
//
//
////                System.out.println("WEIGHT VALUE: " + this.weights[inputI][gradientJ]);
////                System.out.println("GRADIENT for bias " + inputI + " " + gradientJ + " " + gradients[gradientJ]);
//                c_c();
//                //inner gradient of bias is 1, multiply with upstream gradient
//
////                System.out.println("BIAS VALUE: " + this.biases[gradientJ]);
//
//            }
//        }



//        System.out.println(Arrays.toString(createdGradients));
        return createdWeightGradients;
    }
}
