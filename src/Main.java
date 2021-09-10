import activations.SigmoidActivation;
import layer.DenseLayer;
import layer.Layer;
import losses.MSELoss;
import models.Model;

public class Main {
    public static void main(String[] args) {
//        double[][] weightTest = new double[][]{{1, 2, 3}, {4, 5, 6}};
//        System.out.println(weightTest[0][2]);
        System.out.println("hi lol");


        Model model = new Model(new Layer[]{
                new DenseLayer(1, 1, "dense_0"),
                new SigmoidActivation()
        }, 0.01, new MSELoss());

        //categorical example(if x > 3 output is 1 else output is 0)
        double[][] xs = new double[][]{{1}, {3}, {4}, {5}};
        double[][] ys = new double[][]{{0}, {0}, {1}, {1}};

        //loss will slowly decrease, and model will learn how to classify digits
        System.out.println("TRAIN WITH BATCH SIZE: ");
        model.train(xs, ys, 1000, 1, false);
//        System.out.println("SINGLE BATCH TRAINING: ");
//        model.trainSingleBatch(xs, ys, 100);
    }

}
