import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
//        double[][] weightTest = new double[][]{{1, 2, 3}, {4, 5, 6}};
//        System.out.println(weightTest[0][2]);
        System.out.println("hi lol");


        Model model = new Model(new Layer[]{
                new DenseLayer(1, 1, "dense_0"),
                new SigmoidActivation()
        }, 0.01, new MSELoss());

        //example formula is [x^2]
        double[][] xs = new double[][]{{1}, {3}, {4}, {5}};
        double[][] ys = new double[][]{{0}, {0}, {1}, {1}};

        //loss will slowly decrease
        System.out.println("TRAIN WITH BATCH SIZE: ");
        model.train(xs, ys, 100, 1, false);
//        System.out.println("SINGLE BATCH TRAINING: ");
//        model.trainSingleBatch(xs, ys, 100);
    }

}
