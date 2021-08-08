import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
//        double[][] weightTest = new double[][]{{1, 2, 3}, {4, 5, 6}};
//        System.out.println(weightTest[0][2]);
        System.out.println("hi lol");


        Model model = new Model(new Layer[]{
                new DenseLayer(2, 3, "dense_0"),
                new ReluActivation(),
                new DenseLayer(3, 2, "dense_1"),
                new ReluActivation(),
                new DenseLayer(2, 2, "dense_2")
        }, 0.001, new MSELoss());

        //example formula is [x^2, y^2]
        double[][] xs = new double[][]{new double[]{1.0, 2.0}, new double[]{3.0, 4.0}};
        double[][] ys = new double[][]{new double[]{1.0, 4.0}, new double[]{9.0, 16.0}};

        //loss will slowly decrease
        model.train(xs, ys, 100);
    }
}
