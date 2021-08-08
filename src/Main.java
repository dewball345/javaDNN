import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        System.out.println("hi lol");
        Model model = new Model(new DenseLayer[]{
           new DenseLayer(1, 1, "dense_0")
        }, 0.01, new MSELoss());

        double[][] xs = new double[][]{new double[]{0.0, 1.0}, new double[]{1.0, 1.0}};
        double[][] ys = new double[][]{new double[]{1.0}, new double[]{3.0}};

        model.train(xs, ys, 100);
    }
}
