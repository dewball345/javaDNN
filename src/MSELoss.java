import java.lang.Math;

public class MSELoss {

    public double loss(double[] yTrue, double[] yPred){
        double[] mseVal = new double[yTrue.length];
        for(int yI = 0; yI < mseVal.length; yI++){
            mseVal[yI] = (yTrue[yI] - yPred[yI]) * (yTrue[yI] - yPred[yI]);
        }

        double mseValSum = 0;
        for(double mseItem: mseVal){
            mseValSum += mseItem;
        }
        return mseValSum/mseVal.length;
    }

    public double[] lossGradient(double[] yTrue, double[] yPred){
        double[] mseValGradients = new double[yTrue.length];
        for(int yI = 0; yI < mseValGradients.length; yI++){
            mseValGradients[yI] = 2 * (yTrue[yI] - yPred[yI]);
        }

        return mseValGradients;
    }
}
