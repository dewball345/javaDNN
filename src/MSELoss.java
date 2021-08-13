import java.lang.Math;

public class MSELoss {

    //loss for each data point
    public double lossStep(double[] yTrue, double[] yPred){
        double[] mseVal = new double[yTrue.length];

        //mse is (yTrue-yPred) ^ 2
        for(int yI = 0; yI < mseVal.length; yI++){
            mseVal[yI] = (yTrue[yI] - yPred[yI]) * (yTrue[yI] - yPred[yI]);
        }

        //compute the mean
        double mseValSum = 0;

        //iterates through mseVal to get sum
        for(int mseI = 0; mseI < mseVal.length; mseI++){
            mseValSum += mseVal[mseI];
        }

        //mean is sum/length
        return mseValSum/mseVal.length;
    }

    //loss for batch of datapoints
    public double loss(double[][] yTrue, double[][] yPred){
        double[] mseVals = new double[yTrue.length];

        //calculates loss at each item in batch
        for(int yI = 0; yI < mseVals.length; yI++){
            double[] currentY = yTrue[yI];
            double[] currentPred = yPred[yI];
            mseVals[yI] = lossStep(currentY, currentPred);
        }

        //computes mean of all the losses
        double mseValSum = 0;
        //finds sum of all the loses
        for(int mseI = 0; mseI < mseVals.length; mseI++){
            mseValSum += mseVals[mseI];
        }
        //mean = sum/length
        return mseValSum/mseVals.length;
    }

    //gradient for each loss step
    public double[] lossStepGradient(double[] yTrue, double[] yPred){
        double[] mseValGradients = new double[yTrue.length];
        //gradient is 2 * (output - actual)
        for(int yI = 0; yI < mseValGradients.length; yI++){
            mseValGradients[yI] = 2 * (yTrue[yI] - yPred[yI]);
        }

        return mseValGradients;
    }

    //gradients for batch
    public double[] lossGradient(double[][] yTrue, double[][] yPred){
        double[][] mseValsGradients = new double[yTrue.length][];

        //calculates gradients for each item in batch
        for(int yI = 0; yI < mseValsGradients.length; yI++){
            mseValsGradients[yI] = lossStepGradient(yTrue[yI], yPred[yI]);
        }

        //gets means of all the gradients
        double[] gradientMeans = new double[mseValsGradients[0].length];
        //first sums up all the gradients together
        for(int mseI = 0; mseI < mseValsGradients.length; mseI++){
            double[] currentGradient = mseValsGradients[mseI];
            for(int gradI = 0; gradI < currentGradient.length; gradI++){
                gradientMeans[gradI] += currentGradient[gradI];
            }
        }
        //then divides by length of gradients(mean = sum/length)
        for(int mseI = 0; mseI < gradientMeans.length; mseI++){
            gradientMeans[mseI] /= mseValsGradients.length;
        }

        return gradientMeans;
    }
}
