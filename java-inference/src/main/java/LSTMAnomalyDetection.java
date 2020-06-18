
import java.util.Random;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class LSTMAnomalyDetection {

    String modelsDir;
    Session speedModel;
    Session rpmModel;
    Session throttleModel;

    double speedAnomalyThreshold;
    double rpmAnomalyThreshold;
    double throttleAnomalyThreshold;

    private int historySize;

    public int getHistorySize() {
        return historySize;
    }

    public LSTMAnomalyDetection(String modelsDir, int historySize){
        this.modelsDir = modelsDir;
        speedModel = SavedModelBundle.load(this.modelsDir + "/speed/1/", "serve").session();
        rpmModel = SavedModelBundle.load(this.modelsDir + "/rpm/1/", "serve").session();
        throttleModel = SavedModelBundle.load(this.modelsDir + "/throttle/1/", "serve").session();

        this.speedAnomalyThreshold = 0.25;
        this.throttleAnomalyThreshold = 0.25;
        this.rpmAnomalyThreshold = 0.25;

        this.historySize = historySize;
    }


    private double meanAbsoluteError(float[][][] input, float[][][] prediction) {
        double sum = 0.0;
        for (int i = 0; i < getHistorySize(); i++) {
            sum += Math.abs(input[0][i][0] - prediction[0][i][0]);
        }

        return sum / getHistorySize(); //mean absolute error


    }

    public double calculateSpeedAnomalyScore(float[][][] input)
    {
        Tensor inputTensor = Tensor.create(input, Float.class);
        Tensor predictionTensor = speedModel.runner().feed("lstm_2_input", inputTensor).fetch("time_distributed_1/Reshape_1").run().get(0);
        float[][][] prediction = new float[1][getHistorySize()][1];
        predictionTensor.copyTo(prediction);
        double mae =  meanAbsoluteError(input, prediction);
        return mae < this.speedAnomalyThreshold ? 0.0 : mae;
    }

    public double calculateRPMAnomalyScore(float[][][] input)
    {
        Tensor inputTensor = Tensor.create(input, Float.class);
        Tensor predictionTensor = rpmModel.runner().feed("lstm_2_input", inputTensor).fetch("time_distributed_1/Reshape_1").run().get(0);
        float[][][] prediction = new float[1][getHistorySize()][1];
        predictionTensor.copyTo(prediction);
        double mae =  meanAbsoluteError(input, prediction);
        return mae < this.rpmAnomalyThreshold ? 0.0 : mae;
    }

    public double calculateThrottleAnomalyScore(float[][][] input)
    {
        Tensor inputTensor = Tensor.create(input, Float.class);
        Tensor predictionTensor = throttleModel.runner().feed("lstm_2_input", inputTensor).fetch("time_distributed_1/Reshape_1").run().get(0);
        float[][][] prediction = new float[1][getHistorySize()][1];
        predictionTensor.copyTo(prediction);
        double mae =  meanAbsoluteError(input, prediction);
        return mae < this.throttleAnomalyThreshold ? 0.0 : mae;
    }

    // This is for testing
    public static void main(String[] args)  {

        int  historySize = 150;
        LSTMAnomalyDetection lstmAnomaly = new LSTMAnomalyDetection("/home/sakkas/github/indycar-lstm-anomalydetection-java/models", historySize);

        Random rand = new Random();

        //Generate random input to test it.
        float[][][] testData = new float[1][historySize][1];
        for(int i=0; i < historySize ;i++){
            testData[0][i][0] = rand.nextFloat();
            //System.out.println(testData[0][i][0]);
        }

        double speedAnomalyScore = lstmAnomaly.calculateSpeedAnomalyScore(testData);
        double rpmAnomalyScore = lstmAnomaly.calculateRPMAnomalyScore(testData);
        double throttleAnomalyScore = lstmAnomaly.calculateThrottleAnomalyScore(testData);

        System.out.println("Speed anomaly score:" + speedAnomalyScore);
        System.out.println("RPM anomaly score:" + rpmAnomalyScore);
        System.out.println("Throttle anomaly score:" + throttleAnomalyScore);
        //// TODO: 6/17/20 All three models give same output. Check if something wrong.

        benchmark(testData);

    }

    public static void benchmark(float[][][] testData)
    {
        int  historySize = 150;
        LSTMAnomalyDetection lstmAnomaly = new LSTMAnomalyDetection("/home/sakkas/github/indycar-lstm-anomalydetection-java/models", historySize);
        long start = System.currentTimeMillis();
        for (int i = 0; i < 1000; i++) {
            lstmAnomaly.calculateSpeedAnomalyScore(testData);
        }
        long finish = System.currentTimeMillis();
        long timeElapsed = finish - start;
        System.out.println("elapsed time: " + timeElapsed);

    }
}
