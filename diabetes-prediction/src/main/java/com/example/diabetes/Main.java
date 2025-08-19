package com.example.diabetes;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;

public class Main {

    private static final String DEFAULT_DATASET = "data/pima-indians-diabetes.arff";
    private static final String DEFAULT_MODEL = "models/random-forest.model";

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            printUsageAndExit();
            return;
        }

        String command = args[0];
        switch (command) {
            case "train" -> train(Arrays.copyOfRange(args, 1, args.length));
            case "evaluate" -> evaluate(Arrays.copyOfRange(args, 1, args.length));
            case "predict" -> predict(Arrays.copyOfRange(args, 1, args.length));
            default -> {
                System.err.println("Unknown command: " + command);
                printUsageAndExit();
            }
        }
    }

    private static void printUsageAndExit() {
        System.out.println("Diabetes Prediction CLI (Weka)\n" +
                "Usage:\n" +
                "  java -jar diabetes-prediction.jar train [dataset.csv] [modelPath]\n" +
                "  java -jar diabetes-prediction.jar evaluate [dataset.csv]\n" +
                "  java -jar diabetes-prediction.jar predict <modelPath> <feature1,feature2,...>\n" +
                "\nDefaults:\n" +
                "  dataset: " + DEFAULT_DATASET + "\n" +
                "  model:   " + DEFAULT_MODEL + "\n");
    }

    private static void train(String[] args) throws Exception {
        String datasetPath = args.length >= 1 ? args[0] : DEFAULT_DATASET;
        String modelPath = args.length >= 2 ? args[1] : DEFAULT_MODEL;

        Instances data = loadCsvAsInstances(datasetPath);
        data.setClassIndex(data.numAttributes() - 1);

        RandomForest rf = new RandomForest();
        rf.setNumIterations(200);
        rf.setNumFeatures(0); // auto
        rf.setSeed(42);

        rf.buildClassifier(data);

        Path modelFilePath = Paths.get(modelPath);
        Files.createDirectories(modelFilePath.getParent());
        SerializationHelper.write(modelPath, rf);

        System.out.println("Model trained and saved to: " + modelPath);
    }

    private static void evaluate(String[] args) throws Exception {
        String datasetPath = args.length >= 1 ? args[0] : DEFAULT_DATASET;

        Instances data = loadCsvAsInstances(datasetPath);
        data.setClassIndex(data.numAttributes() - 1);

        RandomForest rf = new RandomForest();
        rf.setNumIterations(200);
        rf.setSeed(42);

        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(rf, data, 10, new Random(42));

        System.out.println(eval.toSummaryString("=== 10-fold CV Summary ===", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    private static void predict(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("predict requires: <modelPath> <commaSeparatedFeatures>");
            return;
        }

        String modelPath = args[0];
        String featureCsv = args[1];

        Classifier model = (Classifier) SerializationHelper.read(modelPath);

        // Build a one-row Instances with same header as training CSV
        Instances structure = loadCsvAsInstances(DEFAULT_DATASET);
        structure.setClassIndex(structure.numAttributes() - 1);

        // Create a row from CSV values
        String[] parts = featureCsv.split(",");
        if (parts.length != structure.numAttributes() - 1) {
            throw new IllegalArgumentException("Expected " + (structure.numAttributes() - 1) + " features, got " + parts.length);
        }

        // Compose an ARFF string with one instance
        StringBuilder arff = new StringBuilder();
        arff.append("@RELATION diabetes\n\n");
        for (int i = 0; i < structure.numAttributes() - 1; i++) {
            String name = structure.attribute(i).name();
            arff.append("@ATTRIBUTE ").append(name).append(" NUMERIC\n");
        }
        arff.append("@ATTRIBUTE class {0,1}\n\n");
        arff.append("@DATA\n");
        arff.append(String.join(",", parts)).append(",?");

        Instances toPredict = new ConverterUtils.DataSource(new java.io.ByteArrayInputStream(arff.toString().getBytes())).getDataSet();
        toPredict.setClassIndex(toPredict.numAttributes() - 1);

        double labelIdx = model.classifyInstance(toPredict.instance(0));
        double[] dist = model.distributionForInstance(toPredict.instance(0));
        int label = (int) labelIdx;
        System.out.printf("Predicted class: %d (prob=%.4f)%n", label, dist[label]);
    }

    private static Instances loadCsvAsInstances(String path) throws Exception {
        if (!new File(path).exists()) {
            throw new IllegalArgumentException("Dataset not found: " + path +
                    ". Place CSV at that path or pass an explicit path.");
        }
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
        Instances data = source.getDataSet();
        return data;
    }
}

