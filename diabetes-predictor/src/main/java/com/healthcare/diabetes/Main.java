package com.healthcare.diabetes;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Debug;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class Main {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.out.println("Usage:\n  train <dataset.arff> <model.bin>\n  predict <model.bin> <csvFeatures>\n\nExample features CSV: pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,diabetesPedigree,age");
            return;
        }

        String command = args[0];
        switch (command) {
            case "train":
                if (args.length < 3) {
                    throw new IllegalArgumentException("train requires <dataset.arff> <model.bin>");
                }
                trainModel(args[1], args[2]);
                break;
            case "predict":
                if (args.length < 3) {
                    throw new IllegalArgumentException("predict requires <model.bin> <csvFeatures>");
                }
                predictFromCsv(args[1], args[2]);
                break;
            default:
                throw new IllegalArgumentException("Unknown command: " + command);
        }
    }

    static void trainModel(String arffPath, String modelOutPath) throws Exception {
        DataSource source = new DataSource(arffPath);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        data.randomize(new Random(42));
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        RandomForest model = new RandomForest();
        model.setNumIterations(200);
        model.setMaxDepth(10);
        model.buildClassifier(train);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, test);
        System.out.printf("Accuracy: %.4f, AUC: %.4f, F1: %.4f\n",
                1.0 - eval.errorRate(), eval.areaUnderROC(1), eval.fMeasure(1));

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelOutPath))) {
            oos.writeObject(model);
        }
        System.out.println("Model saved to: " + modelOutPath);
    }

    static void predictFromCsv(String modelPath, String csvFeatures) throws Exception {
        Classifier model = (Classifier) weka.core.SerializationHelper.read(modelPath);

        // Expect 8 numeric features in order of the Pima Diabetes dataset
        String[] parts = csvFeatures.split(",");
        if (parts.length != 8) {
            throw new IllegalArgumentException("Expected 8 numeric features in CSV");
        }

        // Build structure from embedded ARFF header
        Instances structure = new Instances(new java.io.BufferedReader(new java.io.StringReader(ArffSchemas.pimaHeader())));
        structure.setClassIndex(structure.numAttributes() - 1);

        double[] values = new double[8 + 1];
        for (int i = 0; i < 8; i++) {
            values[i] = Double.parseDouble(parts[i]);
        }
        // Placeholder class value
        values[8] = weka.core.Utils.missingValue();

        DenseInstance instance = new DenseInstance(1.0, values);
        instance.setDataset(structure);
        double pred = model.classifyInstance(instance);
        double[] dist = model.distributionForInstance(instance);
        int clsIndex = (int) pred;
        String label = structure.classAttribute().value(clsIndex);
        System.out.printf("Prediction: %s (prob=%.4f)\n", label, dist[clsIndex]);
    }
}

