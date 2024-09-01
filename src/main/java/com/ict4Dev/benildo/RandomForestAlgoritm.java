package com.ict4Dev.benildo;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

public class RandomForestAlgoritm {

        public static void main(String[] args) throws Exception {

            // Load the dataset
            BufferedReader reader = new BufferedReader(new FileReader("caracteristicas.arff"));
            Instances dataset = new Instances(reader);
            reader.close();

            // Set the class attribute as the last attribute and randomize the dataset
            dataset.setClassIndex(dataset.numAttributes() - 1);
            dataset.randomize(new java.util.Random(0));

            // Split the dataset into training and testing sets (80:20 split)
            int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
            int testSize = dataset.numInstances() - trainSize;
            Instances trainSet = new Instances(dataset, 0, trainSize);
            Instances testSet = new Instances(dataset, trainSize, testSize);

            // Build the classifier
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100);// Número de árvores na floresta
            rf.buildClassifier(trainSet);

            // Evaluate the classifier
            Evaluation evaluation = new Evaluation(testSet);
            evaluation.crossValidateModel(rf, testSet, 10, new Random(1));

            // Print the evaluation results
            System.out.println("=== Evaluation Results ===");
            System.out.println(evaluation.toSummaryString());
        }


}
