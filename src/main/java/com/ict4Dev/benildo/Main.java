package com.ict4Dev.benildo;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.BufferedReader;
import java.io.FileReader;

public class Main {
    public static void main(String[] args) throws Exception {
        // Load the dataset
        BufferedReader reader = new BufferedReader(new FileReader("caracteristicas.arff"));
        Instances dataset = new Instances(reader);
        reader.close();

        //if the atributes are numeric
        //NumericToNominal convert = new NumericToNominal();
        //convert.setAttributeIndices("first-6,10-last");
        //convert.setInputFormat(dataset0);
        //Instances dataset = Filter.useFilter(dataset0, convert);

        // Set the class attribute as the last attribute and randomize the dataset
        dataset.setClassIndex(dataset.numAttributes() - 1);
        dataset.randomize(new java.util.Random(0));

        // Split the dataset into training and testing sets (80:20 split)
        int seed = 1;
        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
        int testSize = dataset.numInstances() - trainSize;
        Instances trainSet = new Instances(dataset, 0, trainSize);
        Instances testSet = new Instances(dataset, trainSize, testSize);

        // Build and evaluate the classifier
        NaiveBayes classifier = new NaiveBayes();
        classifier.buildClassifier(trainSet);
        Evaluation evaluation = new Evaluation(trainSet);
        evaluation.evaluateModel(classifier, testSet);

        // Print the evaluation results
        System.out.println("=== Evaluation Results NaiveBayes===");
        System.out.println(evaluation.toSummaryString());
        System.out.println();

        // Build and evaluate the classifier J48
        J48 classifierJ48 = new J48();
        classifierJ48.buildClassifier(trainSet);
        Evaluation evaluationJ48 = new Evaluation(trainSet);
        evaluationJ48.evaluateModel(classifierJ48, testSet);

        // Print the evaluation results
        System.out.println("=== Evaluation Results J48===");
        System.out.println(evaluationJ48.toSummaryString());
    }
}