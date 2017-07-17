package com.armsingh.titanic.api;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import com.armsingh.titanic.data.handler.ARFFHandler;
import com.armsingh.titanic.data.handler.CSVHandler;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;


/**
 * This was my first submission, made solely from the point of view of checking the evaluation process.
 * The submission earned a score of 0.76555, ranked 5322. 
 * TODO: 1. Organize the code better
 *       2. Choose the right classifier, fine tune it for performance.
 *       3. Improve the score.
 * @author armsingh
 *
 */
public class Titanic {
	private static final String TRAINING_FILE_NAME = "train";
	private static final String TEST_FILE_NAME = "test";
	private static final String RESULT_FILE_NAME = "result";
	
	public static void main(String[] args) throws Exception {
		String arffTrainFilePath = CSVHandler.toARFF(TRAINING_FILE_NAME);
		String arffTestFilePath = CSVHandler.toARFF(TEST_FILE_NAME);
		
		BufferedReader reader = new BufferedReader(new FileReader(arffTrainFilePath));
		Instances trainData = new Instances(reader);
		reader.close();
		
		reader = new BufferedReader(new FileReader(arffTestFilePath));
		Instances testDataUnlabeled = new Instances(reader);
		testDataUnlabeled.setClassIndex(testDataUnlabeled.numAttributes()-1);
		reader.close();
		
		trainData.setClassIndex(trainData.numAttributes() - 1);
		
		NumericToNominal filter = new NumericToNominal();
		filter.setAttributeIndices(String.valueOf(trainData.classIndex() + 1));
		filter.setInputFormat(trainData);
		trainData = Filter.useFilter(trainData, filter);
		testDataUnlabeled = Filter.useFilter(testDataUnlabeled, filter);
		System.out.println(testDataUnlabeled.toSummaryString());
		
		J48 tree = new J48();
		tree.buildClassifier(trainData);
		
		for(int i=0; i<testDataUnlabeled.numInstances(); ++i) {
			Double clas = tree.classifyInstance(testDataUnlabeled.instance(i));
			testDataUnlabeled.instance(i).setClassValue(clas);
		}
		
		File resultFile = new File("data/arff/" + RESULT_FILE_NAME + ".arff");
		resultFile.createNewFile();
		BufferedWriter writer = new BufferedWriter(new FileWriter(resultFile));
		writer.write(testDataUnlabeled.toString());
		writer.newLine();
		writer.flush();
		writer.close();
		ARFFHandler.toCSV(RESULT_FILE_NAME);
		System.out.println(testDataUnlabeled.toSummaryString());
	}
}