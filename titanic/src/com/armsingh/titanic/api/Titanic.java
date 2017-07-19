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
 *  Attempt 1: -- The submission earned a score of 0.76555, ranked 5322. --
 *  Attempt 2: -- Feature selection, score: 0.7752, ranked: 4038 
 * TODO: 1. Choose the right classifier, fine tune it for performance.
 *       2. Improve the score.
 * @author armsingh
 *
 */
public class Titanic {
	private static final String TRAINING_FILE_NAME = "train";
	private static final String TEST_FILE_NAME = "test";
	private static final String RESULT_FILE_NAME = "result";
	private static final String RESULT_FILE_PATH = "data/arff/" +RESULT_FILE_NAME + ".arff";
	
	public static void main(String[] args) throws Exception {
		// Save the CSV files in ARFF format
		String arffTrainFilePath = CSVHandler.saveAsARFF(TRAINING_FILE_NAME);
		String arffTestFilePath = CSVHandler.saveAsARFF(TEST_FILE_NAME);
		
		BufferedReader reader = new BufferedReader(new FileReader(arffTrainFilePath));
		Instances trainData = new Instances(reader);
		// Read the test data
		reader = new BufferedReader(new FileReader(arffTestFilePath));
		Instances testData = new Instances(reader);
		reader.close();
		
		testData.setClassIndex(testData.numAttributes() - 1);
		trainData.setClassIndex(trainData.numAttributes() - 1);
		
		//J48 expects the class to be non-numeric. Need to figure out why. Also, the attribute indices are wrt 1.
		Instances filteredTrainData = filterNumericToNominal(trainData, String.valueOf(trainData.classIndex()+1));
		Instances filteredTestData = filterNumericToNominal(testData, String.valueOf(testData.classIndex()+1));
		
		classifyAndSave(filteredTrainData, filteredTestData);
		System.out.println("------- Done -------");
	}
	
	/**
	 * For now this is hard coupled with J48 classifier.
	 * @param trainData
	 * @param testData
	 * @throws Exception
	 */
	private static void classifyAndSave(Instances trainData, Instances testData) throws Exception {
		
		J48 classifier = new J48();
		classifier.buildClassifier(trainData);
		
		for(int i=0; i<testData.numInstances(); ++i) {
			Double clas = classifier.classifyInstance(testData.instance(i));
			testData.instance(i).setClassValue(clas);
		}
		File resultFile = new File(RESULT_FILE_PATH);
		resultFile.createNewFile();
		
		//Write the contents to disk
		BufferedWriter writer = new BufferedWriter(new FileWriter(resultFile));
		writer.write(testData.toString());
		writer.newLine();
		writer.flush();
		writer.close();
		
		//save as a CSV file as well for easy access using excel.
		ARFFHandler.saveAsCSV(RESULT_FILE_NAME);
	}
	
	private static Instances filterNumericToNominal(Instances instances, String attributeIndex) throws Exception {
		NumericToNominal filter = new NumericToNominal();
		filter.setAttributeIndices(attributeIndex);
		filter.setInputFormat(instances);
		return Filter.useFilter(instances, filter);
	}
}