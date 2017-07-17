package com.armsingh.titanic.data.handler;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CSVHandler {
	private static String DATA_FILE_ROOT = "data/"; 
	private static String ARFF = "arff";
	private static String CSV = "csv";
	private static String SLASH = "/";
	private static String DOT = ".";
	
	private CSVHandler() {} 
	
	/**
	 * The method takes in the CSV file name and creates the arff file.
	 * The CSV file is expected to be present in data/csv/ directory. The ARFF file is expected to be present in data/arff/ directory.
	 *  
	 * @param fileName The CSV file name. 
	 * @return filePath The ARFF file path.
	 * @throws IOException If no file with the specified name is found in the expected directories
	 */
	public static String saveAsARFF(String fileName) throws IOException {
		File sourceFile = new File(getCSVPath(fileName));
		CSVLoader csvLoader = new CSVLoader();
		csvLoader.setSource(sourceFile);
		
		//Get the instances from this data-set.
		Instances instances = csvLoader.getDataSet();
		
		File destinationFile = new File(getARFFPath(fileName));
		
		ArffSaver arffSaver = new ArffSaver();
		arffSaver.setInstances(instances);
		arffSaver.setFile(destinationFile);
		arffSaver.writeBatch();
		return destinationFile.getPath();
	}
	
	private static String getCSVPath(String fileName) {
		return DATA_FILE_ROOT + CSV + SLASH + fileName + DOT + CSV;
	}
	private static String getARFFPath(String fileName) {
		return DATA_FILE_ROOT + ARFF + SLASH + fileName + DOT + ARFF;
	}
}
