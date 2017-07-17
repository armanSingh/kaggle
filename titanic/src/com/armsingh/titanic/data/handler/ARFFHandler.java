package com.armsingh.titanic.data.handler;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVSaver;

public class ARFFHandler {
	private static String DATA_FILE_ROOT = "data/"; 
	private static String ARFF = "arff";
	private static String CSV = "csv";
	private static String SLASH = "/";
	private static String DOT = ".";
	
	private ARFFHandler() {} 
	
	/**
	 * The method takes in the ARFF file name and saves the arff file.
	 * The ARFF file is expected to be present in data/arff/ directory. The CSV file is expected to be present in data/csv/ directory.
	 *  
	 * @param fileName The ARFF file name. 
	 * @return filePath The CSV file path.
	 * @throws IOException If no file with the specified name is found in the expected directories
	 */
	public static String saveAsCSV(String fileName) throws IOException {
		File sourceFile = new File(getARFFPath(fileName));
		ArffLoader arffLoader = new ArffLoader();
		arffLoader.setSource(sourceFile);
		
		//Get the instances from this data-set.
		Instances instances = arffLoader.getDataSet();
		
		File destinationFile = new File(getCSVPath(fileName));
		
		CSVSaver csvSaver = new CSVSaver();
		csvSaver.setInstances(instances);
		csvSaver.setFile(destinationFile);
		csvSaver.writeBatch();
		return destinationFile.getPath();
	}
	
	private static String getCSVPath(String fileName) {
		return DATA_FILE_ROOT + CSV + SLASH + fileName + DOT + CSV;
	}
	private static String getARFFPath(String fileName) {
		return DATA_FILE_ROOT + ARFF + SLASH + fileName + DOT + ARFF;
	}
}
