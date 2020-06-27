package cli;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

//import benchmark.trial.GramTrial;
import datastructure.FeatureVector;
import datastructure.Pair;
import datastructure.SparseFeatureVector;

// TODO: support option for computing stddev from each accuracy; 
// over all individual folds!

// TODO: add option for stratified cross validation, i.e., all folds
// should have approximately the same distribution of class labels

/**
 * Computes prediction accuracies using libSVM by cross-validation
 * based on the gram files in a given <code>FOLDER</code>, which all 
 * must be in libSVM format. Each file must be named as follows:
 * <code>DATASET__KERNEL_PARAMETERS.gram</code>
 * 
 * For each dataset and each kernel the accuracy is computed, the 
 * parameters used for prediction as well as the regularization 
 * constant C are selected by cross-validation for each training 
 * fold separately.  
 * 
 * The accuracies are stored in the file <code>rAccFOLDER.txt</code>, 
 * the file <code>rAccSelFOLDER.txt</code> stores information on the 
 * parameters selected. The number of repetitions determines the 
 * number of random fold assignments. The accuracies are averaged 
 * over these repetitions and standard deviation is reported w.r.t.
 * to the results obtained for different fold assignments.
 * 
 * The behavior of the class is configured by several static variables
 * which should be checked carefully.
 * 
 * 
 * @author xxx
 *
 */
public class AccuracyTest {

	/** 
	 * The regularization constant of the C-SVM is selected by cross-validation 
	 * from these values.
	 */
	public static double[] C_RANGE = {
//		Math.pow(10, -7),
//		Math.pow(10, -6),
//		Math.pow(10, -5),
//		Math.pow(10, -4),
		Math.pow(10, -3),
		Math.pow(10, -2),
		Math.pow(10, -1),
		Math.pow(10, 0),
		Math.pow(10, 1),
		Math.pow(10, 2),
		Math.pow(10, 3)
//		Math.pow(10, 4),
//		Math.pow(10, 5),
//		Math.pow(10, 6),
//		Math.pow(10, 7)
	};

	/**
	 * The path containing the considered FOLDER, result files will be places here.
	 */
	//public String BASE_PATH = GramTrial.BASE_PATH;
        public String BASE_PATH = "./";
	
	/**
	 * Used to write temporary files; heavy read/write access in this folder! 
	 */
	public String TMP_PATH = "/tmp/";
	
	/**
	 * The number of parallel threads used. This corresponds to the number
	 * of libSVM processes running simultaneously.
	 */
	public int PARALLEL_THREADS = 24;
	
	/** 
	 * The maximum number of milliseconds to wait before svm processes are killed; set to 
	 * 0 to deactivate timeout.
	 * This was introduced for the following reason:
	 * libSVM often converges slowly for inappropriate choices of the constant C, in this 
	 * case a timeout can be useful, since these choices of C typically lead to bad results.
	 * However, for large data sets libSVM may also require a long time. Setting this variable 
	 * to a high value or deactivating timeouts is safe!
	 */ 
	public long TIMEOUT = 1000000; // ms
	
	/** The number of folds used for accuracy calculation */
	public int FOLDS = 10;
	
	/** Number of folds for learning C; libSVM Parameter -v n-fold cross validation mode */
	public int X_VAL_PARA_OPT_FOLDS = FOLDS;
	
	/** 
	 * Number of repetitions, each repetitions has a different fold assignment. The 
	 * reported standard deviation refers to these repetitions. 
	 */
	public int REPETITIONS = 10;
	
	/** If true all command line calls and libSVM output are printed out*/
	public final boolean DEBUG = false;
	
	/**
	 * Allows to specify the path to libSVM, i.e., the folder containing svm-train and 
	 * svm-predict; leave blank if both are available in the current path.
	 */
	//public String LIBSVM_PATH = "/usr/bin/";
	public String LIBSVM_PATH = "/usr/local/Cellar/libsvm/3.23/bin/";

	
	/**
	 * Gram matrices with a file name containing the stop word are ignored; set to
	 * <code>null</code> to deactivate.
	 */
	public String BLACKLIST_STOP_WORD = null;
	
	
	//------------- no configuration below this line is required -------------
	public String LOG_FILE_PREFIX = "rAcc";
	public Random rng;
	
	
	private String pFolder;
	private String pDataset;
	
	/**
	 * Creates an object for accuracy testing.
	 * @param folder the folder containing the gram files.
	 * @param folder
	 */
	public AccuracyTest(String folder) {
		this(folder, null);
	}
	
	/**
	 * Creates an object for accuracy testing.
	 * @param folder the folder containing the gram files.
	 * @param dataset the data set used for testing; null to test all data sets.
	 */
	public AccuracyTest(String folder, String dataset) {
		this.pFolder = folder;
		this.pDataset = dataset;
		
		// delete old results file
		new File(BASE_PATH+LOG_FILE_PREFIX+folder+".txt").delete();
		new File(BASE_PATH+LOG_FILE_PREFIX+"Sel"+folder+".txt").delete();
	}
	
	/**
	 * Starts accuracy tests.
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public void run() throws IOException, InterruptedException {
		
		// retrieve list of data sets
		HashSet<String> datasets;
		if (pDataset != null) {
			datasets = new HashSet<String>();
			datasets.add(pDataset);
		} else {
			datasets = getDatsets();
		}
		
		// test data sets
		for (String ds : datasets) {
			System.out.println(ds);
			HashSet<String> kernels = getKernels(ds);
			for (String bk : kernels) {
				System.out.println("\t"+bk);
				ArrayList<Gram> grams = getGrams(ds, bk);
				for (Gram g : grams) {
					System.out.println("\t\t"+g.name);
				}
				AccuracyResult r = computeAccurcay(pFolder, grams, REPETITIONS, FOLDS);
				addResult(ds, bk, r.accuracy, r.stdev);
				addSelectionResult(r.parameterSelection);
			}
		}
	}
		
	private void addResult(String dataset, String kernel, double accuracy, double stdev) throws IOException {
		FileWriter fw = new FileWriter(BASE_PATH+LOG_FILE_PREFIX+pFolder+".txt", true);
		BufferedWriter bw = new BufferedWriter(fw);
		bw.append(dataset + "\t" + kernel + "\t" + accuracy + "\t" + stdev + "\n");
		bw.close();
	}
	
	private void addSelectionResult(FeatureVector<String> sel) throws IOException {
		FileWriter fw = new FileWriter(BASE_PATH+LOG_FILE_PREFIX+"Sel"+pFolder+".txt", true);
		BufferedWriter bw = new BufferedWriter(fw);
		for (Entry<String, Double> e : sel.nonZeroEntries()) {
			bw.append(e.getKey() + "\t" + e.getValue() + "\n");
		}
		bw.close();
	}

	/**
	 * The data sets in the considered folder.
	 * @return list of available data sets in the folder
	 */
	private HashSet<String> getDatsets() {
		HashSet<String> ds = new HashSet<String>();
		File folder = new File(BASE_PATH+pFolder);
		for (File f : folder.listFiles()) {
			ds.add(f.getName().split("__")[0]);
		}
		return ds;
	}
	
	/**
	 * The kernels for the given data set. Multiple gram files
	 * may be contained in the folder for each kernel and data set 
	 * having different parameters.
	 * @param dataset the data set
	 * @return list of available kernels for the given data set
	 */
	public HashSet<String> getKernels(String dataset) {
		HashSet<String> bk = new HashSet<String>();
		File folder = new File(BASE_PATH+pFolder);
		for (File f : folder.listFiles()) {
			if (!f.getName().endsWith(".gram")) continue;
			if (f.getName().startsWith(dataset))
				bk.add(f.getName().split("__")[1].split("_")[0]);
		}
		return bk;
	}
	
	/**
	 * Loads all gram files for the given data set and kernel.
	 * @param dataset the data set name
	 * @param kernel the kernel name
	 * @return gram matrices, obtained for each parameterization
	 * @throws IOException
	 */
	private ArrayList<Gram> getGrams(String dataset, String kernel) throws IOException {
		ArrayList<Gram> grams = new ArrayList<Gram>();
		File folder = new File(BASE_PATH+pFolder);
		for (File f : folder.listFiles()) {
			if (f.getName().startsWith(dataset+"__"+kernel+"_"))
				if (BLACKLIST_STOP_WORD == null || !f.getName().contains(BLACKLIST_STOP_WORD))
					grams.add(loadGram(f));
		}
		return grams;
	}
	
	/**
	 * Loads a single gram file into memory.
	 * @param file the file
	 * @return a container storing the gram file
	 * @throws IOException
	 */
	private static Gram loadGram(File file) throws IOException {
		ArrayList<String> r = new ArrayList<String>();
		
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		while ((line = br.readLine()) != null) {
			r.add(line);
		}
		br.close();
		
		return new Gram(file.getName(), r);
	}

	
	/**
	 * Container for the data stored in a gram file. 
	 * @author Nils Kriege
	 */
	public static class Gram {
		public String name;
		public ArrayList<String> gram;
		
		public Gram(String name, ArrayList<String> gram) {
			this.name = name;
			this.gram = gram;
		}
	}
	
	/**
	 * Applies the given partition the gram data.
	 * @param data content of a gram file (lines)
	 * @param partitionIndices the partition
	 * @return partition of the data
	 */
	public static ArrayList<ArrayList<String>> partitionGram(ArrayList<String> data, ArrayList<ArrayList<Integer>> partitionIndices) {
		int n = partitionIndices.size();
		ArrayList<ArrayList<String>> r = new ArrayList<ArrayList<String>>(n);
		for (int i=0; i<n; i++) {
			r.add(new ArrayList<String>());
		}

		for (int i=0; i<partitionIndices.size(); i++) {
			ArrayList<Integer> part = partitionIndices.get(i);
			for (Integer lineIndex : part) {
				r.get(i).add(data.get(lineIndex));
			}
		}
		
		return r;
	}
	
	/**
	 * Creates a random partition for of the set {0, ..., n-1} into the given number
	 * of cells.
	 * @param n number of elements
	 * @param cells number of cells of the desired partition
	 * @return the partition
	 */
	public ArrayList<ArrayList<Integer>> partition(int n, int cells) {
		ArrayList<ArrayList<Integer>> r = new ArrayList<ArrayList<Integer>>(cells);
		for (int i=0; i<cells; i++) {
			r.add(new ArrayList<Integer>());
		}
		
		ArrayList<Integer> pool = new ArrayList<Integer>(n);
		for (int i=0; i<n; i++) pool.add(i);
		
		int i=0;
		while (!pool.isEmpty()) {
			Integer e = pool.remove(rng.nextInt(pool.size()));
			r.get(i).add(e);			
			i = ++i%cells;
		}
		
		return r;
	}
	
	/**
	 * Computes the accuracy by cross validation; all grams are split according
	 * to randomly chosen folds. For each training fold the gram matrix performing
	 * best is selected and used for prediction on the test fold. 
	 * @param id identifier used for temporary files
	 * @param grams the grams (typically different parameterizations of a kernel)
	 * @param repetitions number of repetitions, i.e., different fold assignments
	 * @param foldNo the number of folds
	 * @return the accuracy result
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public AccuracyResult computeAccurcay(String id, ArrayList<Gram> grams, int repetitions, int foldNo) throws IOException, InterruptedException {
		
		double[] accuracy = new double[repetitions];

		SparseFeatureVector<String> selections = new SparseFeatureVector<String>();
		
		rng = new Random(42); // same partitions for each call
		for (int i=0; i<repetitions; i++) {
			System.out.println();
			System.out.println("REPETITION  "+(i+1)+"  of  "+repetitions);
			ArrayList<ArrayList<Integer>> foldIndices = partition(grams.get(0).gram.size(), foldNo);
			double acc = crossValidate(foldIndices, grams, "svmlib"+id, selections);
			accuracy[i] = acc;
			System.out.println("Avg. Accuracy: "+acc);
		}
		System.out.println("========================");
		double avg = 0;
		for (double d : accuracy) avg += d;
		avg /= repetitions;
		
		// compute stdev
		double stdev = 0;
		for (double d : accuracy) stdev += (d-avg)*(d-avg);
		stdev = Math.sqrt(stdev/repetitions);
		
		System.out.println("Accuracy: "+avg);
		System.out.println("Standard deviation: "+stdev);

		return new AccuracyResult(avg, stdev, selections);
	}
	
	/**
	 * Container for the results of a classification experiment.
	 * 
	 * @author Nils Kriege
	 */
	public static class AccuracyResult {
		public double accuracy;
		public double stdev;
		public FeatureVector<String> parameterSelection;
		
		public AccuracyResult(double accuracy, double stdev, FeatureVector<String> parameterSelection) {
			this.accuracy = accuracy;
			this.stdev = stdev;
			this.parameterSelection = parameterSelection;
		}
	}
	

	/**
	 * Computes the accuracy for a given fold assignment. Each fold serves as test 
	 * fold once; the resulting accuracies are averaged. The C parameter and 
	 * kernel parameters (different grams) are selected on the training fold.
	 * @param folds the fold partition given by indices
	 * @param grams the grams (kernel parameterizations)
	 * @param prefix to distinguish temporary files
	 * @param selections used to store the selected paramters (gram and C)
	 * @return average accuracy for the given fold assignment
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public double crossValidate(ArrayList<ArrayList<Integer>> folds, ArrayList<Gram> grams, String prefix, FeatureVector<String> selections) throws IOException, InterruptedException {
		if (DEBUG) {
			System.out.println("Cross-Validation");
		}
		
		// split grams by folds
		ArrayList<ArrayList<ArrayList<String>>> gramsFolds = new ArrayList<ArrayList<ArrayList<String>>>();
		for (Gram g : grams) {
			gramsFolds.add(partitionGram(g.gram, folds));
		}
		
		double avgAccuracy=0;
		for (int iFold=0; iFold<folds.size(); iFold++) {
			
			System.out.println("\tFold  "+(iFold+1)+"  of  "+folds.size());
			
			// iFold indexes the independent evaluation (test) set
			ArrayList<String> names = new ArrayList<String>();
			for (int iGram=0; iGram<grams.size(); iGram++) {
				// write train and test files for each gram
				String name = prefix+"_"+grams.get(iGram).name;
				names.add(name);
				
				ArrayList<String> fold = gramsFolds.get(iGram).get(iFold);
				
				// write test file
				String testFile = name+".test";
				BufferedWriter bw = new BufferedWriter(new FileWriter(TMP_PATH+testFile));
				for (String line : fold) {
					bw.write(line+"\n");
				}
				bw.close();
				
				// write train file
				String trainFile = name+".train";
				bw = new BufferedWriter(new FileWriter(TMP_PATH+trainFile));
				for (int iFold2=0; iFold2<folds.size(); iFold2++) {
					if (iFold2 != iFold) {
						fold = gramsFolds.get(iGram).get(iFold2);
						for (String line : fold) {
							bw.write(line+"\n");
						}
					}
				}
				bw.close();
			}
			
			// parameter optimization
			Pair<String,Double> par = searchBestParameter(names);
			selections.increaseByOne(par.getFirst());
			
			// build model
			String modelFile = buildModel(par.getFirst()+".train", par.getSecond());
			
			// predict
			double accuracy = predict(modelFile, par.getFirst()+".test");
			System.out.println("\tReached Accuracy: "+accuracy);
			avgAccuracy += accuracy/folds.size();
			
			// delete files
			for (String name : names) {
				new File(TMP_PATH+name+".train").delete();
				new File(TMP_PATH+name+".test").delete();
			}
			new File(TMP_PATH+par.getFirst()+".train.model").delete();
			new File(TMP_PATH+par.getFirst()+".train.model.out").delete();
		}
		
//		System.out.println("Avg. Accuracy "+avgAccuracy);
		return avgAccuracy;
	}
	
	/**
	 * Finds the best parameters based on the test set
	 * @param names available kernels
	 * @return name of the best kernel and C parameter
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public Pair<String,Double> searchBestParameter(ArrayList<String> names) throws IOException, InterruptedException {

		ArrayList<Pair<String,Double>> taskKey = new ArrayList<Pair<String,Double>>();
		ArrayList<Future<Double>> taskFutureValue = new ArrayList<Future<Double>>();
		ExecutorService es = Executors.newFixedThreadPool(PARALLEL_THREADS);
		
		for (String name : names) {
			for (double c : C_RANGE) {
				
				taskKey.add(new Pair<String, Double>(name, c));
				taskFutureValue.add(es.submit(new Callable<Double>() {
					@Override
					public Double call() throws Exception {
						String cmd = LIBSVM_PATH+"svm-train -s 0 -t 4 -c "+c+" -v "+X_VAL_PARA_OPT_FOLDS+" "+TMP_PATH+name+".train";
						if (DEBUG) System.out.println(cmd);
						return executeSVM(cmd, TIMEOUT);
					}
				}));
			}
		}
		
		es.shutdown();
		es.awaitTermination(0, TimeUnit.NANOSECONDS);

		Pair<String, Double> bestParam = null;
		double bestAccuracy = Double.MIN_VALUE;

		for (int i=0; i<taskKey.size(); i++) {
			Pair<String, Double> p = taskKey.get(i); 
			double r;
			try {
				r = taskFutureValue.get(i).get();
			} catch (ExecutionException e) {
				throw new IllegalStateException("Execution error libSVM!");
			}
			if (r == -2) { 
				// this usually happens because c gets to high for the test set
				// we return the current best result
				System.out.println("Process not finished for name="+p.getFirst()+" C="+p.getSecond()+" due to timeout.");
			} else { 
				if (r < 0) throw new IllegalStateException("Unexpected libSVM result!");
			}
			
			if (r>bestAccuracy) {
				bestAccuracy=r;
				bestParam = p;
			}
		}
		
		if (bestParam == null) {
			// none found, everything timed out -- this indicates that the
			// configuration should be changed!
			bestParam = taskKey.get(0);
			System.out.println("\t\tWARNING: Cross-validation did not finish! Using first setting.");
		}
		System.out.println("\t\tSelected Gram: "+bestParam.getFirst());
		System.out.println("\t\tSelected C: "+bestParam.getSecond());
		System.out.println("\t\tAnticipated Accuracy: "+bestAccuracy);
		
		return bestParam;
	}
	
	
	/**
	 * Learns a model based on the given training file with the given C 
	 * parameter.
	 * @param trainFile the training file name
	 * @param c the regularization parameter
	 * @return model file name the file name containing the model information
	 * @throws IOException
	 */
	public String buildModel(String trainFile, double c) throws IOException, InterruptedException {
		if (DEBUG) {
			System.out.println("buildModel");
		}

		String modelFile = trainFile+".model";
		String cmd = LIBSVM_PATH+"svm-train -s 0 -t 4 -c "+c+" "+TMP_PATH+trainFile+" "+TMP_PATH+modelFile;
		if (DEBUG) System.out.println(cmd);
		
		double r = executeSVM(cmd, 0);
		if (r != -1) throw new IllegalStateException("Unexpected libSVM result!+\nCommand: "+cmd);
		
		
		return modelFile;
	}
	
	/**
	 * Predict the given test set based on the given model.
	 * @param modelFile model file name
	 * @param testFile test file name
	 * @return the obtained accuracy
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public double predict(String modelFile, String testFile) throws IOException, InterruptedException {
		if (DEBUG) {
			System.out.println("Predict");
		}

		String outFile = modelFile+".out";
		String cmd = LIBSVM_PATH+"svm-predict "+TMP_PATH+testFile+" "+TMP_PATH+modelFile+" "+TMP_PATH+outFile;
		if (DEBUG) System.out.println(cmd);
		
		double r = executeSVM(cmd, 0);
		if (r < 0) throw new IllegalStateException("Unexpected libSVM result!");
		
		return r;
	}
	

	/**
	 * Run the given command, returns the accuracy provided by libSVM.
	 * @param commandLine
	 * @param timeout the process is killed if not finished within the specified time
	 * @return accuracy, or -1 if call does output accuracy or -2 because of timeout, -3 if call failed
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public double executeSVM(final String commandLine, final long timeout) throws IOException, InterruptedException {
		Runtime runtime = Runtime.getRuntime();
		Process process = runtime.exec(commandLine);

		Worker worker = new Worker(process);
		worker.start();
		try {
			worker.join(timeout);
			if (worker.exit != null && worker.exit == 0) {
				return worker.accuracy;
			} else {
				if (worker.exit == null) {
					return -2; // Process not finished: Timeout
				} else {
					return -3; // Process not finished: libSVM call failed!
				}
			}
		} catch(InterruptedException ex) {
			worker.interrupt();
			Thread.currentThread().interrupt();
			throw ex;
		} finally {
			process.destroy();
		}
	}
		
	/**
	 * Worker to handle process output in a different thread.
	 * @author Nils Kriege
	 */
	final class Worker extends Thread {
		final Process process;
		Integer exit;
		double accuracy = -1;
		
		private Worker(Process process) {
			this.process = process;
		}
		
		public void run() {
			try {
				BufferedReader buff = new BufferedReader(new InputStreamReader(process.getInputStream()));
				String str;
				while ((str = buff.readLine()) != null) {
					if (DEBUG) System.out.println(str);
					if (str.contains("Accuracy = ")) {
						String accuracyString = str.split("%")[0].split("Accuracy = ")[1];
						accuracy = Double.valueOf(accuracyString);
					}
				}
				exit = process.waitFor();
			} catch (InterruptedException ignore) {
				return;
			} catch (IOException io) {
				return;
			}
		}
	}

	
}
