package opt.test;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MaxKColorFitnessFunction;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.ga.Vertex;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;


/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class GJContinousPeak {
	
	private static DecimalFormat df = new DecimalFormat("0.000");

	/** The n value **/
    private static final int N = 60;
    /** The t value */
    private static final int T = N / 10;
    /**
     * The test main
     * @param args ignored
     */
    
    public static void main(String[] args) throws IOException {
        
    	int iterations[] = {50, 100, 200, 300, 500, 1000, 2000, 3000, 5000};
    	double start, end;
    	int tot_run = 5;
    	String[] list_rhc_time = new String[iterations.length];
    	String[] list_sa_time = new String[iterations.length];
    	String[] list_ga_time = new String[iterations.length];
    	String[] list_mimic_time = new String[iterations.length];
    	
    	String[] list_rhc_eval = new String[iterations.length];
    	String[] list_sa_eval = new String[iterations.length];
    	String[] list_ga_eval = new String[iterations.length];
    	String[] list_mimic_eval = new String[iterations.length];
    	
    	for (int i=0; i<iterations.length;i++) {
    		list_rhc_time[i] =  "0";
    		list_sa_time[i] = "0";
    		list_ga_time[i] = "0";
    		list_mimic_time[i] = "0";
    		list_rhc_eval[i] = "0";
    		list_sa_eval[i] = "0";
    		list_ga_eval[i] = "0";
    		list_mimic_eval[i] = "0";
    	} 
    	
    	
    	int iter =0;
    	for(int iters: iterations)
    		
    	{
    		
    		System.out.println("Current iteration" + iters);
    		double tot_eval_rhc = 0;
    		double tot_eval_sa = 0;
    		double tot_eval_ga = 0;
    		double tot_eval_mimic = 0;
    		
    		double time_rhc = 0;
    		double time_sa = 0;
    		double time_ga = 0;
    		double time_mimic = 0;
    		double time = 0;
    		for (int currun= 0; currun < tot_run; currun++)
    		{
    			
    			System.out.println("Current: run" + currun);
    			
    			int[] ranges = new int[N];
    	        Arrays.fill(ranges, 2);
    	        EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(T);
    	        Distribution odd = new DiscreteUniformDistribution(ranges);
    	        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
    	        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
    	        CrossoverFunction cf = new SingleCrossOver();
    	        Distribution df = new DiscreteDependencyTree(.1, ranges); 
    	        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
    	        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
    	        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);		        
    	        
		        //Randomized hill climbing
		        start = System.nanoTime();
		        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
		        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iters);
		        fit.train();
		        end = System.nanoTime();
		        time = (end - start)/Math.pow(10,9);
		        time_rhc = time_rhc +  time;
		        tot_eval_rhc = tot_eval_rhc + ef.value(rhc.getOptimal());
		        		        
		        
		        //Simulated Annealing
		        start = System.nanoTime();
		        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
		        fit = new FixedIterationTrainer(sa, iters);
		        fit.train();
		        end = System.nanoTime();
		        time = (end - start)/Math.pow(10,9);
		        time_sa = time_sa +  time;
		        tot_eval_sa = tot_eval_sa + ef.value(sa.getOptimal());
		        
		        
		        //Genetic Algorithm
		        start = System.nanoTime();
		        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
		        fit = new FixedIterationTrainer(ga, iters);
		        fit.train();
		        end = System.nanoTime();
		        time = (end - start)/Math.pow(10,9);
		        time_ga = time_ga +  time;
		        tot_eval_ga = tot_eval_ga + ef.value(ga.getOptimal());
		        
		        
		        //Mimic
		        start = System.nanoTime();
		        MIMIC mimic = new MIMIC(200, 20, pop);
		        fit = new FixedIterationTrainer(mimic, iters);
		        fit.train();
		        end = System.nanoTime();
		        time = (end - start)/Math.pow(10, 9);
		        time_mimic = time_mimic +  time;
		        tot_eval_mimic = tot_eval_mimic + ef.value(mimic.getOptimal());
		        
		        System.out.println("Current run: " + currun + " rhc: " + ef.value(rhc.getOptimal()) + " sa: " +ef.value(sa.getOptimal()) + " ga: " + ef.value(ga.getOptimal())+ " mimic: " +ef.value(mimic.getOptimal()));
	        
    		}
    		
    		//Averaging the total optimal fitness values and total time for test runs
    		tot_eval_rhc /= tot_run;
    		tot_eval_sa /= tot_run;
    		tot_eval_ga /= tot_run;
    		tot_eval_mimic /= tot_run;
    		
    		time_rhc /= tot_run;
    		time_sa /= tot_run;
    		time_ga /= tot_run;
    		time_mimic /= tot_run;
    		
    		//Updating the values to array
    		list_rhc_time[iter] = df.format(time_rhc);
    		list_sa_time[iter] = df.format(time_sa);
    		list_ga_time[iter] = df.format(time_ga);
    		list_mimic_time[iter] = df.format(time_mimic);
    		
    		list_rhc_eval[iter] = df.format(tot_eval_rhc);
    		list_sa_eval[iter] = df.format(tot_eval_sa);
    		list_ga_eval[iter] = df.format(tot_eval_ga);
    		list_mimic_eval[iter] = df.format(tot_eval_mimic);
    		
    		iter+=1;
    }
    	
    	//Writing strings to files
    	
int L = list_rhc_eval.length;
    	
    	FileWriter writer_rhc = new FileWriter("ContinousePeak_rhc.csv");
    	for (int j=0; j<L; j++) {
    		writer_rhc.append(list_rhc_eval[j]);
    		writer_rhc.append("\n");		
    		
    	}
    	writer_rhc.close();
    	
    	FileWriter writer_sa = new FileWriter("ContinousePeak_sa.csv");
    	for (int j=0; j<L; j++) {
    		writer_sa.append(list_sa_eval[j]);
    		writer_sa.append("\n");		
    		
    	}
    	writer_sa.close();
    	
    	FileWriter writer_ga = new FileWriter("ContinousePeak_ga.csv");
    	for (int j=0; j<L; j++) {
    		writer_ga.append(list_ga_eval[j]);
    		writer_ga.append("\n");		
    		
    	}
    	writer_ga.close();
    	
    	FileWriter writer_mimic = new FileWriter("ContinousePeak_mimic.csv");
    	for (int j=0; j<L; j++) {
    		writer_mimic.append(list_mimic_eval[j]);
    		writer_mimic.append("\n");		
    		
    	}
    	writer_mimic.close();
    	
    	FileWriter writer_rhc_time = new FileWriter("ContinousePeak_rhc_time.csv");
    	for (int j=0; j<L; j++) {
    		writer_rhc_time.append(list_rhc_time[j]);
    		writer_rhc_time.append("\n");		
    		
    	}
    	writer_rhc_time.close();
    	
    	FileWriter writer_sa_time = new FileWriter("ContinousePeak_sa_time.csv");
    	for (int j=0; j<L; j++) {
    		writer_sa_time.append(list_sa_time[j]);
    		writer_sa_time.append("\n");		
    		
    	}
    	writer_sa_time.close();
    	
    	FileWriter writer_ga_time = new FileWriter("ContinousePeak_ga_time.csv");
    	for (int j=0; j<L; j++) {
    		writer_ga_time.append(list_ga_time[j]);
    		writer_ga_time.append("\n");		
    		
    	}
    	writer_ga_time.close();
    	
    	FileWriter writer_mimic_time = new FileWriter("ContinousePeak_mimic_time.csv");
    	for (int j=0; j<L; j++) {
    		writer_mimic_time.append(list_mimic_time[j]);
    		writer_mimic_time.append("\n");		
    		
    	}
    	writer_mimic_time.close();
    	
    	///////////////////////////////
    	
    	int sample_N[] = {10,20,30,40,50,60,100,200,300,400,500};
    	double start_n, end_n;
    //	int tot_run = 5;
    	String[] list_rhc_time_n = new String[sample_N.length];
    	String[] list_sa_time_n = new String[sample_N.length];
    	String[] list_ga_time_n = new String[sample_N.length];
    	String[] list_mimic_time_n = new String[sample_N.length];
    	
    	String[] list_rhc_eval_n = new String[sample_N.length];
    	String[] list_sa_eval_n = new String[sample_N.length];
    	String[] list_ga_eval_n = new String[sample_N.length];
    	String[] list_mimic_eval_n = new String[sample_N.length];
    	
    	for (int i=0; i<sample_N.length;i++) {
    		list_rhc_time_n[i] =  "0";
    		list_sa_time_n[i] = "0";
    		list_ga_time_n[i] = "0";
    		list_mimic_time_n[i] = "0";
    		list_rhc_eval_n[i] = "0";
    		list_sa_eval_n[i] = "0";
    		list_ga_eval_n[i] = "0";
    		list_mimic_eval_n[i] = "0";
    	} 
    	
    	int ind =0;
    	for(int n: sample_N)
    		
    	{
    		int TT = n / 10;	
    		System.out.println("Current sample: " + n);
    		double tot_eval_rhc_n = 0;
    		double tot_eval_sa_n = 0;
    		double tot_eval_ga_n = 0;
    		double tot_eval_mimic_n = 0;
    		
    		double time_rhc_n = 0;
    		double time_sa_n = 0;
    		double time_ga_n = 0;
    		double time_mimic_n = 0;
    		double time_n = 0;
    		for (int currun= 0; currun < tot_run; currun++)
    		{
    			
    			
    			
    			int[] ranges = new int[n];
    	        Arrays.fill(ranges, 2);
    	        EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(TT);
    	        Distribution odd = new DiscreteUniformDistribution(ranges);
    	        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
    	        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
    	        CrossoverFunction cf = new SingleCrossOver();
    	        Distribution df = new DiscreteDependencyTree(.1, ranges); 
    	        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
    	        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
    	        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);		        
    	        
		        //Randomized hill climbing
		        start_n = System.nanoTime();
		        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
		        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 3000);
		        fit.train();
		        end_n = System.nanoTime();
		        time_n = (end_n - start_n)/Math.pow(10,9);
		        time_rhc_n = time_rhc_n +  time_n;
		        tot_eval_rhc_n = tot_eval_rhc_n + ef.value(rhc.getOptimal());
		        		        
		        
		        //Simulated Annealing
		        start_n = System.nanoTime();
		        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
		        fit = new FixedIterationTrainer(sa, 3000);
		        fit.train();
		        end_n = System.nanoTime();
		        time_n = (end_n - start_n)/Math.pow(10,9);
		        time_sa_n = time_sa_n +  time_n;
		        tot_eval_sa_n = tot_eval_sa_n + ef.value(sa.getOptimal());
		        
		        
		        //Genetic Algorithm
		        start_n = System.nanoTime();
		        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
		        fit = new FixedIterationTrainer(ga, 3000);
		        fit.train();
		        end_n = System.nanoTime();
		        time_n = (end_n - start_n)/Math.pow(10,9);
		        time_ga_n = time_ga_n +  time_n;
		        tot_eval_ga_n = tot_eval_ga_n + ef.value(ga.getOptimal());
		        
		        
		        //Mimic
		        start = System.nanoTime();
		        MIMIC mimic = new MIMIC(200, 20, pop);
		        fit = new FixedIterationTrainer(mimic, 3000);
		        fit.train();
		        end_n = System.nanoTime();
		        time_n = (end_n - start_n)/Math.pow(10, 9);
		        time_mimic_n = time_mimic_n +  time_n;
		        tot_eval_mimic_n = tot_eval_mimic_n + ef.value(mimic.getOptimal());
		        
		        System.out.println("Current run: " + currun + " rhc: " + ef.value(rhc.getOptimal()) + " sa: " +ef.value(sa.getOptimal()) + " ga: " + ef.value(ga.getOptimal())+ " mimic: " +ef.value(mimic.getOptimal()));
	        
    		}
    		
    		//Averaging the total optimal fitness values and total time for test runs
    		tot_eval_rhc_n /= tot_run;
    		tot_eval_sa_n /= tot_run;
    		tot_eval_ga_n /= tot_run;
    		tot_eval_mimic_n /= tot_run;
    		
    		time_rhc_n /= tot_run;
    		time_sa_n /= tot_run;
    		time_ga_n /= tot_run;
    		time_mimic_n /= tot_run;
    		
    		//Updating the values to array
    		list_rhc_time_n[ind] = df.format(time_rhc_n);
    		list_sa_time_n[ind] = df.format(time_sa_n);
    		list_ga_time_n[ind] = df.format(time_ga_n);
    		list_mimic_time_n[ind] = df.format(time_mimic_n);
    		
    		list_rhc_eval_n[ind] = df.format(tot_eval_rhc_n);
    		list_sa_eval_n[ind] = df.format(tot_eval_sa_n);
    		list_ga_eval_n[ind] = df.format(tot_eval_ga_n);
    		list_mimic_eval_n[ind] = df.format(tot_eval_mimic_n);
    		
    		ind+=1;
    }
    	
    	//Writing strings to files
    	
int LL = list_rhc_eval_n.length;
    	
    	FileWriter writer_rhc_n = new FileWriter("ContinousePeak_rhc_n.csv");
    	for (int j=0; j<LL; j++) {
    		writer_rhc_n.append(list_rhc_eval_n[j]);
    		writer_rhc_n.append("\n");		
    		
    	}
    	writer_rhc_n.close();
    	
    	FileWriter writer_sa_n = new FileWriter("ContinousePeak_sa_n.csv");
    	for (int j=0; j<LL; j++) {
    		writer_sa_n.append(list_sa_eval_n[j]);
    		writer_sa_n.append("\n");		
    		
    	}
    	writer_sa_n.close();
    	
    	FileWriter writer_ga_n = new FileWriter("ContinousePeak_ga_n.csv");
    	for (int j=0; j<LL; j++) {
    		writer_ga_n.append(list_ga_eval_n[j]);
    		writer_ga_n.append("\n");		
    		
    	}
    	writer_ga_n.close();
    	
    	FileWriter writer_mimic_n = new FileWriter("ContinousePeak_mimic_n.csv");
    	for (int j=0; j<LL; j++) {
    		writer_mimic_n.append(list_mimic_eval_n[j]);
    		writer_mimic_n.append("\n");		
    		
    	}
    	writer_mimic_n.close();
    	
    	FileWriter writer_rhc_time_n = new FileWriter("ContinousePeak_rhc_time_n.csv");
    	for (int j=0; j<LL; j++) {
    		writer_rhc_time_n.append(list_rhc_time_n[j]);
    		writer_rhc_time_n.append("\n");		
    		
    	}
    	writer_rhc_time_n.close();
    	
    	FileWriter writer_sa_time_n = new FileWriter("ContinousePeak_sa_time_n.csv");
    	for (int j=0; j<LL; j++) {
    		writer_sa_time_n.append(list_sa_time_n[j]);
    		writer_sa_time_n.append("\n");		
    		
    	}
    	writer_sa_time_n.close();
    	
    	FileWriter writer_ga_time_n = new FileWriter("ContinousePeak_ga_time_n.csv");
    	for (int j=0; j<LL; j++) {
    		writer_ga_time_n.append(list_ga_time_n[j]);
    		writer_ga_time_n.append("\n");		
    		
    	}
    	writer_ga_time_n.close();
    	
    	FileWriter writer_mimic_time_n = new FileWriter("ContinousePeak_mimic_time_n.csv");
    	for (int j=0; j<LL; j++) {
    		writer_mimic_time_n.append(list_mimic_time_n[j]);
    		writer_mimic_time_n.append("\n");		
    		
    	}
    	writer_mimic_time_n.close();
    	
    	
    	
    }
}