package edu.nyu.en.nn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.dom4j.DocumentException;

import edu.nyu.en.ace.acetypes.AceEvent;
import edu.nyu.en.ace.acetypes.AceEventMention;
import edu.nyu.en.perceptron.core.Controller;
import edu.nyu.en.perceptron.featureGenerator.TextFeatureGenerator;
import edu.nyu.en.perceptron.types.Alphabet;
import edu.nyu.en.perceptron.types.Document;
import edu.nyu.en.perceptron.types.Sentence;
import edu.nyu.en.perceptron.types.SentenceInstance;

public class DatasetPreparer {
	
	public static String coreDataPath = "/misc/proteus108/thien/projects/fifth/eventNugget/";
	//public static String coreDataPath = "/misc/proteus108/thien/projects/fourth/jointEE/";
	
	public static HashMap<String, String> subtype2type = new HashMap<String, String>();
	
	public static String ctrl = "skipNonEventSent=true crossSent=false order=0";
	
	public static ArrayList<String> ignoredList = new ArrayList<String>();
	static {
		ignoredList.add("bolt-eng-DF-170-181109-48534.txt");
		ignoredList.add("bolt-eng-DF-170-181109-47916.txt");
	}
	
	public static int[] readData(PrintWriter printer, String sourceDir, String nuggetDir, String hopperDir, 
			Alphabet nodeTypeTargetAlphabet, Alphabet nodeSubTypeTargetAlphabet, Alphabet nodeRealisTargetAlphabet,
			Controller controller, boolean singleToken, boolean skipNonEvent, boolean learnable) throws Exception, DocumentException
	{
		System.out.println("--------Reading data instance from source : " + sourceDir);
		
		TextFeatureGenerator featGen = new TextFeatureGenerator();
		
		File sourceDirFile = new File(sourceDir);
		
		File[] sourceFiles = sourceDirFile.listFiles();
		
		int count = 0, 
				totalIgnoredGroupDueToOverlappingEventMention = 0, mentionGroup = 0, mentionGroupEmployed = 0, singleEventGroup = 0,
				totalIgnoredMentionDueToOverlappingEventMention = 0, numberMentions = 0, numberMentionEmployed =0, singleTokenMentionCounter = 0;
		for (File sfile : sourceFiles) {
			if (sfile.getName().startsWith(".")) continue;
			if (sfile.getName().endsWith(Document.preprocessedFileExt)) continue;
			if (ignoredList.contains(sfile.getName())) continue;
			count++;
			System.out.println("[" + count + "]: " + sfile);
			
			Document doc = Document.createAndPreprocess(sfile, nuggetDir, hopperDir, false, false);
			
			featGen.fillTextFeatures_NoPreprocessing(doc); //, singleToken
			
			totalIgnoredGroupDueToOverlappingEventMention += doc.getAceAnnotations().ignoredCoreferenceGroupDueToOveralpingEventMentions;
			totalIgnoredMentionDueToOverlappingEventMention += doc.getAceAnnotations().ignoredEventMentionsDueToOverlapingEventMentions;
			
			printer.println("#BeginOfDocument " + doc.docID);
			
			int perDocNumberMentionEmployed = 0;
			int perDocSingleTokenMentionCounter = 0;
			
			for(int sent_id=0 ; sent_id<doc.getSentences().size(); sent_id++)
			{
				Sentence sent = doc.getSentences().get(sent_id);
				
				if (skipNonEvent && (sent.eventMentions == null || sent.eventMentions.isEmpty())) continue;
				
				SentenceInstance inst = new SentenceInstance(sent, nodeTypeTargetAlphabet, nodeSubTypeTargetAlphabet, nodeRealisTargetAlphabet, controller, singleToken, learnable);
			
				inst.conllFormatNNWriter(printer);
				
				perDocNumberMentionEmployed += inst.eventMentions.size();
				
				for (AceEventMention ment : inst.eventMentions) {
					if (ment.headIndices.size() == 1) perDocSingleTokenMentionCounter++;
				}
			}
			
			numberMentions += doc.getAceAnnotations().numEventMentionsAfterRemovingOverlapping;
			numberMentionEmployed += perDocNumberMentionEmployed;
			singleTokenMentionCounter += perDocSingleTokenMentionCounter;
			
			String coref = "";
			int countCoref = -1;
			int perDocMentionGroupEmployed = 0;
			int perDocSingleEventGroup = 0;
			
			mentionGroup += doc.getAceAnnotations().events.size();
			for (AceEvent event : doc.getAceAnnotations().events) {
				coref = "";
				for (AceEventMention mention : event.mentions) {
					// handel the removed mentions due to the span and token indexes assignment to sentences
					if (!doc.getAceAnnotations().employedEventMentionIds.contains(mention.getId())) continue;
					coref += mention.getId() + " ";
				}
				if (coref.isEmpty()) continue;
				
				perDocMentionGroupEmployed++;
				if (coref.trim().split(" ").length == 1) perDocSingleEventGroup++;
				
				countCoref++;
				coref = coref.trim().replaceAll(" ", ",");
				printer.println("@Coreference" + "\t" + "C" + countCoref + "\t" + coref);
			}
			mentionGroupEmployed += perDocMentionGroupEmployed;
			singleEventGroup += perDocSingleEventGroup;
			
			printer.println("#EndOfDocument");
			
			System.out.println(" -----Groups------");
			System.out.println("#event groups removed by the overalping: " + doc.getAceAnnotations().ignoredCoreferenceGroupDueToOveralpingEventMentions);
			System.out.println("#event groups after removing the overalping: " + doc.getAceAnnotations().events.size());
			System.out.println("#event groups after removing the overalping and span and token index assignment for sentences (employed): " + perDocMentionGroupEmployed);
			System.out.println("# single event mention group employed: " + perDocSingleEventGroup);
			System.out.println("\n -----Event Mentions------");
			System.out.println("#event mentions removed by the overalping: " + doc.getAceAnnotations().ignoredEventMentionsDueToOverlapingEventMentions);
			System.out.println("#event mentions after removing the overalping: " + doc.getAceAnnotations().numEventMentionsAfterRemovingOverlapping);
			System.out.println("#event mentions after removing the overalping and span and token index assignment for sentences (employed): " + perDocNumberMentionEmployed);
			System.out.println("single token event mentions employed: " + perDocSingleTokenMentionCounter);
			System.out.println();
		}
		
		System.out.println("\n\n Read " + count + " files!");
		System.out.println("#event groups removed by the overalping: " + totalIgnoredGroupDueToOverlappingEventMention);
		System.out.println("#event groups after removing the overalping: " + mentionGroup);
		System.out.println("#event groups after removing the overalping and span and token index assignment for sentences (employed): " + mentionGroupEmployed);
		System.out.println("# single event mention group employed: " + singleEventGroup);
		
		System.out.println("#event mentions removed by the overalping: " + totalIgnoredMentionDueToOverlappingEventMention);
		System.out.println("#event mentions after removing the overalping: " + numberMentions);
		System.out.println("#event mentions after removing the overalping and span and token index assignment for sentences (employed): " + numberMentionEmployed);
		System.out.println("single token event mentions employed: " + singleTokenMentionCounter);
		return new int[] {count, mentionGroupEmployed, numberMentionEmployed};
	}
	
	public static void createDataset(String sourceDir, String nuggetDir, String hopperDir, String outDir, 
			Controller controller, boolean singleToken, boolean skipNonEvent, boolean learnable) throws Throwable {
		Alphabet nodeTypeTargetAlphabet = new Alphabet();
		Alphabet nodeSubTypeTargetAlphabet = new Alphabet();
		Alphabet nodeRealisTargetAlphabet = new Alphabet();
		
		System.out.println("Creating dataset ...");
		
		System.out.println("\n" + controller.toString() + "\n");
		
		PrintWriter printer = new PrintWriter(new FileWriter(outDir));
		

		int[] counter = readData(printer, sourceDir, nuggetDir, hopperDir, 
				nodeTypeTargetAlphabet, nodeSubTypeTargetAlphabet, nodeRealisTargetAlphabet, 
				controller, singleToken, skipNonEvent, learnable);
		
		printer.close();
		
		System.out.println("Done! " + counter[0] + " files in total");
		System.out.println("#Event Groups (employed): " + counter[1]);
		System.out.println("#Event Mentions (employed): " + counter[2]);
		
		System.out.println("--------------Subtype to type-------------: " + subtype2type.size());
		for (String sub : subtype2type.keySet()) {
			System.out.println(sub + " --> " + subtype2type.get(sub));
		}
	}
	
	public static void main(String[] args) throws Throwable {
		
		
//		String sourceDir = "/Users/thien/workspace/fifth/EventNugget/corpus/small/source";
//		String nuggetDir = "/Users/thien/workspace/fifth/EventNugget/corpus/small/event_nugget";
//		String hopperDir = "/Users/thien/workspace/fifth/EventNugget/corpus/small/event_hopper";
//		String outDir = "/Users/thien/workspace/fifth/EventNugget/corpus/small/output.txt";
//		
		String sourceDir = args[0];
		String nuggetDir = args[1];
		String hopperDir = args[2];
		String outDir = args[3];
		
		String[] settings = ctrl.split(" ");
		Controller controller = new Controller();
		controller.setValueFromArguments(settings);
		
		boolean singleToken = Integer.parseInt(args[4]) == 0 ? false : true;
		boolean skipNonEvent = Integer.parseInt(args[5]) == 0 ? false : true;
		boolean learnable = Integer.parseInt(args[6]) == 0 ? false : true;
		
		createDataset(sourceDir, nuggetDir, hopperDir, outDir, controller, singleToken, skipNonEvent, learnable);
		
	}
}
