package edu.nyu.en.perceptron.types;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import edu.nyu.en.ace.acetypes.*;
import edu.nyu.en.perceptron.core.Controller;
import edu.nyu.en.perceptron.featureGenerator.NodeFeatureGenerator;
import edu.nyu.en.perceptron.graph.DependencyGraph;
import edu.nyu.en.perceptron.types.Sentence.Sent_Attribute;
import edu.nyu.en.util.Span;
import edu.nyu.en.util.TokenAnnotations;
import edu.nyu.en.util.TypeConstraints;
import edu.stanford.nlp.trees.Tree;

/**
 * This is a basic object of the learning algrithm
 * it represents a sentence, including text features target assignments, beam in searching etc.
 * 
 * For the text features, it should contain feature vectors for each token in the sentence
 * and the original rich representation of the sentence, e.g. dependency parse tree etc.
 * 
 * For the (target) assignment, it should encode two types of assignment:
 * (1) label assignment for each token: refers to the event trigger classification 
 * (2) assignment for any sub-structure of the sentence, e.g. one assignment indicats that 
 * the second token is argument of the first trigger
 * 
 * Given the first type of assigment, it should be able to get features for the learning algorithm, e.g. token feature vector X assignment
 * similarly, given the second type of assignment, it should be able to get features like: text features assoicated with tokens X assignment
 * Finally, on top of the assignment, it should be able to get arbitrary features, e.g. count how many "triggers" accur in this sentence
 * @author che
 *
 */
public class SentenceInstance
{
	public boolean learnable = false;
	
	// the alphabet of the label for each node (token of trigger)
	public Alphabet nodeTypeTargetAlphabet;
	public Alphabet nodeSubTypeTargetAlphabet;
	public Alphabet nodeRealisTargetAlphabet;
	
	// the settings of the whole perceptron
	public Controller controller;
	
	// the ground-truth assignment for the sentence
	public SentenceAssignment target; 
	
	// the text of the original doc
	public String allText;
	
	public String docID;
	
	/**
	 * the list of event mentions 
	 */
	public List<AceEventMention> eventMentions;
	
	/**
	 * a sequence of token, each token is a vector of features
	 * this is useful for the beam search 
	 */
	Map<InstanceAnnotations, Object> textFeaturesMap = new HashMap<InstanceAnnotations, Object>();
	
	public Sentence sentence;
	
	static public enum InstanceAnnotations
	{
		Token_FEATURE_MAPs,			// list->map<key,value> token feature maps, each map contains basic text features for a token
		DepGraph,	  				// dependency: Collection<TypedDependency> or other kind of data structure
		TOKEN_SPANS,				// List<Span>: the spans of each token in this sent
		POSTAGS,					// POS tags
		NodeTextFeatureVectors,		// node feature Vectors
		ParseTree					// parse tree
	}
	
	public String uid = "";
	
	public Object get(InstanceAnnotations key)
	{
		return textFeaturesMap.get(key);
	}
	
	public Span[] getTokenSpans()
	{
		return (Span[]) textFeaturesMap.get(InstanceAnnotations.TOKEN_SPANS);
	}
	
	public String[] getPosTags()
	{
		return (String[]) textFeaturesMap.get(InstanceAnnotations.POSTAGS);
	}
	
	public List<Map<Class<?>, Object>> getTokenFeatureMaps()
	{
		return (List<Map<Class<?>, Object>>) textFeaturesMap.get(InstanceAnnotations.Token_FEATURE_MAPs);
	}
	
	public SentenceInstance(Alphabet nodeTypeTargetAlphabet, Alphabet nodeSubTypeTargetAlphabet, Alphabet nodeRealisTargetAlphabet,
			Controller controller, boolean learnable)
	{
		this.nodeTypeTargetAlphabet = nodeTypeTargetAlphabet;
		this.nodeSubTypeTargetAlphabet = nodeSubTypeTargetAlphabet;
		this.nodeRealisTargetAlphabet = nodeRealisTargetAlphabet;
		this.controller = controller;
		this.learnable = learnable;
	}
	
	/**
	 * use sentence instance to initialize the training instance
	 * the SentenceInstance object can also be initialized by a file
	 * @param sent
	 */
	public SentenceInstance(Sentence sent, Alphabet nodeTypeTargetAlphabet, Alphabet nodeSubTypeTargetAlphabet, Alphabet nodeRealisTargetAlphabet, 
			Controller controller, boolean singleToken, boolean learnable)
	{
		this(nodeTypeTargetAlphabet, nodeSubTypeTargetAlphabet, nodeRealisTargetAlphabet, controller, learnable);
		
		this.sentence = sent;
		
		// set the text of the doc
		this.allText = sent.doc.text;
		this.docID = sent.doc.docID;
		
		this.uid = this.docID + "#" + sent.sentID;
		
		// fill in token text feature maps
		this.textFeaturesMap.put(InstanceAnnotations.Token_FEATURE_MAPs, sent.get(Sent_Attribute.Token_FEATURE_MAPs));
		
		// fill in Annotations map with dependency paths, later we can even fill in parse tree etc.
		DependencyGraph graph = (DependencyGraph) sent.get(Sent_Attribute.DepGraph);
		this.textFeaturesMap.put(InstanceAnnotations.DepGraph, graph);
		
		// fill in parse tree
		this.textFeaturesMap.put(InstanceAnnotations.ParseTree, sent.get(Sent_Attribute.ParseTree));
		
		// fill in tokens and pos tags
		this.textFeaturesMap.put(InstanceAnnotations.TOKEN_SPANS, sent.get(Sent_Attribute.TOKEN_SPANS));
		this.textFeaturesMap.put(InstanceAnnotations.POSTAGS, sent.get(Sent_Attribute.POSTAGS));
		
		// get node text feature vectors
		List<List<String>> tokenFeatVectors = NodeFeatureGenerator.get_node_text_features(this);
		this.textFeaturesMap.put(InstanceAnnotations.NodeTextFeatureVectors, tokenFeatVectors);
		
		// add event ground-truth
		eventMentions = new ArrayList<AceEventMention>();
		eventMentions.addAll(sent.eventMentions);
		
		// add target as gold-standard assignment
		this.target = new SentenceAssignment(this, singleToken);
	}
	
	public void conllFormatNNWriter(PrintWriter printer) {

		for (int i = 0; i < size(); i++) {
			printer.println(conllFormatLine(i));
		}
		
		printer.println();
	}
	
	private String conllFormatLine(int i) {
		String line = "";
		
		List<Map<Class<?>, Object>> sent = (List<Map<Class<?>, Object>>) this.get(InstanceAnnotations.Token_FEATURE_MAPs);
		Map<Class<?>, Object> token = sent.get(i);
		
		line += i + "\t";
		
		Span span = (Span) token.get(TokenAnnotations.SpanAnnotation.class);
		Integer start = this.sentence.doc.plain2xml.get(span.start());
		Integer end = this.sentence.doc.plain2xml.get(span.end());
		line += start + "\t" + end + "\t";
		
		String word = (String) token.get(TokenAnnotations.TextAnnotation.class);
		line += word + "\t";
		
		String lemma = (String) token.get(TokenAnnotations.LemmaAnnotation.class);
		line += lemma + "\t";
		
		String pos = (String) token.get(TokenAnnotations.PartOfSpeechAnnotation.class);
		line += pos + "\t";
		
		String chunk = (String) token.get(TokenAnnotations.ChunkingAnnotation.class);
		line += chunk + "\t";
		
		String base = (String) token.get(TokenAnnotations.NomlexbaseAnnotation.class);
		if (base == null) base = "NONE";
		line += base + "\t";
		
		Integer clause = (Integer) token.get(TokenAnnotations.ClauseAnnotation.class);
		line += clause + "\t";
		
		List<String> possibleTypes = NodeFeatureGenerator.getPossibleEventTypes(lemma);
		line += concat(possibleTypes) + "\t";
		
		List<String> synonyms = (List<String>) token.get(TokenAnnotations.SynonymsAnnotation.class);
		line += concat(synonyms) + "\t";
		
		List<String> brownClusters = (List<String>) token.get(TokenAnnotations.BrownClusterAnnotation.class);
		line += concat(brownClusters) + "\t";
		
		Vector<String> dep_features = (Vector<String>) token.get(TokenAnnotations.DependencyAnnotation4NN.class);
		line += concat(dep_features) + "\t";
		
//		List<String> entityInfo = (List<String>) token.get(TokenAnnotations.EntityAnnotation4NN.class);
//		line += concat(entityInfo) + "\t";
		
		boolean nonref = false;
		if(word.equalsIgnoreCase("it"))
		{
			Tree tree = (Tree) this.get(InstanceAnnotations.ParseTree);
			nonref = NodeFeatureGenerator.isNonRefPronoun(tree, i);
		}
		line += nonref + "\t";
		
		boolean titleModifier = NodeFeatureGenerator.checkNPModifier(sent, i);
		line += titleModifier + "\t";
		
		int eligible = TypeConstraints.isPossibleTriggerByPOS(this, i) ? 1 : 0;
		line += eligible + "\t";
		
		List<String> nodeFets = NodeFeatureGenerator.get_node_text_features(this, i);
		line += concat(nodeFets) + "\t";
		
		//Annotation
		int type_index = this.target.nodeTypeAssignment.get(i);
		String type_ann = (String) this.nodeTypeTargetAlphabet.lookupObject(type_index);
		
		int subtype_index = this.target.nodeSubTypeAssignment.get(i);
		String subtype_ann = (String) this.nodeSubTypeTargetAlphabet.lookupObject(subtype_index);
		
		int realis_index = this.target.nodeRealisAssignment.get(i);
		String realis_ann = (String) this.nodeRealisTargetAlphabet.lookupObject(realis_index);
		
		String mentId = this.target.nodeIdAssignment.get(i);
		
		line += type_ann + "\t" + subtype_ann + "\t" + realis_ann + "\t" + mentId + "\t";
		
		return line.trim();
	}
	
	private String concat(List<String> list) {
		if (list == null) list = new ArrayList<String>();
		
		if (list.isEmpty()) return "NONE";
		String res = "";
		for (String l : list)
			res += l + " ";
		return res.trim();
	}

	/**
	 * the size of the sentence
	 * @return
	 */
	public int size()
	{
		return this.getTokenSpans().length;
	}
}
