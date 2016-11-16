package edu.nyu.en.perceptron.featureGenerator;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import edu.nyu.en.nn.DatasetPreparer;
import edu.nyu.en.perceptron.types.SentenceInstance;
import edu.nyu.en.perceptron.types.SentenceInstance.InstanceAnnotations;
import edu.nyu.en.util.TokenAnnotations;
import edu.stanford.nlp.trees.Tree;

/**
 * this class is to implement features based on data structure of Document
 * this class may need read some resources, like dictionaries
 * this is for local feature generator
 * 
 *
 */
public class NodeFeatureGenerator 
{
	// the delimiter of token features for feature table
	static public final String Feature_Delimiter = " ";

	// Event subtype --> trigger token with high confidence value
	public static Map<String, List<String>> triggerTokensHighQuality = new HashMap<String, List<String>>();
	static
	{
		// initialize priorityQueueEntities
		try
		{
			// initialize dict of triggerTokens
			BufferedReader reader = new BufferedReader(new FileReader(DatasetPreparer.coreDataPath + "data/triggerTokens"));
			String line = "";
			while((line = reader.readLine()) != null)
			{
				if(line.length() == 0)
				{
					continue;
				}
				String[] fields = line.split("\\t");
				String eventSubType = fields[0];
				String triggerToken = fields[1];
				Double confidence = Double.parseDouble(fields[2]);
				
				if(confidence >= 0.80)
				{
					List<String> triggers = triggerTokensHighQuality.get(eventSubType);
					if(triggers == null)
					{
						triggers = new ArrayList<String>();
						triggerTokensHighQuality.put(eventSubType, triggers);
					}
					if(!triggers.contains(triggerToken))
					{
						triggers.add(triggerToken);
					}
				}
			}
			reader.close();
		} 
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}
	
	public static List<String> getPossibleEventTypes(String lemma)
	{
		List<String> ret = new ArrayList<String>();
		for(String key : triggerTokensHighQuality.keySet())
		{
			List<String> tokens = triggerTokensHighQuality.get(key);
			if(tokens.contains(lemma))
			{
				ret.add(key);
				continue;
			}
		}
		return ret;
	}
	
	public NodeFeatureGenerator() 
	{
		;
	}

	/**
	 * check if the pattern like "retired Chairman Steve"
	 * @param sent
	 * @param i
	 * @return
	 */
	public static boolean checkNPModifier(List<Map<Class<?>, Object>> sent, int i)
	{
		final String stopwords = "a|'s";
		Map<Class<?>, Object> token = sent.get(i);
		String chunkTag = (String) token.get(TokenAnnotations.ChunkingAnnotation.class);
		String text = (String) token.get(TokenAnnotations.TextAnnotation.class);
		
		if(chunkTag.equals("B-NP") && !text.matches(stopwords))
		{
			boolean hasTitle = false;
			for(int j=i+1; j<sent.size(); j++)
			{
				Map<Class<?>, Object> token2 = sent.get(j);
				String chunkTag2 = (String) token2.get(TokenAnnotations.ChunkingAnnotation.class);
				if(!chunkTag2.equals("I-NP"))
				{
					break;
				}
				else
				{
					List<String> entityInfo = (List<String>) token2.get(TokenAnnotations.EntityAnnotation.class);
					if(entityInfo != null && (entityInfo.contains("Job-Title") || entityInfo.contains("Title")))
					{
						hasTitle = true;
					}
				}
			}
			if(hasTitle)
			{
				return true;
			}
		}
		return false;
	}
	
	/**
	 * get a conjunction feature of i-th time step in sentence sent
	 * given Class<NECoreAnnotation>[] for features and int[] for time steps. 
	 * E.g. given a sent, an index i of current time step, featureName, and two arrays [Text, Text, Text], [-1,0,1]
	 * this method should return a conjunction of text annotation of -1, current, +1 time steps 
	 * @param sent
	 * @param normalize use entity information to normalize
	 * @return
	 */
	protected static String getConjuctionFeature(List<Map<Class<?>, Object>> sent, int i, String featureName, Class<?>[] features, int[] positions,
			boolean normalize)
	{
		final String delimiter = "#";
		
		if(i >= sent.size() || i<0)
		{
			return null; 
		}
		String ret = featureName + "=";
		for(int j=0; j<features.length && j<positions.length; j++)
		{
			Class<?> featureType = features[j];
			int position = (positions[j] + i);
			String partialFeature = "";
			
			// hit margins of a sentence, then pick up Padding to feed the feature
			if(position >= sent.size() || position < 0)
			{
				return null;
			}
			else
			{
				Object temp = sent.get(position).get(featureType);
				if(temp == null)
				{
					return null;
				}
				else
				{
					partialFeature = sent.get(position).get(featureType).toString();
					// use entity information to normalize Text or Lemma
					if(normalize && 
							(featureType.equals(TokenAnnotations.LemmaAnnotation.class) || featureType.equals(TokenAnnotations.TextAnnotation.class)))
					{
						List<String> entityInfo = (List<String>) sent.get(position).get(TokenAnnotations.EntityAnnotation.class);
						if(entityInfo != null && !entityInfo.isEmpty())
						{
							String entityType = entityInfo.get(0);
							partialFeature = entityType;
						}
					}
				}
			}
			ret += partialFeature;
			
			if(j<features.length-1)
			{
				ret += delimiter;
			}
		}
		return ret;	
	}
	
	/**
	 * get text feature vector for the whole sentence
	 * @param sent
	 * @return
	 */
	public static List<List<String>> get_node_text_features(SentenceInstance sent)
	{
		List<List<String>> ret = new ArrayList<List<String>>();
		for(int i=0; i<sent.size(); i++)
		{
			List<String> vector = get_node_text_features(sent, i);
			ret.add(vector);
		}
		return ret;
	}
	
	/**
	 * Assume that text features for each token has been provided in feature map List<Map<Class<?>, Object>> sent, 
	 * this function gathers all features for a token in a line, just before printing it out
	 * @param sent
	 * @param i
	 * @param token
	 * @return
	 */
	public static List<String> get_node_text_features(SentenceInstance inst, int i)
	{
		List<Map<Class<?>, Object>> sent = (List<Map<Class<?>, Object>>) inst.get(InstanceAnnotations.Token_FEATURE_MAPs);
		Map<Class<?>, Object> token = sent.get(i);
		List<String> featureLine = new ArrayList<String>();
		String feature = "";
		
		// text feature
		String word = (String) token.get(TokenAnnotations.TextAnnotation.class);
		feature = "W=" + word;
		featureLine.add(feature);
		
		// lemma feature
		String lemma = (String) token.get(TokenAnnotations.LemmaAnnotation.class);
		feature = "Lem=" + lemma;
		featureLine.add(feature);
		
		// Nomlex base from Noun --> verb. e.g. retirement --> retire
		String base = (String) token.get(TokenAnnotations.NomlexbaseAnnotation.class);
		if(base != null)
		{
			feature = "Lem=" + base;
			if(!featureLine.contains(feature))
			{
				featureLine.add(feature);
			}
		}
		
		List<String> possibleTypes = getPossibleEventTypes(lemma);
		for(String possible : possibleTypes)
		{
			feature = "Possible=" + possible;
			featureLine.add(feature);
		}
		
		// POS feature
		String pos = (String) token.get(TokenAnnotations.PartOfSpeechAnnotation.class);
		feature = "POS=" + pos;
		featureLine.add(feature);
		
//		String chunking = (String) token.get(TokenAnnotations.ChunkingAnnotation.class);
//		feature = "chunk=" + chunking;
//		featureLine.add(feature);
		
		// get WordNet Synonyms
		List<String> synonyms = (List<String>) token.get(TokenAnnotations.SynonymsAnnotation.class);
		if(synonyms != null)
		{
			for(String syn : synonyms)
			{
				feature = "Synonym=" + syn;
				featureLine.add(feature);
			}
		}
		
		// get Brown clusters 
		List<String> brownClusters = (List<String>) token.get(TokenAnnotations.BrownClusterAnnotation.class);
		if(brownClusters != null)
		{
			for(String clusterPrefix : brownClusters)
			feature = "Brown=" + clusterPrefix;
			featureLine.add(feature);
		}
		
		// get dependency features
		Vector<String> dep_features = (Vector<String>) token.get(TokenAnnotations.DependencyAnnotation.class);
		if(dep_features != null)
		{
			for(String dep_feature : dep_features)
			{
				featureLine.add(dep_feature);
			}
		}
		
		// if the current token is "it", then check it this "it"
		// is a non-referential pronun
		if(word.equalsIgnoreCase("it"))
		{
			Tree tree = (Tree) inst.get(InstanceAnnotations.ParseTree);
			boolean nonref = isNonRefPronoun(tree, i);
			feature = "NonRefProIt=" + nonref;
			featureLine.add(feature);
		}
		
		boolean titleModifier = checkNPModifier(sent, i);
		if(titleModifier)
		{
			feature = "TitleModifier=" + true;
			featureLine.add(feature);
		}
		
		// conjunction features
		addConjuctionFeatures(sent, i, featureLine);
		
		return featureLine;
	}

	/**
	 * if the current token is "it", then check it this "it" is a non-referential pronun
	 * method: if there is "s" node under "it", then it's nonref
	 * (NP (PRP it))
                  (VP (VBD was)
                    (ADJP (JJ impossible)
                      (S
                        (VP (TO to)
                          (VP (VB avert)
                            (NP (DT this) (NN war))))))))))))))
	 * @param tree
	 * @param i
	 * @return
	 */
	public static boolean isNonRefPronoun(Tree tree, int i)
	{
		List<Tree> leaves = tree.getLeaves();
		Tree leaf = leaves.get(i);
		Tree grandfather = leaf.ancestor(2, tree);
		Tree ancestor = leaf.ancestor(3, tree);
		Tree[] children = ancestor.children();
		boolean flag_after = false;
		for(Tree child : children)
		{
			if(child == grandfather)
			{
				flag_after = true;
				continue;
			}
			if(flag_after)
			{
				// check if this child has a "s" decendent
				List<Tree> subTrees = child.getChildrenAsList();
				List<Tree> queue = new ArrayList<Tree>();
				queue.addAll(subTrees);
				while(queue.size() > 0)
				{
					Tree subtree = queue.remove(0);
					if(!subtree.isPreTerminal() && 
							!subtree.isLeaf() && subtree.value().equals("S"))
					{
						return true;
					}
					else if(!subtree.isPreTerminal() && !subtree.value().matches("SBAR|SBARQ|SINV|SQ"))
					{
						queue.addAll(subtree.getChildrenAsList());
					}
				}
			}
		}
		return false;
	}

	protected static Vector<String> getConjuctionFeatureSingleVector(List<Map<Class<?>, Object>> sent, int i, Class<?> featureType, String featureName, int position)
	{
		Vector<String> ret = new Vector<String>();
		
		if(i>=sent.size() || i<0)
		{
			return null; 
		}
		position = (position + i);
		
		// hit margins of a sentence, then pick up Padding to feed the feature
		if(position >= sent.size())
		{
			return ret;
		}
		else if(position < 0)
		{
			return ret;
		}
		else
		{
			Vector<String> temp = (Vector<String>) sent.get(position).get(featureType);
			if(temp == null)
			{
				return ret;
			}
			else
			{
				for(String value : temp)
				{
					ret.add(featureName + "=" + value);
				}
			}
			
		}	
		return ret;	
	}
	
	/**
	 * add conjunction features for feature line that to be printed
	 * @param sent
	 * @param i
	 * @param featureLine
	 */
	private static void addConjuctionFeatures(List<Map<Class<?>, Object>> sent, int i, List<String> featureLine) 
	{
		// conjunction features about text (window 4, left+right, bigram, unigram
		///////////////////////////////////////
		Class<?>[] features = new Class<?>[]{TokenAnnotations.TextAnnotation.class, TokenAnnotations.TextAnnotation.class};
		int[] positions = new int[]{-1,0};
		String conjunction_feature = getConjuctionFeature(sent, i, "W-1W0", features, positions, false);
		if(conjunction_feature != null) featureLine.add(conjunction_feature);
		
		String normalized_conjunction_feature = getConjuctionFeature(sent, i, "W-1W0", features, positions, true);
		if(normalized_conjunction_feature != null && !normalized_conjunction_feature.equals(conjunction_feature)) 
			featureLine.add(normalized_conjunction_feature);
		
		features = new Class<?>[]{TokenAnnotations.TextAnnotation.class, TokenAnnotations.TextAnnotation.class};
		positions = new int[]{0,1};
		conjunction_feature = getConjuctionFeature(sent, i, "W0W1", features, positions, false);
		if(conjunction_feature != null) featureLine.add(conjunction_feature);
		
		normalized_conjunction_feature = getConjuctionFeature(sent, i, "W0W1", features, positions, true);
		if(normalized_conjunction_feature != null && !normalized_conjunction_feature.equals(conjunction_feature)) 
			featureLine.add(normalized_conjunction_feature);
		
		features = new Class<?>[]{TokenAnnotations.TextAnnotation.class};
		positions = new int[]{-1};
		conjunction_feature = getConjuctionFeature(sent, i, "W-1", features, positions, false);
		if(conjunction_feature != null) featureLine.add(conjunction_feature);
		
		normalized_conjunction_feature = getConjuctionFeature(sent, i, "W-1", features, positions, true);
		if(normalized_conjunction_feature != null && !normalized_conjunction_feature.equals(conjunction_feature)) 
			featureLine.add(normalized_conjunction_feature);
		
		features = new Class<?>[]{TokenAnnotations.TextAnnotation.class};
		positions = new int[]{1};
		conjunction_feature = getConjuctionFeature(sent, i, "W1", features, positions, false);
		if(conjunction_feature != null) featureLine.add(conjunction_feature);
		
		normalized_conjunction_feature = getConjuctionFeature(sent, i, "W1", features, positions, true);
		if(normalized_conjunction_feature != null && !normalized_conjunction_feature.equals(conjunction_feature)) 
			featureLine.add(normalized_conjunction_feature);
		
		features = new Class<?>[]{TokenAnnotations.TextAnnotation.class};
		positions = new int[]{-2};
		conjunction_feature = getConjuctionFeature(sent, i, "W-2", features, positions, false);
		if(conjunction_feature != null) featureLine.add(conjunction_feature);
		
		normalized_conjunction_feature = getConjuctionFeature(sent, i, "W-2", features, positions, true);
		if(normalized_conjunction_feature != null && !normalized_conjunction_feature.equals(conjunction_feature)) 
			featureLine.add(normalized_conjunction_feature);
		
		features = new Class<?>[]{TokenAnnotations.TextAnnotation.class};
		positions = new int[]{2};
		conjunction_feature = getConjuctionFeature(sent, i, "W2", features, positions, false);
		if(conjunction_feature != null) featureLine.add(conjunction_feature);

		normalized_conjunction_feature = getConjuctionFeature(sent, i, "W2", features, positions, true);
		if(normalized_conjunction_feature != null && !normalized_conjunction_feature.equals(conjunction_feature)) 
			featureLine.add(normalized_conjunction_feature);
		
		/////////////////////////////
		// conjunction feature about POS
		//////////////////////////////
		features = new Class<?>[]{TokenAnnotations.PartOfSpeechAnnotation.class};
		positions = new int[]{-1};
		conjunction_feature = getConjuctionFeature(sent, i, "POS-1", features, positions, false);
		if(conjunction_feature != null) featureLine.add(conjunction_feature);
		
		features = new Class<?>[]{TokenAnnotations.PartOfSpeechAnnotation.class};
		positions = new int[]{1};
		conjunction_feature = getConjuctionFeature(sent, i, "POS1", features, positions, false);
		if(conjunction_feature != null) featureLine.add(conjunction_feature);
			
		features = new Class<?>[]{TokenAnnotations.PartOfSpeechAnnotation.class, TokenAnnotations.PartOfSpeechAnnotation.class};
		positions = new int[]{-1,0};
		conjunction_feature = getConjuctionFeature(sent, i, "POS-1POS0", features, positions, false);
		if(conjunction_feature != null) featureLine.add(conjunction_feature);
		
		features = new Class<?>[]{TokenAnnotations.PartOfSpeechAnnotation.class, TokenAnnotations.PartOfSpeechAnnotation.class};
		positions = new int[]{0,1};
		conjunction_feature = getConjuctionFeature(sent, i, "POS0POS1", features, positions, false);
		if(conjunction_feature != null) featureLine.add(conjunction_feature);	
	}
	
	public static void main(String[] args) throws IOException
	{
		;
	}
}
