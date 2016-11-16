package edu.nyu.en.perceptron.featureGenerator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Vector;

import edu.nyu.en.nn.DatasetPreparer;
import edu.nyu.en.perceptron.graph.DependencyGraph;
import edu.nyu.en.perceptron.graph.GraphNode;
import edu.nyu.en.perceptron.graph.GraphEdge;
import edu.nyu.en.perceptron.types.Document;
import edu.nyu.en.perceptron.types.Sentence;
import edu.nyu.en.perceptron.types.Sentence.Sent_Attribute;
import edu.nyu.en.util.BrownClusters;
import edu.nyu.en.util.ChunkWrapper;
import edu.nyu.en.util.Nomlex;
import edu.nyu.en.util.POSTaggerWrapperStanford;
import edu.nyu.en.util.ParserWrapper;
import edu.nyu.en.util.Span;
import edu.nyu.en.util.TokenAnnotations;
import edu.nyu.en.util.WordNetWrapper;
import edu.nyu.en.util.ParserWrapper.ParseResult;
import edu.mit.jwi.item.ISynset;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;

/**
 * this class is to implement features based on data structure of Document
 * this class may need read some resources, like dictionaries
 * this is for local feature generator
 */
public class TextFeatureGenerator 
{
	// dictionary for title/time words
	Map<String, List<String>> dicts = new HashMap<String, List<String>>();
	
	public void readDict(String dirName) throws IOException
	{
		File dir = new File(dirName);
		File[] children = dir.listFiles();
		for(File child : children)
		{
			if(child.getName().contains("svn") || child.isHidden())
			{
				continue;
			}
			BufferedReader reader = new BufferedReader(new FileReader(child));
			String line = "";
			while((line = reader.readLine()) != null)
			{
				line = line.trim();
				line = line.toLowerCase();
				if(line.length() > 0)
				{
					List<String> list = dicts.get(child.getName());
					if(list == null)
					{
						list = new ArrayList<String>();
						dicts.put(child.getName(), list);
					}
					list.add(line);
				}
			}
			reader.close();
		}
	}
	
	public TextFeatureGenerator() 
	{
		try
		{
			readDict(DatasetPreparer.coreDataPath + "data/dict");
		} 
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}
	
	public static void doPreprocessCheap(Document doc) //, boolean singleToken
	{
		for(Sentence sent : doc.getSentences())
		{
			String[] tokens = (String[]) sent.get(Sent_Attribute.TOKENS); 
			Span[] tokenSpans = (Span[]) sent.get(Sent_Attribute.TOKEN_SPANS);
			try
			{
				// get pos tags
				String[] posTags;
				posTags = POSTaggerWrapperStanford.getPosTagger().posTag(tokens);
				sent.put(Sent_Attribute.POSTAGS, posTags);
				// get chunks
				String[] chunks = ChunkWrapper.getChunker().chunk(tokens, posTags);
				sent.put(Sent_Attribute.CHUNKS, chunks);
				
				List<Map<Class<?>, Object>> tokenFeatureMaps = new ArrayList<Map<Class<?>, Object>>();
				sent.put(Sent_Attribute.Token_FEATURE_MAPs, tokenFeatureMaps);
				for(int idx=0; idx < tokenSpans.length; idx++)
				{
					HashMap<Class<?>, Object> map = new HashMap<Class<?>, Object>();
					// change the first token in each sentence to lowercase
					if(idx == 0 && tokens[0] != null && Character.isUpperCase(tokens[0].charAt(0)))
					{
						tokens[0] = tokens[0].toLowerCase();
					}
					map.put(TokenAnnotations.TextAnnotation.class, tokens[idx]);
					map.put(TokenAnnotations.PartOfSpeechAnnotation.class, posTags[idx]);
					String lemma = ParserWrapper.lemmanize(tokens[idx], posTags[idx]);
					map.put(TokenAnnotations.LemmaAnnotation.class, lemma.toLowerCase());
					map.put(TokenAnnotations.SpanAnnotation.class, tokenSpans[idx]);
					tokenFeatureMaps.add(map);
				}
				// fill in ace annotations such as event/relation/entity mentions
				sent.fillAceAnnotaions(); //singleToken
			
				// synonyms etc.
//				fillFeatures_local(doc);
			} 
			catch (IOException e)
			{
				e.printStackTrace();
			}
		}
	}
	
	public static void doPreprocess(Document doc)
	{
		for(Sentence sent : doc.getSentences())
		{
			String[] tokens = (String[]) sent.get(Sent_Attribute.TOKENS); 
			Span[] tokenSpans = (Span[]) sent.get(Sent_Attribute.TOKEN_SPANS);
			try
			{
				// get pos tags
				String[] posTags;
				posTags = POSTaggerWrapperStanford.getPosTagger().posTag(tokens);
				sent.put(Sent_Attribute.POSTAGS, posTags);
				
				// get parse/deps tree
				ParseResult parse = ParserWrapper.getParserWrapper().getTypedDeps(tokens);
				// create dependency graph representation, in order to faciliate graph manupulation
				Collection<TypedDependency> tdl = parse.deps;
				DependencyGraph graph = new DependencyGraph(tdl, tokens.length);
				sent.put(Sentence.Sent_Attribute.DepGraph, graph);
				sent.put(Sent_Attribute.ParseTree, parse.tree);
				
				// get chunks
				String[] chunks = ChunkWrapper.getChunker().chunk(tokens, posTags);
				sent.put(Sent_Attribute.CHUNKS, chunks);
				
				List<Map<Class<?>, Object>> tokenFeatureMaps = new ArrayList<Map<Class<?>, Object>>();
				sent.put(Sent_Attribute.Token_FEATURE_MAPs, tokenFeatureMaps);
				for(int idx=0; idx < tokenSpans.length; idx++)
				{
					HashMap<Class<?>, Object> map = new HashMap<Class<?>, Object>();
					// change the first token in each sentence to lowercase
					if(idx == 0 && tokens[0] != null && Character.isUpperCase(tokens[0].charAt(0)))
					{
						tokens[0] = tokens[0].toLowerCase();
					}
					map.put(TokenAnnotations.TextAnnotation.class, tokens[idx]);
					map.put(TokenAnnotations.PartOfSpeechAnnotation.class, posTags[idx]);
					String lemma = ParserWrapper.lemmanize(tokens[idx], posTags[idx]).toLowerCase();
					map.put(TokenAnnotations.LemmaAnnotation.class, lemma);
					map.put(TokenAnnotations.ChunkingAnnotation.class, chunks[idx]);
					map.put(TokenAnnotations.SpanAnnotation.class, tokenSpans[idx]);
					
					// get base form of verb and noun according to Nomlex. e.g. retirement --> retire
					if(posTags[idx].startsWith("V") && Nomlex.getSingleTon().contains(lemma))
					{
						map.put(TokenAnnotations.NomlexbaseAnnotation.class, lemma);
					}
					else if(posTags[idx].startsWith("N"))
					{	
						String comlexBase = Nomlex.getSingleTon().getBaseForm(lemma);
						if(comlexBase != null)
						{
							map.put(TokenAnnotations.NomlexbaseAnnotation.class, comlexBase);
						}
					}
					
					tokenFeatureMaps.add(map);
				}
				
			} 
			catch (IOException e)
			{
				e.printStackTrace();
				return;
			}
		}
	}
	
	public static void fillAceAnnotations(Document doc) { //, boolean singleToken
		for(Sentence sent : doc.getSentences()) {
			// fill in ace annotations such as event/relation/entity mentions
			sent.fillAceAnnotaions(); //singleToken
		}
	}
	
	protected static void fillClauseNumber(Document doc)
	{
		for(Sentence sent : doc.getSentences())
		{
			fillClauseNumber(sent);
		}
	}
	
	/**
	 * fill in clause number for each token in the sentence 
	 * tokens with the same clause number are considered in the same clause
	 * @param sent
	 */
	protected static void fillClauseNumber(Sentence sent)
	{
		List<Map<Class<?>, Object>> tokens = (List<Map<Class<?>, Object>>) sent.get(Sent_Attribute.Token_FEATURE_MAPs);
		Tree root = (Tree) sent.get(Sent_Attribute.ParseTree);
		if(root == null)
		{
			return;
		}
		int clauseNum = 0;
		// default clause number is 0
		for(int i=0; i<tokens.size(); i++)
		{
			Map<Class<?>, Object> current_token = tokens.get(i);
			current_token.put(TokenAnnotations.ClauseAnnotation.class, clauseNum);
		}
		String snodes = "S|SBAR|SBARQ|SQ";
		List<Tree> terminals = root.getLeaves();
		// traverse the tree
		Queue<Tree> queue = new LinkedList<Tree>();
		queue.add(root);
		while(!queue.isEmpty())
		{
			Tree node = queue.poll();
			Tree[] children = node.children();
			for(Tree child : children)
			{
				if(!child.isPreTerminal())
				{
					if(child.value().matches(snodes))
					{
						clauseNum++;
						// a clause node, get all leaves
						List<Tree> leaves = child.getLeaves();
						for(Tree leaf : leaves)
						{
							int index = terminals.indexOf(leaf);
							Map<Class<?>, Object> token = tokens.get(index);
							token.put(TokenAnnotations.ClauseAnnotation.class, clauseNum);
						}
						queue.add(child);
					}
					else
					{
						queue.add(child);
					}
				}
			}
		}
	}
	
	protected static void fillDependencyFeatures(Document doc)
	{
		for(Sentence sent : doc.getSentences())
		{
			fillDependencyFeatures(sent);
		}
	}
	
	protected static void fillDependencyFeatures4NN(Document doc)
	{
		for(Sentence sent : doc.getSentences())
		{
			fillDependencyFeatures4NN(sent);
		}
	}
	
	/**
	 * fill in dependency features
	 * or it's ROOT of the sentences
	 * before calling this, the dictionary features should have been filled
	 */
	protected static void fillDependencyFeatures(Sentence sent)
	{
		List<Map<Class<?>, Object>> tokens = (List<Map<Class<?>, Object>>) sent.get(Sent_Attribute.Token_FEATURE_MAPs);
		DependencyGraph graph = (DependencyGraph) sent.get(Sent_Attribute.DepGraph);
		
		// traverse each dependency link
		if(graph == null)
		{
			return;
		}
		
		for(int i=0; i<tokens.size(); i++)
		{
			Map<Class<?>, Object> current_token = tokens.get(i);
			GraphNode node = graph.getVertices().get(i);
			for(GraphEdge edge : node.edges)
			{
				int index = edge.getGovernor();
				String label = "Gov";
				if(index == i)
				{
					index = edge.getDependent();
					label = "Dep";
				}				
				Vector<String> dep_features = (Vector<String>) current_token.get(TokenAnnotations.DependencyAnnotation.class);
				if(dep_features == null)
				{
					dep_features = new Vector<String>(); 
				}
				// use entity information to normalize the dependencies
				String related_word = "";
				String related_pos = ""; 
					
				// use the lemma to normalize the dependencies
				Map<Class<?>, Object> related_token = tokens.get(index);

				related_word = (String) related_token.get(TokenAnnotations.LemmaAnnotation.class);
				// use the pos to normalize the depdencies
				related_pos = (String) related_token.get(TokenAnnotations.PartOfSpeechAnnotation.class);
				
				if(related_word.length() > 0)
				{
					String feature = label + "=" + edge.getRelation() + "_" + related_word;
					dep_features.add(feature);
				}
				if(related_pos.length() > 0)
				{
					String feature = label + "=" + edge.getRelation() + "_" + related_pos;
					dep_features.add(feature);
				}
				// only consider the dependency type
				String feature = label + "=" + edge.getRelation();
				dep_features.add(feature);
				
				current_token.put(TokenAnnotations.DependencyAnnotation.class, dep_features);
			}
		}
	}
	
	protected static void fillDependencyFeatures4NN(Sentence sent)
	{
		List<Map<Class<?>, Object>> tokens = (List<Map<Class<?>, Object>>) sent.get(Sent_Attribute.Token_FEATURE_MAPs);
		DependencyGraph graph = (DependencyGraph) sent.get(Sent_Attribute.DepGraph);
		
		// traverse each dependency link
		if(graph == null)
		{
			return;
		}
		
		for(int i=0; i<tokens.size(); i++)
		{
			Map<Class<?>, Object> current_token = tokens.get(i);
			GraphNode node = graph.getVertices().get(i);
			for(GraphEdge edge : node.edges)
			{
				int index = edge.getGovernor();
				String label = "Gov";
				if(index == i)
				{
					index = edge.getDependent();
					label = "Dep";
				}				
				Vector<String> dep_features = (Vector<String>) current_token.get(TokenAnnotations.DependencyAnnotation4NN.class);
				if(dep_features == null)
				{
					dep_features = new Vector<String>(); 
				}

				String feature = label + "=" + edge.getRelation();
				dep_features.add(feature);
				
				current_token.put(TokenAnnotations.DependencyAnnotation4NN.class, dep_features);
			}
		}
	}
	
	protected void fillEntityInformation(Document doc)
	{
		for(Sentence sent : doc.getSentences())
		{
			fillEntityInformation(sent);
		}
	}
	
	protected void fillEntityInformation4NN(Document doc)
	{
		for(Sentence sent : doc.getSentences())
		{
			fillEntityInformation4NN(sent);
		}
	}
	
	protected void fillEntityInformation(Sentence sent)
	{
		List<Map<Class<?>, Object>> tokens = (List<Map<Class<?>, Object>>) sent.get(Sent_Attribute.Token_FEATURE_MAPs);
		
		// use Title list to detect titles
		for(int index=0; index<sent.size(); index++)
		{
			String text = (String) tokens.get(index).get(TokenAnnotations.TextAnnotation.class);
			text = text.toLowerCase();
			String pos = (String) tokens.get(index).get(TokenAnnotations.PartOfSpeechAnnotation.class);
			if(pos.startsWith("N")) // only consider Non for Title
			{
				if(dicts.get("TITLE") != null && dicts.get("TITLE").contains(text))
				{
					List<String> entityInfo = (List<String>) tokens.get(index).get(TokenAnnotations.EntityAnnotation.class);
					if(entityInfo == null)
					{
						entityInfo = new ArrayList<String>();
						tokens.get(index).put(TokenAnnotations.EntityAnnotation.class, entityInfo);
					}
					
					if(!entityInfo.contains("Title"))
					{
						entityInfo.add("Title");
					}
				}
			}
		}	
	}
	
	protected void fillEntityInformation4NN(Sentence sent)
	{
		List<Map<Class<?>, Object>> tokens = (List<Map<Class<?>, Object>>) sent.get(Sent_Attribute.Token_FEATURE_MAPs);
		
		// use Title list to detect titles
		for(int index=0; index<sent.size(); index++)
		{
			String text = (String) tokens.get(index).get(TokenAnnotations.TextAnnotation.class);
			text = text.toLowerCase();
			String pos = (String) tokens.get(index).get(TokenAnnotations.PartOfSpeechAnnotation.class);
			if(pos.startsWith("N")) // only consider Non for Title
			{
				if(dicts.get("TITLE") != null && dicts.get("TITLE").contains(text))
				{
					List<String> entityInfo = (List<String>) tokens.get(index).get(TokenAnnotations.EntityAnnotation4NN.class);
					if(entityInfo == null)
					{
						entityInfo = new ArrayList<String>();
						tokens.get(index).put(TokenAnnotations.EntityAnnotation4NN.class, entityInfo);
					}
					
					if(!entityInfo.contains("Title"))
					{
						entityInfo.add("Title");
					}
				}
			}
		}	
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
	 * fill in features for a Document doc
	 * this is only API provided by this class, derived class should Override this method
	 * @param doc
	 * @throws IOException 
	 */
	
	public void fillTextFeatures_NoPreprocessing(Document doc) throws IOException { //, boolean singleToken
		fillAceAnnotations(doc); //, singleToken
		fillFeatures_local(doc);
		fillEntityInformation(doc);
		fillDependencyFeatures(doc);
		
		fillEntityInformation4NN(doc);
		fillDependencyFeatures4NN(doc);
		
		fillClauseNumber(doc);
	}
	
	/**
	 * fill in features the are local to a token
	 * @param doc
	 * @throws IOException 
	 */
	protected static void fillFeatures_local(Document doc) throws IOException 
	{
		for(Sentence sent : doc.getSentences())
		{
			List<Map<Class<?>, Object>> tokens = (List<Map<Class<?>, Object>>) sent.get(Sent_Attribute.Token_FEATURE_MAPs);
			for(Map<Class<?>, Object> token : tokens)
			{				
				// Wordnet Hypernym
				String text = (String) token.get(TokenAnnotations.TextAnnotation.class);
				String lemma = (String) token.get(TokenAnnotations.LemmaAnnotation.class);
				String pos = (String) token.get(TokenAnnotations.PartOfSpeechAnnotation.class);
				ISynset hypernym = WordNetWrapper.getSingleTon().getHypernym(lemma, pos);
				if(hypernym != null)
				{
					token.put(TokenAnnotations.HypernymAnnotation.class, hypernym.getID().toString());
				}
				
				// Wordnet Synonyms
				List<String> synonyms = WordNetWrapper.getSingleTon().getSynonyms(lemma, pos);
				if(synonyms != null)
				{
					token.put(TokenAnnotations.SynonymsAnnotation.class, synonyms);
				}
				
				// brown clusters
				List<String> clusters = BrownClusters.getSingleton().getBrownCluster(text);
				if(clusters != null)
				{
					token.put(TokenAnnotations.BrownClusterAnnotation.class, clusters);
				}
			}
		}		
	}

	/**
	 * given a lemma, get possible event types according to 
	 * stats from training data
	 * @param lemma
	 * @return
	 */
	static List<String> getPotentialEventTypes(String lemma)
	{
		List<String> ret = new ArrayList<String>();
		for(String eventType : Document.triggerTokensFineGrained.keySet())
		{
			if(Document.triggerTokensFineGrained.get(eventType).contains(lemma))
			{
				ret.add(eventType);
			}
		}
		
		return ret;
	}
	
	static List<String> getPotentialEventTypesHighConf(String lemma)
	{
		List<String> ret = new ArrayList<String>();
		for(String eventType : Document.triggerTokensHighQuality.keySet())
		{
			if(Document.triggerTokensHighQuality.get(eventType).contains(lemma))
			{
				ret.add(eventType);
			}
		}
		
		return ret;
	}
	
}
