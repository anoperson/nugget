package edu.nyu.en.perceptron.types;

import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import edu.nyu.en.ace.acetypes.AceEventMention;
import edu.nyu.en.ace.acetypes.AceMention;
import edu.nyu.en.perceptron.graph.DependencyGraph;
import edu.nyu.en.util.Span;

/**
 * This class represents the original rich representation about one sentence
 * it may contains some reference to the whole document, or even the whole corpus
 * @author che
 *
 */
public class Sentence implements java.io.Serializable
{
	/**
	 * defines the sentence attributes types
	 * @author check
	 *
	 */
	static public enum Sent_Attribute
	{
		TOKENS,  		// tokens: String[]
		POSTAGS, 		// POS tags: String[]
		CHUNKS,  		// chunk tags: String[]
		DepGraph,		// graph representation about the deps 
		Token_FEATURE_MAPs,   // feature maps for each token: list->map<key,value>s
		TOKEN_SPANS,		  // absolute spans for each token
		ParseTree		// the parse tree
	}
	
	/**
	 * the sentID, the number of sentence in its document
	 */
	protected int sentID;
	
	/**
	 * a reference to its document object
	 */
	protected Document doc;
	
	/**
	 * the span of the sentence
	 */
	private Span extent;
	
	/**
	 * the actual content of the sentence, encoded in a map
	 */
	protected Map<Sent_Attribute, Object> map = new HashMap<Sent_Attribute, Object>();
	
	/**
	 * mentions of events, entities, and values
	 */
	/**
	 * Qi: all mentions of events / values / entities 
	 */
	public List<AceEventMention> eventMentions = new ArrayList<AceEventMention>();
	
	public Sentence(Document doc, int sentID)
	{
		this.doc = doc;
		this.sentID = sentID;
	}
	
	public void fillAceAnnotaions() //boolean singleToken
	{
		if(doc.aceAnnotations != null)
		{
			// push the ace annotations to the sentence object
			fillMentions2Sent(doc.aceAnnotations.eventMentions, eventMentions);
			
			for(Iterator<AceEventMention> iter = eventMentions.iterator(); iter.hasNext();)
			{
				AceEventMention mention = iter.next();
				
//				if (singleToken) {
//					mention.setHeadIndices((Span[]) this.get(Sent_Attribute.TOKEN_SPANS));
//				}
//				else {
//					mention.setHeadIndices((Span[]) this.get(Sent_Attribute.TOKEN_SPANS), (String[]) this.get(Sent_Attribute.POSTAGS));
//				}
				mention.setHeadIndices((Span[]) this.get(Sent_Attribute.TOKEN_SPANS));
				
				if(mention.getHeadIndices() == null || mention.getHeadIndices().size() == 0)
				{
					// remove event mentions whose trigger is empty. They are empty because their pos tags are not one of (Verb, Noun and Adj)
					iter.remove();
					doc.aceAnnotations.eventMentions.remove(mention);
				}
				else {
					doc.aceAnnotations.employedEventMentionIds.add(mention.getId());
				}
			}
		}
	}
	
	/**
	 * fill ace annotations to the sentence 
	 * @param eventMentions2
	 * @param eventMentions3
	 * @param class1
	 */
	private void fillMentions2Sent(List mentions, List sentMentions)
	{
		for(Object obj : mentions)
		{
			if(obj instanceof AceMention)
			{
				AceMention mention = (AceMention) obj;
				Span mention_extent = mention.extent;
				
				// if the mention is overlapped with the sent, add it to the sent 
				if(extent.overlap(mention_extent))
				{
					sentMentions.add(mention);
				}
			}
		}	
	}

	public Object get(Sent_Attribute key)
	{
		return map.get(key);
	}
	
	public void put(Sent_Attribute key, Object value)
	{
		map.put(key, value);
	}

	public void setExtent(Span extent)
	{
		this.extent = extent;
	}

	public Span getExtent()
	{
		return extent;
	}
	
	public int size()
	{
		String[] tokens = (String[]) get(Sent_Attribute.TOKENS);
		if(tokens == null)
		{
			return 0;
		}
		else
		{
			return tokens.length;
		}
	}
	
	public void printBasicSent(PrintStream out)
	{
		String[] tokens = (String[]) this.get(Sent_Attribute.TOKENS);
		String[] posTags = (String[]) this.get(Sent_Attribute.POSTAGS);
		String[] chunks = (String[]) this.get(Sent_Attribute.CHUNKS);
		DependencyGraph graph = (DependencyGraph) this.get(Sent_Attribute.DepGraph);
		
		// print pos/chunks/tdl
		out.print(Arrays.toString(tokens) + "\t");
		out.print(Arrays.toString(posTags) + "\t");
		out.print(Arrays.toString(chunks) + "\t");
		out.print(graph);
		out.println();
		
		// print tokens 
		List<Map<Class<?>, Object>> list =  (List<Map<Class<?>, Object>>) this.get(Sent_Attribute.Token_FEATURE_MAPs);
		for(Map<Class<?>, Object> token_features : list)
		{
			for(Class<?> key : token_features.keySet())
			{
				Object value = token_features.get(key);
				out.print(value);
				out.print("\t");
			}
			out.println();
		}
		
		// print ace annotations
		printAceAnnotatoin(out, this.eventMentions);
	}
	
	private void printAceAnnotatoin(PrintStream out, List mentions)
	{
		for(Object obj : mentions)
		{
			if(obj instanceof AceMention)
			{
				AceMention mention = (AceMention) obj;
				mention.write(new PrintWriter(out, true));
			}
		}
	}
}
