package edu.nyu.en.perceptron.types;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.SerializationException;
import org.apache.commons.lang3.SerializationUtils;

import opennlp.tools.util.InvalidFormatException;

import edu.nyu.en.util.Span;

import edu.nyu.en.ace.acetypes.AceDocument;
import edu.nyu.en.nn.DatasetPreparer;
import edu.nyu.en.perceptron.core.Controller;
import edu.nyu.en.perceptron.featureGenerator.TextFeatureGenerator;
import edu.nyu.en.perceptron.types.Sentence.Sent_Attribute;
import edu.nyu.en.util.SentDetectorWrapper;
import edu.nyu.en.util.TokenizerWrapper;
import edu.nyu.en.util.TypeConstraints;

/**
 * read the source data of i2b2
 * @author che
 *
 */
public class Document implements java.io.Serializable
{
	private static final long serialVersionUID = 2307017698146800811L;
	
	static public final String preprocessedFileExt = ".preprocessed";
	
	// the id (base file name) of the document
	public String docID;
	public String text;
	public String xmlText;
	/*
	 * the mapping from plain to xml
	 */
	public HashMap<Integer, Integer> plain2xml;
	public HashMap<Integer, Integer> xml2plain;
	
	/* the list of sentences
	 * they are instances in the learning process, there can be a dummy list of sentences, where each sentence is a cluster of sentence
	 * e.g. the dummy sentence can be concatenation of sentences that linked by entity coreference 
	 */
	protected List<Sentence> sentences;
	
	/**
	 * this object contains the parsed information from apf file (pretty much everything)
	 * it can be considered as gold standard for event extraction or can provide perfect entities etc.
	 */
	protected AceDocument aceAnnotations;
	
	// Event type --> Trigger token
	public static Map<String, List<String>> triggerTokens = new HashMap<String, List<String>>();
	// Event subtype --> Trigger token
	public static Map<String, List<String>> triggerTokensFineGrained = new HashMap<String, List<String>>();
	// Event subtype --> trigger token with high confidence value
	public static Map<String, List<String>> triggerTokensHighQuality = new HashMap<String, List<String>>();
	
	static
	{
		// initialize priorityQueueEntities
		try
		{
			// initialize dict of triggerTokens
			BufferedReader reader = new BufferedReader(new FileReader(DatasetPreparer.coreDataPath + "data/triggerTokens"));
			String line = null;
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
				if(confidence < 0.150)
				{
					continue;
				}
				
				if (!TypeConstraints.aceSubTypeMap.containsKey(eventSubType)) continue;
				
				eventSubType = TypeConstraints.aceSubTypeMap.get(eventSubType);
				
				String eventType = TypeConstraints.eventTypeMap.get(eventSubType);
				List<String> triggers = triggerTokens.get(eventType);
				if(triggers == null)
				{
					triggers = new ArrayList<String>();
					triggerTokens.put(eventType, triggers);
				}
				if(!triggers.contains(triggerToken))
				{
					triggers.add(triggerToken);
				}
				
				triggers = triggerTokensFineGrained.get(eventSubType);
				if(triggers == null)
				{
					triggers = new ArrayList<String>();
					triggerTokensFineGrained.put(eventSubType, triggers);
				}
				if(!triggers.contains(triggerToken))
				{
					triggers.add(triggerToken);
				}
				
				if(confidence >= 0.50)
				{
					triggers = triggerTokensHighQuality.get(eventSubType);
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
	
	public AceDocument getAceDocument() {
		return this.aceAnnotations;
	}
	
	/**
	 * implicit constructor
	 */
	protected Document()
	{}
	

	public Document(File sourceFile, String nuggetDir, String hopperDir) throws Exception
	{
		String name = sourceFile.getName();
		name = name.substring(0, name.lastIndexOf("."));
		docID = name;
		
		sentences = new ArrayList<Sentence>();
		readDoc(sourceFile);
		
		if(nuggetDir != null && hopperDir != null)
		{
			String nuggetFilePath = nuggetDir + "/" + name + ".event_nuggets.xml";
			String hopperFilePath = hopperDir + "/" + name + ".event_hoppers.xml";
			setAceAnnotations(new AceDocument(this, nuggetFilePath, hopperFilePath));
		}
		else {
			setAceAnnotations(null);
		}
	}
	
	public static Document createAndPreprocess(File sourceFile, String nuggetDir, String hopperDir, boolean tryLoadExisting, boolean dumpNewDoc) throws Exception, SerializationException {
		Document doc = null;
		File preprocessed = new File(sourceFile + preprocessedFileExt);
		if (tryLoadExisting && preprocessed.isFile()) {
			doc = (Document) SerializationUtils.deserialize(new FileInputStream(preprocessed));
		}
		
		if (doc==null) {
			doc = new Document(sourceFile, nuggetDir, hopperDir);
			TextFeatureGenerator.doPreprocess(doc);
			
			if (dumpNewDoc) {
				try {
					SerializationUtils.serialize(doc, new FileOutputStream(preprocessed));
				}
				catch (IOException e) {
					Files.deleteIfExists(preprocessed.toPath());
					throw e;
				}
				catch (SerializationException e) {
					Files.deleteIfExists(preprocessed.toPath());
					throw e;
				}
			}
		}
		return doc;
	}
	
	public static class TextSegment
	{
		public String tag;
		public String text;
		public int start = -1;
		public int end = -1;
		
		public boolean hasTag()
		{
			if(tag == null || tag.equals(""))
			{
				return false;
			}
			return true;
		}
		
		public TextSegment(String tag, int start, int end)
		{
			this.tag = tag;
			this.start = start;
			this.end = end;
		}
		
		public void setText(String text) {
			this.text = text;
		}
		
		public boolean setCleanText(String intext) {
			String str = intext;
			
			int startPos = 0;
			while (startPos < intext.length() && 
					(intext.charAt(startPos) == '\n' || intext.charAt(startPos) == ' ' || intext.charAt(startPos) == '\t')) {
				startPos++;
				this.start++;
			}
			
			int endPos = intext.length()-1;
			while (endPos > 0 && 
					(intext.charAt(endPos) == '\n' || intext.charAt(endPos) == ' ' || intext.charAt(endPos) == '\t')) {
				endPos--;
				this.end--;
			}
			
			if (endPos > startPos) {
				this.text = intext.substring(startPos, endPos+1);
				return true;
			}
			return false;
		}
	}
	
	/**
	 * read txt document, do POS tagging, chunking, parsing
	 * @param txtFile
	 * @throws IOException
	 */
	public void readDoc(File sourceFile) throws Exception
	{
		// read text from the original data
		List<TextSegment> segmnets = getSegments(sourceFile);
		
		// do sentence split and tokenization
		Span[] sentSpans = null;
		sentSpans = splitSents(segmnets);
		
		int sentID = 0;
		for(Span sentSpan : sentSpans)
		{	
			String sentText = sentSpan.getCoveredText(this.text).toString();
			edu.nyu.en.util.Span[] tokenSpans = TokenizerWrapper.getTokenizer().tokenizeSpan(sentText);
			for(int idx=0; idx < tokenSpans.length; idx++)
			{
				// record offset for each token/sentence
				Span tokenSpan = tokenSpans[idx];
				// calculate absolute offset for tokens
				int offset = sentSpan.start();
				int absoluteStart = offset + tokenSpan.start();
				int absoluteEnd = offset + tokenSpan.end();
				Span absoluteTokenSpan = new Span(absoluteStart, absoluteEnd);
				tokenSpans[idx] = absoluteTokenSpan;
			}
			
			Sentence sent = new Sentence(this, sentID++);
			sent.put(Sent_Attribute.TOKEN_SPANS, tokenSpans);
			
			String[] tokens = new String[tokenSpans.length];
			for(int idx=0; idx < tokenSpans.length; idx++)
			{
				// get tokens
				Span tokenSpan = tokenSpans[idx];
				tokens[idx] = tokenSpan.getCoveredText(this.text).toString();
			}
			
			sent.put(Sent_Attribute.TOKENS, tokens);
			// save span of the sent
			sent.setExtent(sentSpan);
			List<Map<Class<?>, Object>> tokenFeatureMaps = new ArrayList<Map<Class<?>, Object>>();
			sent.put(Sent_Attribute.Token_FEATURE_MAPs, tokenFeatureMaps);
			this.sentences.add(sent);
		}
	}

	private Span[] splitSents(List<TextSegment> segments) throws InvalidFormatException, IOException
	{
		List<Span> ret = new ArrayList<Span>();
		for(TextSegment sgm : segments)
		{
			Span[] sentSpans = SentDetectorWrapper.getSentDetector().detectPos(sgm.text);
			
			for(Span sent : sentSpans)
			{
				sent.setStart(sent.start() + sgm.start);
				sent.setEnd(sent.end() + sgm.start);
				ret.add(sent);
			}
		}
		
		return ret.toArray(new Span[ret.size()]);
	}

	protected List<TextSegment> getSegments(File sourceFile) throws Exception
	{
		List<TextSegment> tsgs = new ArrayList<TextSegment>();
		
		this.plain2xml = new HashMap<Integer, Integer>();
		
		BufferedReader reader = new BufferedReader(new FileReader(sourceFile));
		
		String line = "", part = "", storeLine = "", tag = "";
		StringBuffer plainBuffer = new  StringBuffer(), xmlBuffer = new StringBuffer();
		int xmlId = -1, plainId = -1, startIndex = -1, endIndex = -1, previousPlainId = 0;
		TextSegment its = null;
		while ((line = reader.readLine()) != null) {
			if (line.equals("</DOC")) line += ">"; //handle some exceptions
			storeLine = line;
			xmlBuffer.append(line);
			while (line.indexOf("<") >= 0) {
				startIndex = line.indexOf("<");
				part = line.substring(0, startIndex);
				for (int i = 1; i <= startIndex; i++) {
					plain2xml.put(++plainId,++xmlId);
				}
				plainBuffer.append(part);
				
				xmlId++;
				
				line = line.substring(startIndex+1);
				
				endIndex = line.indexOf(">");
				if (endIndex < 0) throw new Exception("<> not match!: " + storeLine);
				xmlId += endIndex + 1;
				line = line.substring(endIndex+1);
			}
			if (!line.isEmpty()) {
				for (int i = 1; i <= line.length(); i++) {
					plain2xml.put(++plainId,++xmlId);
				}
				plainBuffer.append(line);
			}
			
			//new line
			plain2xml.put(++plainId,++xmlId);
			plainBuffer.append("\n");
			xmlBuffer.append("\n");
			
			if (storeLine.startsWith("</") && storeLine.endsWith(">")) {
				tag = storeLine.substring(2, storeLine.length()-1);
				its = new TextSegment(tag, previousPlainId, plainId-1);
				if (its.setCleanText(plainBuffer.substring(previousPlainId, plainId)) && !tag.equalsIgnoreCase("dateline"))
					tsgs.add(its);
				previousPlainId = plainId+1;
			}
		}
		
		reader.close();
		
		this.text = plainBuffer.toString();
		this.xmlText = xmlBuffer.toString();
		
		this.xml2plain = new HashMap<Integer, Integer>();
		for (Integer pl : this.plain2xml.keySet())
			this.xml2plain.put(this.plain2xml.get(pl), pl);
		
		return tsgs;
	}
	

	public void printDocBasic(PrintStream out)
	{
		for(int i=0; i<this.sentences.size(); i++)
		{
			Sentence sent = this.sentences.get(i);
			out.println("Sent num:\t" + i);
			sent.printBasicSent(out);
		}
	}
	
	public List<Sentence> getSentences() 
	{
		return this.sentences;
	}
	
	protected void setAceAnnotations(AceDocument aceAnnotations)
	{
		this.aceAnnotations = aceAnnotations;
	}

	public AceDocument getAceAnnotations()
	{
		return aceAnnotations;
	}
	
	
	static public void main(String[] args) throws IOException
	{
	}

}
