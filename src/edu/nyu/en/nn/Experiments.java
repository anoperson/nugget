package edu.nyu.en.nn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class Experiments {
	
	public static void analyzeCorpus(String dirPath) throws Exception {
		File dirPathFile = new File(dirPath);
		
		File[] files = dirPathFile.listFiles();
		
		String line = "", tag = "";
		int getIndex = -1;
		HashSet<String> allTags = new HashSet<String>();
		System.out.println("Processing : " + dirPath + " ...");
		for (File file : files) {
			BufferedReader reader = new BufferedReader(new FileReader(file));
			
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				
				//if (line.contains("<a")) System.out.println("-- " + line);
				
				if (line.contains("headline>>")) System.out.println(line);
				
				if (line.startsWith("<") && line.endsWith(">")) {
					line = line.startsWith("</") ? line.substring(2) : line.substring(1);
					
					if (line.indexOf(" ") >= 0) getIndex = line.indexOf(" ");
					else getIndex = line.indexOf(">");
					
					tag = line.substring(0, getIndex);
					
					if (tag.equals("headline>")) System.out.println("---" + line);
					
					allTags.add(tag);
				}
			}
			
			reader.close();
		}
		
		System.out.println("All Tags: ");
		for (String t : allTags) {
			System.out.println(t);
		}
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
	
	static HashMap<Integer, Integer> plain2xml = new HashMap<Integer, Integer>();
	public static List<TextSegment> analyzeOneFile(String filePath) throws Exception {
		
		List<TextSegment> tsgs = new ArrayList<TextSegment>();
		
		plain2xml = new HashMap<Integer, Integer>();
		
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		
		String line = "", part = "", storeLine = "", tag = "";
		StringBuffer plainBuffer = new  StringBuffer(), xmlBuffer = new StringBuffer();
		int xmlId = -1, plainId = -1, startIndex = -1, endIndex = -1, previousPlainId = 0;
		TextSegment its = null;
		while ((line = reader.readLine()) != null) {
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
				if (endIndex < 0) throw new Exception("<> not match!");
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
				its = new TextSegment(tag, previousPlainId, plainId);
				if (its.setCleanText(plainBuffer.substring(previousPlainId, plainId)) && !tag.equalsIgnoreCase("dateline"))
					tsgs.add(its);
				previousPlainId = plainId+1;
			}
		}
		
		reader.close();
		
		System.out.println(plainBuffer.toString() + "---");
		
		HashMap<Integer, Integer> xml2plain = new HashMap<Integer, Integer>();
		for (Integer num : plain2xml.keySet()) xml2plain.put(plain2xml.get(num), num);
		
		System.out.println("*****************");
		//System.out.println("Plain: " + plainBuffer.substring(xml2plain.get(startxml), xml2plain.get(endxml)) + " : " + xml2plain.get(startxml) + " -> " + xml2plain.get(endxml));
		System.out.println("XML  : " + xmlBuffer.substring(startxml, endxml) + " : " + startxml + " -> " + endxml);
		System.out.println("------------" + plainBuffer.length());
		
		System.out.println(">>>>>>>>>>>>>");
		for (int i = 0; i < tsgs.size(); i++) {
			System.out.println("-------segment:------: " + i + " : " + tsgs.get(i).start + " -> " + tsgs.get(i).end + " : " + tsgs.get(i).tag);
			String plainText = tsgs.get(i).text;
			System.out.println("plain: " + plainText);
			String xmlText = xmlBuffer.substring(plain2xml.get(tsgs.get(i).start), plain2xml.get(tsgs.get(i).end));
			System.out.println("xml  : " + xmlText);
			//if (!plainText.equals(xmlText)) throw new Exception("Text are not simialr!!!!");
		}
		System.out.println("<<<<<<<<<<<<<");
		
		return tsgs;
	}
	
	static int start = 101, end = 105, startxml = 1705, endxml = 1712;
	public static void main(String[] args) throws Exception {
		
		String corpusPath = "/Users/thien/workspace/fifth/EventNugget/corpus/LDC2016E36_TAC_KBP_English_Event_Nugget_Detection_and_Coreference_Comprehensive_Training_and_Evaluation_Data_2014-2015/data/2015/eval/source";
		String filePath = "/Users/thien/workspace/fifth/EventNugget/corpus/small/source/bolt-eng-DF-170-181109-48534.txt";
		//analyzeCorpus(corpusPath);
		List<TextSegment> ts = analyzeOneFile(filePath);
		System.out.println();
		for (Integer plain : plain2xml.keySet()) {
			System.out.print("[" + plain + ":" + plain2xml.get(plain) + "] ");
		}
		
		
		//System.out.println(corpusPath.substring(0, 0).isEmpty());
	}

}
