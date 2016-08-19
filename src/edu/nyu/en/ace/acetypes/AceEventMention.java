// -*- tab-width: 4 -*-
//Title:        JET
//Copyright:    2005
//Author:       Ralph Grishman
//Description:  A Java-based Information Extraction Toolkil
//              (ACE extensions)

package edu.nyu.en.ace.acetypes;

import java.util.*;
import java.io.*;

import org.w3c.dom.*;

import edu.nyu.en.nn.DatasetPreparer;
import edu.nyu.en.util.Span;

/**
 *  an Ace event mention, with information from the ACE key.
 */

public class AceEventMention extends AceMention{

	/**
	 * calculate the indice of each head token accodring to span of 
	 * token in this sentence. For Value/Timex, this corrresponds to extent, for Entity this corresponds to head, and anchor for event
	 * @param tokenSpans
	 */
	public void setHeadIndices(Span[] tokenSpans)
	{
		
		int pre = -1;
		
		if(headIndices == null)
		{
			headIndices = new Vector<Integer>();
			int i = 0;
			for(Span tokenSpan : tokenSpans)
			{
				if(this.anchorExtent.overlap(tokenSpan))
				{
					if ((pre >= 0) && (i > (pre+1))) break;
					
					headIndices.add(i);
					pre = i;
				}
				i++;
			}
		}
	}
	
	/**
	 * set the indices of event trigger, if the trigger contains multiple words, choose one of them
	 * only keep triggers with nouns, verbs and adjectives
	 * @param tokenSpans
	 * @param posTags
	 */
	public void setHeadIndices(Span[] tokenSpans, String[] posTags)
	{
		int pre = -1;
		if(headIndices == null)
		{
			headIndices = new Vector<Integer>();
			for(int i=0; i<tokenSpans.length; i++)
			{
				Span tokenSpan = tokenSpans[i];
				if(this.anchorExtent.overlap(tokenSpan))
				{
					if ((pre >= 0) && (i > (pre+1))) break;
					
					headIndices.add(i);
					
					pre = i;
				}
			}
			// if event trigger has more than one word, use simple rule to shrink it
			if(headIndices.size() > 1)
			{
				int final_trigger = -1;
				boolean hasNoun = false;
				boolean hasAdj = false;
				for(Integer index : headIndices)
				{
					if(posTags[index].charAt(0) == 'N' && !hasNoun) // noun
					{
						final_trigger = index;
						hasNoun = true;
					}
					if(posTags[index].charAt(0) == 'J' && !hasNoun && !hasAdj)
					{
						final_trigger = index;
						hasAdj = true;
					}
					if(posTags[index].charAt(0) == 'V' && !hasNoun && !hasAdj) // verb
					{						
						final_trigger = index;
					}
				}
				// set the first token as default if no verbs or nouns
				if(final_trigger == -1)
				{
					final_trigger = headIndices.get(0);
				}
				
				headIndices.clear();
				headIndices.add(final_trigger);
			}
		}
	}
	
	public String realis = null;
	
	/**
	 *  the subtype of the event
	 */
	public String subtype = null;
	
	public Span ldc_extent;
	public String ldc_text;
	/**
	 *  the span of the extent of the event, with start and end positions based
	 *  on Jet offsets (and so including following whitespace).
	 **/
	public Span jetExtent;
	
	
	/**
	 *  the span of the anchor of the event, with start and end positions based
	 *  on the ACE offsets (excluding XML tags).
	 */
	public Span anchorExtent;
	/**
	 *  the span of the anchor of the event, with start and end positions based
	 *  on Jet offsets (and so including following whitespace).
	 **/
	public Span anchorJetExtent;
	/**
	 *  the text of the anchor
	 */
	public String anchorText;
	/**
	 *  our confidence in the presence of this event mention
	 */
	public double confidence = 1.0;

	public AceEvent event;
	
	public AceDocument acedoc;
	
	public String getId() {
		return id;
	}
	
	/**
	 *  create an AceEventMention from the information in the APF file.
	 *
	 *  @param mentionElement the XML element from the APF file containing
	 *                       information about this mention
	 *  @param acedoc        the AceDocument to which this relation mention
	 *                       belongs
	 */

	public AceEventMention (AceEvent event, Element mentionElement, AceDocument acedoc) {
		id = mentionElement.getAttribute("id");
		confidence = 0.0f;
		this.event = event;
		this.acedoc = acedoc;
		
		if (this.event.type == null) {
			this.event.type = mentionElement.getAttribute("type");
		}
		else {
			if (!this.event.type.equals(mentionElement.getAttribute("type"))) {
				System.out.println("type for event mentions not matched: " + this.event.type + " vs " + mentionElement.getAttribute("type"));
				System.exit(-1);
			}
		}
		
		this.subtype = mentionElement.getAttribute("subtype");
		
		this.realis = mentionElement.getAttribute("realis");
		
		DatasetPreparer.subtype2type.put(this.subtype, this.event.type);
		
		NodeList triggers = mentionElement.getElementsByTagName("trigger");
		Element triggerElement = (Element) triggers.item(0);
		int offset = Integer.parseInt(triggerElement.getAttribute("offset"));
		int length = Integer.parseInt(triggerElement.getAttribute("length"));
		this.ldc_text = triggerElement.getNodeValue();
		
		Integer start = offset;
		Integer end = start + length - 1;
		
		//System.out.println("---" + start + " : " + end + " : " + ldc_text);
		
		start = acedoc.sourceDoc.xml2plain.get(start);
		end = acedoc.sourceDoc.xml2plain.get(end);
		
		if (start == null || end == null) {
			System.out.println("Cannot find indexes in xml2plain mapping: " + start + " : " + end);
			System.exit(-1);
		}
		
		anchorExtent = new Span(start, end);
		anchorJetExtent = new Span(start, end+1);
		anchorText = acedoc.sourceDoc.text.substring(start, end+1);
		
		extent = new Span(start, end);;
		jetExtent = new Span(start, end+1);;
		text = acedoc.sourceDoc.text.substring(start, end+1);
	}

	void setId (String id) {
		this.id = id;
	}

	/**
	 *  write the APF representation of the event mention to <CODE>w</CODE>.
	 */
	 
	public void write (PrintWriter w) {
		w.println("    <event_mention id=\"" + id + "\"" + " type=\"" + getType() + "\"" +
				" subtype=\"" + getSubType() + "\"" +
				" realis=\"" + getRealis() + "\"" +
				">");
		w.println("      <trigger source=\"" + this.acedoc.sourceDoc.docID + "\"" +
				" offset=\"" + this.acedoc.sourceDoc.plain2xml.get(anchorExtent.start()) +  "\"" + 
				" length=\"" + (anchorExtent.end()-anchorExtent.start()+1) + "\"" + ">" + 
				anchorText + "</trigger>");
		w.println("    </event_mention>");
	}

	public boolean equals (Object o) {
		if (!(o instanceof AceEventMention))
			return false;
		AceEventMention p = (AceEventMention) o;
		if (!this.subtype.equals(p.subtype))
			return false;
		if (!this.anchorExtent.overlap(p.anchorExtent))
			return false;
		/*if (this.arguments.size()!=p.arguments.size())
			return false;
		for (int i=0;i<this.arguments.size();i++){
			if (!checkArg(this.arguments.get(i),p.arguments))
				return false;
		}*/
		return true;
	}
	
	public boolean spanEquals(Object o) {
		if (!(o instanceof AceEventMention)) return false;
		AceEventMention p = (AceEventMention) o;
		return this.anchorExtent.overlap(p.anchorExtent);
	}
	
	public String toString () {
		StringBuffer buf = new StringBuffer();
		buf.append(anchorText);
		buf.append(" ");
		buf.append(subtype);
		buf.append(" ");
		buf.append(realis);
		buf.append(" ");
		// buf.append("[" + text + "]"); // display extent
		return buf.toString();
	}

	@Override
	public AceEventArgumentValue getParent()
	{
		return this.event;
	}

	@Override
	public String getType()
	{
		return this.event.type;
	}

	public String getSubType()
	{
		return this.subtype;
	}
	
	public String getRealis() {
		return this.realis;
	}
}
