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

/**
 *  an Ace Event, with information from the ACE key or system output.
 */

public class AceEvent extends AceEventArgumentValue{

	/**
	 *  the type of the event:
	 */
	public String type = null;
	
	/**
	 *  a list of the mentions of this event (each of type AceEventMention)
	 */
	public ArrayList<AceEventMention> mentions = new ArrayList<AceEventMention>();

  /**
   *  create a new event with the specified id, type, subtype, and arguments.
   */

	
	/**
	 *  create an AceEvent from the information in the APF file.
	 *  @param eventElement the XML element from the APF file containing
	 *                       information about this entity
	 *  @param acedoc  the AceDocument of which this AceEvent is a part
	 *  @param fileText  the text of the document
	 */

	public AceEvent (Element eventElement, AceDocument acedoc) {
			id = eventElement.getAttribute("id");
			
			NodeList mentionElements = eventElement.getElementsByTagName("event_mention");
			for (int j=0; j<mentionElements.getLength(); j++) {
				Element mentionElement = (Element) mentionElements.item(j);
				AceEventMention mention = new AceEventMention (this, mentionElement, acedoc);
				addMention(mention);
			}
	}

	public AceEvent() {
		// TODO Auto-generated constructor stub
	}

	void setId (String id) {
		this.id = id;
	}
	
	public int cleanMentions(ArrayList<AceEventMention> storingMentions) {
		ArrayList<Integer> toRemoved = new ArrayList<Integer>();
		for (int i = mentions.size()-1; i>=0; i--) {
			AceEventMention ment = mentions.get(i);
			if (mentionOverlap(ment, storingMentions)) {
				toRemoved.add(i);
			}
			else {
				storingMentions.add(ment);
			}
		}
		
		for (int rm : toRemoved) {
			mentions.remove(rm);
		}
		
		return toRemoved.size();
	}
	
	private boolean mentionOverlap(AceEventMention mention, ArrayList<AceEventMention> storingMentions) {
		boolean res = false;
		for (AceEventMention sment : storingMentions) {
			if (mention.spanEquals(sment)) {
				res = true;
				break;
			}
		}
		return res;
	}

	/**
	 *  add mention 'mention' to the event.
	 */

	public void addMention (AceEventMention mention) {
		mentions.add(mention);
	}

	/**
	 *  write the event to 'w' in APF format.
	 */

	public void write (PrintWriter w) {
		w.println("  <hopper id=\"" + id + "\">");
		for (int i=0; i<mentions.size(); i++) {
			AceEventMention mention = (AceEventMention) mentions.get(i);
			mention.write(w);
		}
		w.println("  </hopper>");
	}

	public String toString () {
		StringBuffer buf = new StringBuffer();
		buf.append("event ");
		buf.append(type);
		buf.append("{");
		for (int i=0; i<mentions.size(); i++) {
			AceEventMention mention = (AceEventMention) mentions.get(i);
			buf.append(mention.toString());
		}
		buf.append("} ");
		return buf.toString();
	}
}
