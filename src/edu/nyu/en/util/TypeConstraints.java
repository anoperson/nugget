package edu.nyu.en.util;

import java.util.HashMap;
import java.util.Map;

import edu.nyu.en.perceptron.types.SentenceInstance;

/**
 * In this class, we make a list of mapping between trigger type and argument type
 * according to the ACE 2005 guidelilne: "Each Event type and subtype will have its own set of potential participant roles for the Entities which occur within the scopes of its exemplars"
 * @author che
 *
 */
public class TypeConstraints
{
	
	// a mapping from event subtype to type
	public static Map<String, String> eventTypeMap = new HashMap<String, String>();
	public static Map<String, String> aceSubTypeMap = new HashMap<String, String>();
//	public static Map<String, String> eventTypeMapModified = new HashMap<String, String>();
	
	static
	{
		eventTypeMap.put("declarebankruptcy", "business");
		
		eventTypeMap.put("artifact", "manufacture");
		
		eventTypeMap.put("startposition", "personnel");
		eventTypeMap.put("endposition", "personnel");
		eventTypeMap.put("nominate", "personnel");
		eventTypeMap.put("elect", "personnel");
		
		eventTypeMap.put("demonstrate", "conflict");
		eventTypeMap.put("attack", "conflict");
		
		eventTypeMap.put("broadcast", "contact");
		eventTypeMap.put("contact", "contact");
		eventTypeMap.put("correspondence", "contact");
		eventTypeMap.put("meet", "contact");
		
		eventTypeMap.put("transfermoney", "transaction");
		eventTypeMap.put("transferownership", "transaction");
		eventTypeMap.put("transaction", "transaction");
		
		eventTypeMap.put("transportartifact", "movement");
		eventTypeMap.put("transportperson", "movement");
		
		eventTypeMap.put("startorg", "business");
		eventTypeMap.put("endorg", "business");
		eventTypeMap.put("mergeorg", "business");
		
		eventTypeMap.put("die", "life");
		eventTypeMap.put("divorce", "life");
		eventTypeMap.put("marry", "life");
		eventTypeMap.put("beborn", "life");
		eventTypeMap.put("injure", "life");
		
		eventTypeMap.put("pardon", "justice");
		eventTypeMap.put("sue", "justice");
		eventTypeMap.put("convict", "justice");
		eventTypeMap.put("chargeindict", "justice");
		eventTypeMap.put("trialhearing", "justice");
		eventTypeMap.put("sentence", "justice");
		eventTypeMap.put("appeal", "justice");
		eventTypeMap.put("releaseparole", "justice");
		eventTypeMap.put("extradite", "justice");
		eventTypeMap.put("fine", "justice");
		eventTypeMap.put("execute", "justice");
		eventTypeMap.put("arrestjail", "justice");
		eventTypeMap.put("acquit", "justice");
		
		aceSubTypeMap.put("Be-Born","beborn");
		aceSubTypeMap.put("Marry","marry");
		aceSubTypeMap.put("Divorce","divorce");
		aceSubTypeMap.put("Injure","injure");
		aceSubTypeMap.put("Die","die");
		//aceSubTypeMap.put("Transport","Movement");
		aceSubTypeMap.put("Transfer-Ownership","transferownership");
		aceSubTypeMap.put("Transfer-Money","transfermoney");
		aceSubTypeMap.put("Start-Org","startorg");
		aceSubTypeMap.put("Merge-Org","mergeorg");
		//aceSubTypeMap.put("Declare-Bankruptcy","Business");
		aceSubTypeMap.put("End-Org","endorg");
		aceSubTypeMap.put("Attack","attack");
		aceSubTypeMap.put("Demonstrate","demonstrate");
		aceSubTypeMap.put("Meet","meet");
		aceSubTypeMap.put("Phone-Write","contact");
		aceSubTypeMap.put("Start-Position","startposition");
		aceSubTypeMap.put("End-Position","endposition");
		aceSubTypeMap.put("Nominate","nominate");
		aceSubTypeMap.put("Elect","elect");
		aceSubTypeMap.put("Arrest-Jail","arrestjail");
		aceSubTypeMap.put("Release-Parole","releaseparole");
		aceSubTypeMap.put("Trial-Hearing","trialhearing");
		aceSubTypeMap.put("Charge-Indict","chargeindict");
		aceSubTypeMap.put("Sue","sue");
		aceSubTypeMap.put("Convict","convict");
		aceSubTypeMap.put("Sentence","sentence");
		aceSubTypeMap.put("Fine","fine");
		aceSubTypeMap.put("Execute","execute");
		aceSubTypeMap.put("Extradite","extradite");
		aceSubTypeMap.put("Acquit","acquit");
		aceSubTypeMap.put("Appeal","appeal");
		aceSubTypeMap.put("Pardon","pardon");
		
//		eventTypeMap.put("Be-Born","Life");
//		eventTypeMap.put("Marry","Life");
//		eventTypeMap.put("Divorce","Life");
//		eventTypeMap.put("Injure","Life");
//		eventTypeMap.put("Die","Life");
//		eventTypeMap.put("Transport","Movement");
//		eventTypeMap.put("Transfer-Ownership","Transaction");
//		eventTypeMap.put("Transfer-Money","Transaction");
//		eventTypeMap.put("Start-Org","Business");
//		eventTypeMap.put("Merge-Org","Business");
//		eventTypeMap.put("Declare-Bankruptcy","Business");
//		eventTypeMap.put("End-Org","Business");
//		eventTypeMap.put("Attack","Conflict");
//		eventTypeMap.put("Demonstrate","Conflict");
//		eventTypeMap.put("Meet","Contact");
//		eventTypeMap.put("Phone-Write","Contact");
//		eventTypeMap.put("Start-Position","Personnel");
//		eventTypeMap.put("End-Position","Personnel");
//		eventTypeMap.put("Nominate","Personnel");
//		eventTypeMap.put("Elect","Personnel");
//		eventTypeMap.put("Arrest-Jail","Justice");
//		eventTypeMap.put("Release-Parole","Justice");
//		eventTypeMap.put("Trial-Hearing","Justice");
//		eventTypeMap.put("Charge-Indict","Justice");
//		eventTypeMap.put("Sue","Justice");
//		eventTypeMap.put("Convict","Justice");
//		eventTypeMap.put("Sentence","Justice");
//		eventTypeMap.put("Fine","Justice");
//		eventTypeMap.put("Execute","Justice");
//		eventTypeMap.put("Extradite","Justice");
//		eventTypeMap.put("Acquit","Justice");
//		eventTypeMap.put("Appeal","Justice");
//		eventTypeMap.put("Pardon","Justice");
//		
//		eventTypeMapModified.put("Be-Born","Life");
//		eventTypeMapModified.put("Marry","Life");
//		eventTypeMapModified.put("Divorce","Life");
//		eventTypeMapModified.put("Transport","Movement");
//		eventTypeMapModified.put("Transfer-Ownership","Transaction");
//		eventTypeMapModified.put("Transfer-Money","Transaction");
//		eventTypeMapModified.put("Start-Org","Business");
//		eventTypeMapModified.put("Merge-Org","Business");
//		eventTypeMapModified.put("Declare-Bankruptcy","Business");
//		eventTypeMapModified.put("End-Org","Business");
//		eventTypeMapModified.put("Injure","Conflict");
//		eventTypeMapModified.put("Die","Conflict");
//		eventTypeMapModified.put("Attack","Conflict");
//		eventTypeMapModified.put("Demonstrate","Conflict");
//		eventTypeMapModified.put("Meet","Contact");
//		eventTypeMapModified.put("Phone-Write","Contact");
//		eventTypeMapModified.put("Start-Position","Personnel");
//		eventTypeMapModified.put("End-Position","Personnel");
//		eventTypeMapModified.put("Nominate","Personnel");
//		eventTypeMapModified.put("Elect","Personnel");
//		eventTypeMapModified.put("Arrest-Jail","Justice");
//		eventTypeMapModified.put("Release-Parole","Justice");
//		eventTypeMapModified.put("Trial-Hearing","Justice");
//		eventTypeMapModified.put("Charge-Indict","Justice");
//		eventTypeMapModified.put("Sue","Justice");
//		eventTypeMapModified.put("Convict","Justice");
//		eventTypeMapModified.put("Sentence","Justice");
//		eventTypeMapModified.put("Fine","Justice");
//		eventTypeMapModified.put("Execute","Justice");
//		eventTypeMapModified.put("Extradite","Justice");
//		eventTypeMapModified.put("Acquit","Justice");
//		eventTypeMapModified.put("Appeal","Justice");
//		eventTypeMapModified.put("Pardon","Justice");
	}
	
	/**
	 * Given an event subtype, return the type. e.g. End_Position --> Personnel
	 * @param type
	 * @return
	 */
	public static String getEventSuperType(String type)
	{
		return eventTypeMap.get(type);
	}
	
	/**
	 * judge if the current node is a possible trigger
	 * basically, if current token is not one of (Verb, Noun, or Adj), it's not a possible trigger
	 * @param problem
	 * @param i
	 * @return
	 */
	public static boolean isPossibleTriggerByPOS(SentenceInstance problem, int i)
	{
		final String allowedPOS = "IN|JJ|RB|DT|VBG|VBD|NN|NNPS|VB|VBN|NNS|VBP|NNP|PRP|VBZ";
		String[] posTags = problem.getPosTags();
		if(posTags[i].matches(allowedPOS))
		{
			return true;
		}
		return false;
	}
	
	
	public static void main(String[] args) {
	}
}
