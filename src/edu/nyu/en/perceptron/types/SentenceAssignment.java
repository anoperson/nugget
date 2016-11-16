package edu.nyu.en.perceptron.types;

import java.util.Vector;

import edu.nyu.en.ace.acetypes.AceEventMention;
import edu.nyu.en.perceptron.core.Controller;
import edu.nyu.en.util.TypeConstraints;

/**
 * For the (target) assignment, it should encode two types of assignment:
 * (1) label assignment for each token: refers to the event trigger classification 
 * (2) assignment for any sub-structure of the sentence, e.g. one assignment indicats that 
 * the second token is argument of the first trigger
 * in the simplest case, this type of assignment involves two tokens in the sentence
 * 
 * 
 * @author che
 *
 */
public class SentenceAssignment
{
	public static final String PAD_Trigger_Label = "O"; // pad for the intial state
	public static final String Default_TYPE_Label = "NONE"; //O
	public static final String Default_SUBTYPE_Label = "NONE"; //O
	public static final String Default_REALIS_Label = "NONE"; //O
	public static final String Default_Id_Label = "NONE";
	
	// the alphabet of the label for each node (token), shared by the whole application
	// they should be consistent with SentenceInstance object
	public Alphabet nodeTypeTargetAlphabet;
	public Alphabet nodeSubTypeTargetAlphabet;
	public Alphabet nodeRealisTargetAlphabet;
	
	public Controller controller;
	
	/**
	 * assignment to each node, node-->assignment
	 */
	protected Vector<Integer> nodeTypeAssignment;
	protected Vector<Integer> nodeSubTypeAssignment;
	protected Vector<Integer> nodeRealisAssignment;
	protected Vector<String> nodeIdAssignment;
	
	public Vector<Integer> getNodeTypeAssignment()
	{
		return nodeTypeAssignment;
	}
	
	public Vector<Integer> getNodeSubTypeAssignment()
	{
		return nodeSubTypeAssignment;
	}
	
	public Vector<Integer> getNodeRealisAssignment()
	{
		return nodeRealisAssignment;
	}
	
	public SentenceAssignment(Alphabet nodeTypeTargetAlphabet, Alphabet nodeSubTypeTargetAlphabet, Alphabet nodeRealisTargetAlphabet, 
			Controller controller)
	{
		this.nodeTypeTargetAlphabet = nodeTypeTargetAlphabet;
		this.nodeSubTypeTargetAlphabet = nodeSubTypeTargetAlphabet;
		this.nodeRealisTargetAlphabet = nodeRealisTargetAlphabet;
		this.controller = controller;
		
		nodeTypeAssignment = new Vector<Integer>();
		nodeSubTypeAssignment = new Vector<Integer>();
		nodeRealisAssignment = new Vector<Integer>();
		nodeIdAssignment = new Vector<String>();
	}
	
	/**
	 * given an labeled instance, create a target assignment as ground-truth
	 * also create a full featureVectorSequence
	 * @param inst
	 */
	public SentenceAssignment(SentenceInstance inst, boolean singleToken)
	{
		this(inst.nodeTypeTargetAlphabet, inst.nodeSubTypeTargetAlphabet, inst.nodeRealisTargetAlphabet, inst.controller);
		
		for(int i=0; i < inst.size(); i++)
		{
			int type_index = this.nodeTypeTargetAlphabet.lookupIndex(Default_TYPE_Label);
			this.nodeTypeAssignment.add(type_index);
			
			int subtype_index = this.nodeSubTypeTargetAlphabet.lookupIndex(Default_SUBTYPE_Label);
			this.nodeSubTypeAssignment.add(subtype_index);
			
			int realis_index = this.nodeRealisTargetAlphabet.lookupIndex(Default_REALIS_Label);
			this.nodeRealisAssignment.add(realis_index);
			
			this.nodeIdAssignment.add(Default_Id_Label);
		}
		
		int type_index = -1, subtype_index = -1, realis_index = -1;
		for(AceEventMention mention : inst.eventMentions)
		{
			Vector<Integer> headIndices = mention.getHeadIndices();
			
			if (singleToken) {
				// for event, only pick up the first token as trigger
				int trigger_index = headIndices.get(0);
				// ignore the triggers that are with other POS
				if(!TypeConstraints.isPossibleTriggerByPOS(inst, trigger_index))
				{	
					continue;
				}
				
				type_index = this.nodeTypeTargetAlphabet.lookupIndex(mention.getType());
				this.nodeTypeAssignment.set(trigger_index, type_index);
				
				subtype_index = this.nodeSubTypeTargetAlphabet.lookupIndex(mention.getSubType());
				this.nodeSubTypeAssignment.set(trigger_index, subtype_index);
				
				realis_index = this.nodeRealisTargetAlphabet.lookupIndex(mention.getRealis());
				this.nodeRealisAssignment.set(trigger_index, realis_index);
				
				this.nodeIdAssignment.set(trigger_index, mention.getId());
			}
			else {
				int index = -1;
				
				for (Integer trigger_index : headIndices) {
					index++;
					
					String bioType = (index == 0) ? "B-" : "I-";
					bioType += mention.getType();
					type_index = this.nodeTypeTargetAlphabet.lookupIndex(bioType);
					this.nodeTypeAssignment.set(trigger_index, type_index);
					
					String bioSubType = (index == 0) ? "B-" : "I-";
					bioSubType += mention.getSubType();
					subtype_index = this.nodeSubTypeTargetAlphabet.lookupIndex(bioSubType);
					this.nodeSubTypeAssignment.set(trigger_index, subtype_index);
					
					String bioRealis = (index == 0) ? "B-" : "I-";
					bioRealis += mention.getRealis();
					realis_index = this.nodeRealisTargetAlphabet.lookupIndex(bioRealis);
					this.nodeRealisAssignment.set(trigger_index, realis_index);
					
					this.nodeIdAssignment.set(trigger_index, mention.getId());
				}
			}
		}
	}
	
}
