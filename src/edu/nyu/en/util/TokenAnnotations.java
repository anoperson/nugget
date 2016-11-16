package edu.nyu.en.util;

import java.util.Vector;

public class TokenAnnotations 
{
	/**
	 * this feature means the high confidence trigger types in
	 * the cluster of the current sentence
	 * @author XX
	 *
	 */

	public class SynonymsAnnotation
	{

	}

	public class NomlexbaseAnnotation
	{

	}

	public class ClauseAnnotation implements TokenAnnotation
	{
		@Override
		public Class<?> getType()
		{
			return Integer.class;
		}
	}
	
	public class EntityAnnotation implements TokenAnnotation
	{
		@Override
		public Class<?> getType()
		{
			return String.class;
		}
	}
	
	public class EntityAnnotation4NN implements TokenAnnotation
	{
		@Override
		public Class<?> getType()
		{
			return String.class;
		}
	}

  /**
   * it store hypernym's id of a given token based on WordNet
   * @author z0034d5z
   *
   */
  public class HypernymAnnotation implements TokenAnnotation {
	  public Class<String> getType() {
	      return String.class;
	    }
  }

  private TokenAnnotations() {
  } // only static members

  /**
   * the text of the token
   */
  public static class TextAnnotation implements TokenAnnotation {
    public Class<String> getType() {
      return String.class;
    }
  }


  /**
   * The CoreMap key for getting the lemma (morphological stem) of a token.
   *
   * This key is typically set on token annotations.
   *
   */
  public static class LemmaAnnotation implements TokenAnnotation {
    public Class<String> getType() {
      return String.class;
    }
  }

  /**
   * The CoreMap key for getting the Penn part of speech of a token.
   *
   * This key is typically set on token annotations.
   */
  public static class PartOfSpeechAnnotation implements TokenAnnotation {
    public Class<String> getType() {
      return String.class;
    }
  }

  /**
   * chunking annotation, BIO tags for chunking, for exmaple, B-NP means start of NP chunking
   */
  public static class ChunkingAnnotation implements TokenAnnotation {
    public Class<String> getType() {
      return String.class;
    }
  }
  
  /**
   *
   * the key for gold-standard label
   */
  public static class LabelAnnotation implements TokenAnnotation {
    public Class<String> getType() {
      return String.class;
    }
  }

  /**
   * determine if this word is intial capitalized 
   * @author che
   *
   */
  public static class CaseAnnotation implements TokenAnnotation{
	  public Class<String> getType() {
	      return String.class;
	    }
  }
  
  /**
   * determine a shape of a word, such like "Jintao" --> "Xxxxxx", "2011" --> "dddd"
   * @author che
   *
   */
  public static class ShapeAnnotation implements TokenAnnotation {
	  public Class<String> getType() {
	      return String.class;
	  }  
  }
  
  /**
   * determine if the token is first token or last token of a sentence
   * "head" for first token, "tail" for last token, "middle" for other
   * @author che
   *
   */
  public static class PositionAnnotation implements TokenAnnotation {
	    public Class<String> getType() {
	      return String.class;
	    }
  }
  
  public static class FrequencyAnnotation implements TokenAnnotation {
    public Class<String> getType() {
      return String.class;
    }
  }
  
  public static class BigramFrequencyAnnotation implements TokenAnnotation {
	    public Class<String> getType() {
	      return String.class;
	    }
  }
 
  /**
   * whether lowercase appeared in the document
   */
  public static class LowerCaseOccurrenceAnnotation implements TokenAnnotation {
	  public Class<String> getType() {
	      return String.class;
	    }
  }

  /**
   * whether lowercase appeared in the document
   */
  public static class UpperCaseOccurrenceAnnotation implements TokenAnnotation {
	  public Class<String> getType() {
	      return String.class;
	    }
  }
  
  public static class BrownClusterAnnotation implements TokenAnnotation {
	  public Class<?> getType() {
	      return Vector.class;
	    }
  }
  
  public static class DictionaryAnnotation implements TokenAnnotation {
	  public Class<?> getType() {
	      return Vector.class;
	    }
  }
  
  public static class DependencyAnnotation implements TokenAnnotation {
	  public Class<?> getType() {
	      return Vector.class;
	    }
  }
  
  public static class DependencyAnnotation4NN implements TokenAnnotation {
	  public Class<?> getType() {
	      return Vector.class;
	    }
  }
  
  public static class PrefixAnnotation implements TokenAnnotation {
	  public Class<?> getType() {
	      return Vector.class;
	    }
  }
  
  public static class SuffixAnnotation implements TokenAnnotation {
	  public Class<?> getType() {
	      return Vector.class;
	    }
  }
  
  public static class NormalWordAnnotation implements TokenAnnotation {
	  public Class<?> getType() {
	      return Vector.class;
	    }
  }
  
  public static class CharAnnotation implements TokenAnnotation{
	  public Class<?> getType() {
	      return Vector.class;
	    }
  }
  
  /**
   * The absolute character-based offset of the token
   * @author z0034d5z
   *
   */
  public static class SpanAnnotation implements TokenAnnotation
  {
	  public Class<?> getType() {
	      return String.class;
	  }
  }
}
