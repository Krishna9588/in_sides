"""
NLP utilities for Founder Intelligence System
"""
import re
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

from ..config.settings import settings


class NLPUtils:
    """NLP utilities with fallback for missing dependencies"""
    
    def __init__(self):
        self._sentence_model = None
        self._classification_pipeline = None
        self._nlp = None
        
        if NLP_AVAILABLE:
            self._load_models()
    
    def _load_models(self):
        """Load NLP models lazily"""
        try:
            # Load sentence transformer model
            self._sentence_model = SentenceTransformer(settings.SENTENCE_MODEL)
            print(f"Loaded sentence model: {settings.SENTENCE_MODEL}")
        except Exception as e:
            print(f"Failed to load sentence model: {e}")
        
        try:
            # Load classification pipeline
            self._classification_pipeline = pipeline(
                "text-classification",
                model=settings.CLASSIFICATION_MODEL
            )
            print(f"Loaded classification model: {settings.CLASSIFICATION_MODEL}")
        except Exception as e:
            print(f"Failed to load classification model: {e}")
        
        try:
            # Load spaCy model
            self._nlp = spacy.load("en_core_web_sm")
            print("Loaded spaCy model: en_core_web_sm")
        except Exception as e:
            print(f"Failed to load spaCy model: {e}")
    
    @property
    def sentence_model(self):
        """Get sentence transformer model"""
        if not self._sentence_model and NLP_AVAILABLE:
            self._load_models()
        return self._sentence_model
    
    @property
    def classification_pipeline(self):
        """Get classification pipeline"""
        if not self._classification_pipeline and NLP_AVAILABLE:
            self._load_models()
        return self._classification_pipeline
    
    @property
    def nlp(self):
        """Get spaCy NLP model"""
        if not self._nlp and NLP_AVAILABLE:
            self._load_models()
        return self._nlp
    
    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences to embeddings"""
        if not self.sentence_model:
            # Fallback: simple TF-IDF-like encoding
            return self._fallback_encode(sentences)
        
        try:
            embeddings = self.sentence_model.encode(
                sentences, 
                batch_size=32, 
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            print(f"Sentence encoding failed: {e}")
            return self._fallback_encode(sentences)
    
    def _fallback_encode(self, sentences: List[str]) -> np.ndarray:
        """Fallback encoding method"""
        # Simple word count encoding
        embeddings = []
        for sentence in sentences:
            words = sentence.lower().split()
            # Create simple feature vector (word count, char count, etc.)
            features = [
                len(words),  # word count
                len(sentence),  # char count
                sum(1 for c in sentence if c.isupper()),  # uppercase count
                sum(1 for c in sentence if c.isdigit()),  # digit count
            ]
            # Pad to 10 dimensions
            while len(features) < 10:
                features.append(0)
            embeddings.append(features[:10])
        
        return np.array(embeddings, dtype=np.float32)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except ImportError:
            # Fallback: simple dot product similarity
            return float(np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            ))
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        if not self.nlp:
            # Fallback: simple regex-based entity extraction
            return self._fallback_entity_extraction(text)
        
        try:
            doc = self.nlp(text)
            entities = {
                'persons': [],
                'organizations': [],
                'products': [],
                'locations': []
            }
            
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['persons'].append(ent.text)
                elif ent.label_ == 'ORG':
                    entities['organizations'].append(ent.text)
                elif ent.label_ == 'PRODUCT':
                    entities['products'].append(ent.text)
                elif ent.label_ == 'GPE':
                    entities['locations'].append(ent.text)
            
            return entities
        except Exception as e:
            print(f"Entity extraction failed: {e}")
            return self._fallback_entity_extraction(text)
    
    def _fallback_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """Fallback entity extraction using regex"""
        entities = {
            'persons': [],
            'organizations': [],
            'products': [],
            'locations': []
        }
        
        # Simple patterns (very basic)
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            entities['persons'].extend(emails)
        
        # URLs (organizations)
        urls = re.findall(r'https?://[^\s]+', text)
        if urls:
            entities['organizations'].extend(urls)
        
        return entities
    
    def classify_text(self, text: str) -> Dict[str, float]:
        """Classify text using local model"""
        if not self.classification_pipeline:
            # Fallback: keyword-based classification
            return self._fallback_classify(text)
        
        try:
            result = self.classification_pipeline(text)
            if isinstance(result, list) and len(result) > 0:
                return {item['label']: item['score'] for item in result}
            return {'UNKNOWN': 0.0}
        except Exception as e:
            print(f"Text classification failed: {e}")
            return self._fallback_classify(text)
    
    def _fallback_classify(self, text: str) -> Dict[str, float]:
        """Fallback classification using keywords"""
        text_lower = text.lower()
        
        # Problem indicators
        problem_words = ['problem', 'issue', 'bug', 'error', 'complaint', 'broken']
        problem_score = sum(1 for word in problem_words if word in text_lower) / len(problem_words)
        
        # Feature indicators
        feature_words = ['feature', 'functionality', 'capability', 'option', 'add']
        feature_score = sum(1 for word in feature_words if word in text_lower) / len(feature_words)
        
        # Positive indicators
        positive_words = ['good', 'great', 'excellent', 'love', 'amazing']
        positive_score = sum(1 for word in positive_words if word in text_lower) / len(positive_words)
        
        # Negative indicators
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'worst']
        negative_score = sum(1 for word in negative_words if word in text_lower) / len(negative_words)
        
        return {
            'problem': problem_score,
            'feature': feature_score,
            'positive': positive_score,
            'negative': negative_score
        }
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        if not self.nlp:
            # Fallback: simple word frequency
            return self._fallback_keywords(text, max_keywords)
        
        try:
            doc = self.nlp(text)
            
            # Extract noun chunks and filter
            keywords = []
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 2:  # Filter short chunks
                    keywords.append(chunk.text.lower())
            
            # Remove duplicates and limit
            unique_keywords = list(dict.fromkeys(keywords))[:max_keywords]
            return unique_keywords
        except Exception as e:
            print(f"Keyword extraction failed: {e}")
            return self._fallback_keywords(text, max_keywords)
    
    def _fallback_keywords(self, text: str, max_keywords: int) -> List[str]:
        """Fallback keyword extraction"""
        # Simple word frequency
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove HTML entities
        import html
        text = html.unescape(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', ' ', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\!]{2,}', '!', text)
        text = re.sub(r'[\?]{2,}', '?', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        if not text:
            return {'flesch_score': 0.0, 'reading_level': 'unknown'}
        
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if not sentences or not words:
            return {'flesch_score': 0.0, 'reading_level': 'unknown'}
        
        # Simple readability calculation
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Flesch reading ease (simplified)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 100))
        
        # Determine reading level
        if flesch_score >= 90:
            reading_level = 'very_easy'
        elif flesch_score >= 80:
            reading_level = 'easy'
        elif flesch_score >= 70:
            reading_level = 'fairly_easy'
        elif flesch_score >= 60:
            reading_level = 'standard'
        elif flesch_score >= 50:
            reading_level = 'fairly_difficult'
        elif flesch_score >= 30:
            reading_level = 'difficult'
        else:
            reading_level = 'very_difficult'
        
        return {
            'flesch_score': max(0, min(100, flesch_score)),
            'reading_level': reading_level,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length
        }


# Global NLP utilities instance
nlp_utils = NLPUtils()
