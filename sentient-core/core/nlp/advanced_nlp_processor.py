import asyncio
import re
import json
import spacy
import nltk
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Some NLP features will be limited.")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("Warning: textstat not available. Readability analysis will be limited.")

class IntentType(Enum):
    """Types of user intents."""
    QUESTION = "question"
    REQUEST = "request"
    COMMAND = "command"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    UNKNOWN = "unknown"

class EntityType(Enum):
    """Types of named entities."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    PERCENTAGE = "percentage"
    TECHNOLOGY = "technology"
    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    TOOL = "tool"
    FILE_PATH = "file_path"
    URL = "url"
    EMAIL = "email"
    PHONE = "phone"
    CUSTOM = "custom"

class SentimentPolarity(Enum):
    """Sentiment polarity levels."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

class ComplexityLevel(Enum):
    """Text complexity levels."""
    VERY_SIMPLE = "very_simple"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

@dataclass
class Entity:
    """Represents a named entity."""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Intent:
    """Represents user intent."""
    intent_type: IntentType
    confidence: float
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Sentiment:
    """Represents sentiment analysis result."""
    polarity: SentimentPolarity
    score: float  # -1.0 to 1.0
    confidence: float
    emotions: Dict[str, float] = field(default_factory=dict)

@dataclass
class TextComplexity:
    """Represents text complexity analysis."""
    level: ComplexityLevel
    readability_score: float
    sentence_count: int
    word_count: int
    avg_sentence_length: float
    difficult_words_ratio: float
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class KeyPhrase:
    """Represents an extracted key phrase."""
    phrase: str
    importance_score: float
    frequency: int
    context: str = ""

@dataclass
class SemanticSimilarity:
    """Represents semantic similarity between texts."""
    similarity_score: float
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NLPAnalysisResult:
    """Comprehensive NLP analysis result."""
    text: str
    timestamp: datetime
    entities: List[Entity] = field(default_factory=list)
    intent: Optional[Intent] = None
    sentiment: Optional[Sentiment] = None
    complexity: Optional[TextComplexity] = None
    key_phrases: List[KeyPhrase] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)

class EntityExtractor:
    """Advanced entity extraction with custom patterns."""
    
    def __init__(self):
        self.nlp = None
        self.custom_patterns = {
            EntityType.PROGRAMMING_LANGUAGE: [
                r'\b(Python|JavaScript|Java|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|TypeScript|PHP|Scala|R|MATLAB|Perl|Haskell|Clojure|Erlang|Elixir|F#|VB\.NET|Objective-C|Dart|Julia|Lua|Shell|Bash|PowerShell)\b',
            ],
            EntityType.FRAMEWORK: [
                r'\b(React|Angular|Vue|Django|Flask|FastAPI|Express|Spring|Laravel|Rails|ASP\.NET|Symfony|CodeIgniter|CakePHP|Zend|Yii|Pyramid|Tornado|Bottle|Falcon|Sanic|Quart|Starlette|Responder|Molten|Hug|Eve|Connexion|Chalice|Zappa|Serverless)\b',
                r'\b(TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Matplotlib|Seaborn|Plotly|Bokeh|Altair|Streamlit|Dash|Jupyter|IPython|Anaconda|Conda|Pip|Pipenv|Poetry|Virtualenv|Venv|Docker|Kubernetes|Terraform|Ansible|Chef|Puppet|Vagrant|Jenkins|GitLab|GitHub|Bitbucket|CircleCI|Travis|Azure|AWS|GCP|Heroku|Netlify|Vercel)\b'
            ],
            EntityType.TOOL: [
                r'\b(Git|SVN|Mercurial|Bazaar|CVS|Perforce|TFS|SourceSafe|ClearCase|AccuRev|Fossil|Darcs|Monotone|BitKeeper|SCCS|RCS|Arch|GNU|Arch|Cogito|StGit|Quilt|MQ|TopGit|Guilt|Stacked|Git|Flow|Git|Kraken|SourceTree|GitKraken|Tower|Fork|GitUp|Gitbox|Working|Copy|Cornerstone|Versions|SmartGit|GitEye|EGit|JGit|libgit2|dulwich|pygit2|nodegit|isomorphic-git|simple-git|git-js|git-node|git-promise|git-exec|git-cli|git-wrapper|git-utils|git-tools|git-extras|git-flow|git-lfs|git-annex|git-subtree|git-worktree|git-submodule|git-subrepo|git-slave|git-deps|git-when-merged|git-what-branch|git-recent|git-effort|git-line-summary|git-summary|git-count|git-delete-merged-branches|git-fresh|git-ignore|git-info|git-release|git-changelog|git-authors|git-rank-contributors|git-repl|git-undo|git-setup|git-touch|git-obliterate|git-feature|git-refactor|git-bug|git-promote|git-local-commits|git-archive-file|git-missing|git-lock|git-unlock|git-reset-file|git-pr|git-delta|git-show-tree|git-show-merged-branches|git-show-unmerged-branches|git-show-tree|git-alias|git-browse|git-bulk|git-churn|git-clear|git-coauthor|git-copy-branch-name|git-create-branch|git-delete-branch|git-delete-merged|git-delete-squashed|git-delete-tag|git-force-clone|git-fork|git-fresh|git-get|git-guilt|git-history|git-ignore-io|git-improved-merge|git-ink|git-integrate|git-interactive-rebase-tool|git-imerge|git-machete|git-merge-into|git-move-commits|git-now|git-open|git-plus|git-pr-release|git-publish|git-quick-stats|git-recent|git-remote-hg|git-rename-branch|git-repo|git-restore-mtime|git-revise|git-run|git-scp|git-secret|git-semver|git-series|git-sh|git-standup|git-stats|git-sweep|git-sync|git-town|git-trim|git-up|git-url|git-vendor|git-when-merged|git-where|git-wtf)\b'
            ],
            EntityType.FILE_PATH: [
                r'[a-zA-Z]:\\[^\s<>:"|?*]+',  # Windows paths
                r'/[^\s<>:"|?*]+',  # Unix paths
                r'\./[^\s<>:"|?*]+',  # Relative paths
                r'\.\.?/[^\s<>:"|?*]+',  # Parent directory paths
            ],
            EntityType.URL: [
                r'https?://[^\s<>"]+',
                r'www\.[^\s<>"]+',
                r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>"]*)?'
            ],
            EntityType.EMAIL: [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            ]
        }
        
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP model."""
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy English model loaded")
        except OSError:
            try:
                # Fallback to basic English model
                self.nlp = spacy.load("en")
                print("✓ spaCy basic English model loaded")
            except OSError:
                print("✗ spaCy model not found. Please install: python -m spacy download en_core_web_sm")
                self.nlp = None
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using both spaCy and custom patterns."""
        entities = []
        
        # spaCy entity extraction
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                entities.append(Entity(
                    text=ent.text,
                    entity_type=entity_type,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.8,  # Default confidence for spaCy
                    metadata={"spacy_label": ent.label_}
                ))
        
        # Custom pattern extraction
        for entity_type, patterns in self.custom_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Check for overlap with existing entities
                    overlaps = any(
                        (match.start() < e.end_pos and match.end() > e.start_pos)
                        for e in entities
                    )
                    
                    if not overlaps:
                        entities.append(Entity(
                            text=match.group(),
                            entity_type=entity_type,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=0.9,  # High confidence for pattern matches
                            metadata={"pattern": pattern}
                        ))
        
        return sorted(entities, key=lambda e: e.start_pos)
    
    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy entity labels to our EntityType enum."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENTAGE,
        }
        return mapping.get(label, EntityType.CUSTOM)

class IntentClassifier:
    """Intent classification using rule-based and ML approaches."""
    
    def __init__(self):
        self.intent_patterns = {
            IntentType.QUESTION: [
                r'\b(what|how|why|when|where|who|which|can you|could you|would you|do you|does|is|are|will|should)\b',
                r'\?$'
            ],
            IntentType.REQUEST: [
                r'\b(please|could you|would you|can you|help me|assist me|I need|I want|I would like)\b',
                r'\b(create|make|build|generate|develop|implement|add|include)\b'
            ],
            IntentType.COMMAND: [
                r'^(run|execute|start|stop|restart|install|update|delete|remove|configure|setup)\b',
                r'\b(do this|go ahead|proceed|continue|next)\b'
            ],
            IntentType.CLARIFICATION: [
                r'\b(what do you mean|I don\'t understand|can you explain|clarify|elaborate)\b',
                r'\b(confused|unclear|not sure|uncertain)\b'
            ],
            IntentType.FEEDBACK: [
                r'\b(good|great|excellent|perfect|amazing|wonderful|fantastic|awesome|brilliant)\b',
                r'\b(bad|terrible|awful|horrible|wrong|incorrect|error|mistake|issue|problem)\b'
            ],
            IntentType.GREETING: [
                r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b'
            ],
            IntentType.GOODBYE: [
                r'\b(bye|goodbye|see you|farewell|take care|until next time)\b'
            ]
        }
        
        # Initialize transformer-based classifier if available
        self.transformer_classifier = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.transformer_classifier = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",
                    return_all_scores=True
                )
                print("✓ Transformer-based intent classifier loaded")
            except Exception as e:
                print(f"✗ Failed to load transformer classifier: {e}")
    
    def classify_intent(self, text: str) -> Intent:
        """Classify the intent of the given text."""
        text_lower = text.lower().strip()
        
        # Rule-based classification
        best_intent = IntentType.UNKNOWN
        best_score = 0.0
        matched_keywords = []
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            keywords = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    score += len(matches) * 0.3
                    keywords.extend(matches)
            
            if score > best_score:
                best_score = score
                best_intent = intent_type
                matched_keywords = keywords
        
        # Boost confidence if transformer is available
        confidence = min(best_score, 1.0)
        if self.transformer_classifier and confidence < 0.7:
            try:
                transformer_result = self.transformer_classifier(text)
                # This is a simplified integration - in practice, you'd need
                # a model specifically trained for intent classification
                confidence = max(confidence, 0.6)
            except Exception as e:
                print(f"Transformer classification failed: {e}")
        
        return Intent(
            intent_type=best_intent,
            confidence=confidence,
            keywords=matched_keywords,
            metadata={"method": "rule_based"}
        )

class SentimentAnalyzer:
    """Advanced sentiment analysis with emotion detection."""
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.emotion_pipeline = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                print("✓ Sentiment analysis pipeline loaded")
            except Exception as e:
                print(f"✗ Failed to load sentiment pipeline: {e}")
            
            try:
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
                print("✓ Emotion analysis pipeline loaded")
            except Exception as e:
                print(f"✗ Failed to load emotion pipeline: {e}")
    
    def analyze_sentiment(self, text: str) -> Sentiment:
        """Analyze sentiment and emotions in text."""
        # Fallback rule-based sentiment analysis
        polarity_score = self._rule_based_sentiment(text)
        
        emotions = {}
        confidence = 0.5
        
        # Use transformer models if available
        if self.sentiment_pipeline:
            try:
                sentiment_results = self.sentiment_pipeline(text)
                # Map results to our polarity scale
                for result in sentiment_results:
                    if result['label'] == 'POSITIVE':
                        polarity_score = result['score']
                    elif result['label'] == 'NEGATIVE':
                        polarity_score = -result['score']
                    confidence = max(confidence, result['score'])
            except Exception as e:
                print(f"Sentiment analysis failed: {e}")
        
        if self.emotion_pipeline:
            try:
                emotion_results = self.emotion_pipeline(text)
                emotions = {result['label']: result['score'] for result in emotion_results}
            except Exception as e:
                print(f"Emotion analysis failed: {e}")
        
        # Determine polarity category
        if polarity_score >= 0.6:
            polarity = SentimentPolarity.VERY_POSITIVE
        elif polarity_score >= 0.2:
            polarity = SentimentPolarity.POSITIVE
        elif polarity_score >= -0.2:
            polarity = SentimentPolarity.NEUTRAL
        elif polarity_score >= -0.6:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.VERY_NEGATIVE
        
        return Sentiment(
            polarity=polarity,
            score=polarity_score,
            confidence=confidence,
            emotions=emotions
        )
    
    def _rule_based_sentiment(self, text: str) -> float:
        """Simple rule-based sentiment analysis as fallback."""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'perfect', 'outstanding', 'superb', 'magnificent',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'delighted'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'dislike', 'angry', 'frustrated', 'disappointed', 'sad', 'upset',
            'annoyed', 'irritated', 'furious', 'outraged', 'disgusted'
        }
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_words

class TextComplexityAnalyzer:
    """Analyze text complexity and readability."""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def analyze_complexity(self, text: str) -> TextComplexity:
        """Analyze text complexity using multiple metrics."""
        # Basic metrics
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        sentence_count = len(sentences)
        word_count = len(words)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Calculate difficult words ratio (words with 3+ syllables)
        difficult_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        difficult_words_ratio = difficult_words / word_count if word_count > 0 else 0
        
        # Readability scores
        metrics = {}
        readability_score = 0.0
        
        if TEXTSTAT_AVAILABLE:
            try:
                metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
                metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
                metrics['gunning_fog'] = textstat.gunning_fog(text)
                metrics['automated_readability_index'] = textstat.automated_readability_index(text)
                metrics['coleman_liau_index'] = textstat.coleman_liau_index(text)
                
                # Use Flesch Reading Ease as primary readability score
                readability_score = metrics['flesch_reading_ease']
            except Exception as e:
                print(f"Readability analysis failed: {e}")
        else:
            # Simple readability approximation
            readability_score = max(0, 100 - (avg_sentence_length * 2) - (difficult_words_ratio * 100))
            metrics['simple_readability'] = readability_score
        
        # Determine complexity level
        if readability_score >= 80:
            level = ComplexityLevel.VERY_SIMPLE
        elif readability_score >= 60:
            level = ComplexityLevel.SIMPLE
        elif readability_score >= 40:
            level = ComplexityLevel.MODERATE
        elif readability_score >= 20:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.VERY_COMPLEX
        
        return TextComplexity(
            level=level,
            readability_score=readability_score,
            sentence_count=sentence_count,
            word_count=word_count,
            avg_sentence_length=avg_sentence_length,
            difficult_words_ratio=difficult_words_ratio,
            metrics=metrics
        )
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        # Every word has at least one syllable
        return max(1, syllable_count)

class KeyPhraseExtractor:
    """Extract key phrases and topics from text."""
    
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        # Add technical stop words
        self.stop_words.update({
            'use', 'using', 'used', 'make', 'making', 'made', 'get', 'getting',
            'got', 'take', 'taking', 'took', 'give', 'giving', 'gave', 'put',
            'putting', 'go', 'going', 'went', 'come', 'coming', 'came', 'see',
            'seeing', 'saw', 'know', 'knowing', 'knew', 'think', 'thinking',
            'thought', 'want', 'wanting', 'wanted', 'need', 'needing', 'needed'
        })
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[KeyPhrase]:
        """Extract key phrases using TF-IDF and n-gram analysis."""
        # Tokenize and clean text
        words = nltk.word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Generate n-grams (1-3 words)
        phrases = []
        
        # Unigrams
        phrases.extend(words)
        
        # Bigrams
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        
        # Trigrams
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # Count phrase frequencies
        phrase_counts = Counter(phrases)
        
        # Calculate importance scores (simple TF-based)
        total_phrases = len(phrases)
        key_phrases = []
        
        for phrase, count in phrase_counts.most_common(max_phrases * 2):
            importance_score = count / total_phrases
            
            # Boost score for longer phrases
            word_count = len(phrase.split())
            importance_score *= (1 + (word_count - 1) * 0.2)
            
            # Boost score for technical terms
            if any(tech_word in phrase for tech_word in ['api', 'database', 'server', 'client', 'framework', 'library', 'algorithm', 'function', 'method', 'class', 'object', 'variable', 'parameter', 'return', 'import', 'export', 'module', 'package', 'dependency', 'configuration', 'deployment', 'testing', 'debugging', 'optimization', 'performance', 'security', 'authentication', 'authorization', 'encryption', 'decryption', 'validation', 'verification', 'integration', 'implementation', 'development', 'production', 'staging', 'environment']):
                importance_score *= 1.5
            
            key_phrases.append(KeyPhrase(
                phrase=phrase,
                importance_score=importance_score,
                frequency=count
            ))
        
        # Sort by importance and return top phrases
        key_phrases.sort(key=lambda kp: kp.importance_score, reverse=True)
        return key_phrases[:max_phrases]

class SemanticSimilarityCalculator:
    """Calculate semantic similarity between texts."""
    
    def __init__(self):
        self.sentence_transformer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ Sentence transformer model loaded")
            except Exception as e:
                print(f"✗ Failed to load sentence transformer: {e}")
    
    def calculate_similarity(self, text1: str, text2: str) -> SemanticSimilarity:
        """Calculate semantic similarity between two texts."""
        if self.sentence_transformer:
            try:
                # Use sentence transformer for semantic similarity
                embeddings = self.sentence_transformer.encode([text1, text2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                
                return SemanticSimilarity(
                    similarity_score=float(similarity),
                    method="sentence_transformer",
                    metadata={"model": "all-MiniLM-L6-v2"}
                )
            except Exception as e:
                print(f"Semantic similarity calculation failed: {e}")
        
        # Fallback to simple word overlap
        words1 = set(nltk.word_tokenize(text1.lower()))
        words2 = set(nltk.word_tokenize(text2.lower()))
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        return SemanticSimilarity(
            similarity_score=jaccard_similarity,
            method="jaccard",
            metadata={"intersection_size": len(intersection), "union_size": len(union)}
        )

class AdvancedNLPProcessor:
    """
    Advanced NLP processor that combines multiple NLP capabilities:
    - Entity extraction
    - Intent classification
    - Sentiment analysis
    - Text complexity analysis
    - Key phrase extraction
    - Semantic similarity
    - Topic modeling
    """
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.intent_classifier = IntentClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.complexity_analyzer = TextComplexityAnalyzer()
        self.keyphrase_extractor = KeyPhraseExtractor()
        self.similarity_calculator = SemanticSimilarityCalculator()
        
        # Cache for performance
        self.analysis_cache: Dict[str, NLPAnalysisResult] = {}
        self.cache_max_size = 1000
        
        print("Advanced NLP Processor initialized")
    
    async def analyze_text(self, text: str, 
                          include_entities: bool = True,
                          include_intent: bool = True,
                          include_sentiment: bool = True,
                          include_complexity: bool = True,
                          include_keyphrases: bool = True) -> NLPAnalysisResult:
        """Perform comprehensive NLP analysis on text."""
        # Check cache first
        cache_key = f"{hash(text)}_{include_entities}_{include_intent}_{include_sentiment}_{include_complexity}_{include_keyphrases}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        start_time = datetime.now()
        
        # Initialize result
        result = NLPAnalysisResult(
            text=text,
            timestamp=start_time
        )
        
        # Detect language (simple heuristic)
        result.language = self._detect_language(text)
        
        # Perform analysis components
        if include_entities:
            result.entities = self.entity_extractor.extract_entities(text)
        
        if include_intent:
            result.intent = self.intent_classifier.classify_intent(text)
        
        if include_sentiment:
            result.sentiment = self.sentiment_analyzer.analyze_sentiment(text)
        
        if include_complexity:
            result.complexity = self.complexity_analyzer.analyze_complexity(text)
        
        if include_keyphrases:
            result.key_phrases = self.keyphrase_extractor.extract_key_phrases(text)
        
        # Extract topics from key phrases
        if result.key_phrases:
            result.topics = [kp.phrase for kp in result.key_phrases[:5]]
        
        # Add processing time to metadata
        processing_time = (datetime.now() - start_time).total_seconds()
        result.metadata['processing_time'] = processing_time
        
        # Cache result
        if len(self.analysis_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.analysis_cache))
            del self.analysis_cache[oldest_key]
        
        self.analysis_cache[cache_key] = result
        
        return result
    
    async def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare two texts across multiple dimensions."""
        # Analyze both texts
        analysis1 = await self.analyze_text(text1)
        analysis2 = await self.analyze_text(text2)
        
        # Calculate semantic similarity
        similarity = self.similarity_calculator.calculate_similarity(text1, text2)
        
        # Compare entities
        entities1 = {e.text.lower() for e in analysis1.entities}
        entities2 = {e.text.lower() for e in analysis2.entities}
        entity_overlap = len(entities1.intersection(entities2)) / len(entities1.union(entities2)) if entities1.union(entities2) else 0
        
        # Compare sentiments
        sentiment_similarity = 1.0 - abs(analysis1.sentiment.score - analysis2.sentiment.score) / 2.0 if analysis1.sentiment and analysis2.sentiment else 0
        
        # Compare complexity
        complexity_similarity = 1.0 - abs(analysis1.complexity.readability_score - analysis2.complexity.readability_score) / 100.0 if analysis1.complexity and analysis2.complexity else 0
        
        return {
            "semantic_similarity": similarity,
            "entity_overlap": entity_overlap,
            "sentiment_similarity": sentiment_similarity,
            "complexity_similarity": complexity_similarity,
            "analysis1": analysis1,
            "analysis2": analysis2
        }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (placeholder for more sophisticated detection)."""
        # This is a very basic heuristic - in practice, you'd use a proper language detection library
        english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = text.lower().split()
        english_count = sum(1 for word in words if word in english_indicators)
        
        if english_count / len(words) > 0.1 if words else False:
            return "en"
        else:
            return "unknown"
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get NLP processor analytics."""
        return {
            "cache_size": len(self.analysis_cache),
            "cache_max_size": self.cache_max_size,
            "components_loaded": {
                "spacy": self.entity_extractor.nlp is not None,
                "transformers": TRANSFORMERS_AVAILABLE,
                "textstat": TEXTSTAT_AVAILABLE,
                "sentence_transformer": self.similarity_calculator.sentence_transformer is not None
            }
        }

# Global instance
_nlp_processor = None

def get_nlp_processor() -> AdvancedNLPProcessor:
    """Get the global NLP processor instance."""
    global _nlp_processor
    if _nlp_processor is None:
        _nlp_processor = AdvancedNLPProcessor()
    return _nlp_processor

def initialize_nlp_processor() -> AdvancedNLPProcessor:
    """Initialize and return the NLP processor."""
    global _nlp_processor
    _nlp_processor = AdvancedNLPProcessor()
    return _nlp_processor