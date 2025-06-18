#!/usr/bin/env python3
"""
Smart Text Categorizer - AI Challenge Solution
Classifies user messages into high-level categories using LLM + fallback rules
"""

import json
import re
import argparse
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Set tokenizers parallelism to false to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import OpenAI with fallback
try:
    import openai
    # Try new OpenAI client first
    try:
        from openai import OpenAI
        OPENAI_NEW_CLIENT = True
    except ImportError:
        # Fall back to older API
        OPENAI_NEW_CLIENT = False
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OPENAI_NEW_CLIENT = False

# Try to import sentence-transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

@dataclass
class CategoryResult:
    """Result of text categorization"""
    message: str
    category: str
    confidence: float
    explanation: str
    method: str  # 'llm', 'rule', 'fallback'

class TextCategorizer:
    """Smart text categorizer using LLM + rule-based fallbacks"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize categorizer with OpenAI API and sentence transformers"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None
        self.sentence_model = None
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE and self.api_key:
            if OPENAI_NEW_CLIENT:
                self.client = OpenAI(api_key=self.api_key)
            else:
                # For older OpenAI versions
                openai.api_key = self.api_key
                self.client = "legacy"
        
        # Default taxonomy - can be overridden
        self.default_categories = {
            "account_issue": "Problems with user accounts, login, passwords, access",
            "billing": "Financial concerns, payments, charges, refunds, pricing",
            "recommendation_request": "Asking for suggestions, recommendations, advice",
            "feedback": "User opinions, reviews, complaints, praise",
            "technical_support": "Technical problems, bugs, errors, troubleshooting",
            "product_inquiry": "Questions about features, functionality, capabilities",
            "general_inquiry": "General questions, information requests",
            "other": "Messages that don't fit other categories"
        }
        
        # Don't load sentence transformer immediately - load lazily when needed
        # This saves memory and startup time when LLM is available
        self.sentence_model = None
        self._sentence_model_attempted = False
        
        # Rule-based patterns for fallback
        self.rule_patterns = {
            "account_issue": [
                r"\b(login|password|account|access|locked|suspended|verify|authenticate)\b",
                r"\b(can't (log|sign) in|lost access|forgot password)\b",
                r"\b(account (locked|suspended|disabled))\b"
            ],
            "billing": [
                r"\b(charge|charged|bill|billing|payment|refund|money|cost|price|invoice)\b",
                r"\b(why was i charged|double charge|unexpected charge)\b",
                r"\b(cancel subscription|billing issue)\b"
            ],
            "recommendation_request": [
                r"\b(recommend|suggest|advice|which|what.*should|best.*for)\b",
                r"\b(can you recommend|any suggestions|what do you think)\b"
            ],
            "feedback": [
                r"\b(great|terrible|awesome|horrible|love|hate|disappointed|satisfied)\b",
                r"\b(smooth|easy|difficult|frustrating|amazing|brilliant)\b",
                r"\b(feedback|review|opinion)\b"
            ],
            "technical_support": [
                r"\b(error|bug|broken|not working|doesn't work|problem|issue|crash|crashing|crashes)\b",
                r"\b(help.*fix|troubleshoot|support|keeps (crash|fail))\b"
            ],
            "product_inquiry": [
                r"\b(features?|functionality|capabilities|plan|premium|what.*include)\b",
                r"\b(how (does|do)|what is|tell me about)\b"
            ]
        }
    
    def _load_sentence_transformer(self):
        """Lazy load sentence transformer only when needed"""
        if self._sentence_model_attempted:
            return self.sentence_model is not None
        
        self._sentence_model_attempted = True
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("⚠️  Sentence transformers not available, falling back to rule-based classification")
            return False
        
        try:
            # Check if model is already cached
            cache_dir = os.path.expanduser('~/.cache/sentence_transformers')
            model_cache_path = os.path.join(cache_dir, 'models--sentence-transformers--all-MiniLM-L6-v2')
            
            if os.path.exists(model_cache_path):
                print("🔄 LLM failed, loading cached sentence transformer model...")
            else:
                print("🔄 LLM failed, loading sentence transformer model...")
                print("📥 Downloading model on first run (~90MB). This may take a few minutes...")
                print("💡 Tip: Run 'python3 setup_models.py' to pre-download for faster startup")
            
            # Use cache_folder to ensure proper model caching
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', 
                                                     cache_folder=cache_dir)
            self._prepare_category_embeddings()
            print("✅ Sentence transformer model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load sentence transformer: {e}")
            print("🔄 Continuing with rule-based classification only...")
            self.sentence_model = None
            return False

    def _prepare_category_embeddings(self):
        """Pre-compute embeddings for category descriptions with examples"""
        if not self.sentence_model:
            return
        
        # Enhanced category descriptions with examples for better embeddings
        enhanced_categories = {
            "account_issue": [
                "Problems with user accounts, login, passwords, access",
                "I can't log in to my account",
                "I forgot my password and need to reset it",
                "My account is locked or suspended",
                "I lost access to my account",
                "Login problems and authentication issues"
            ],
            "billing": [
                "Financial concerns, payments, charges, refunds, pricing questions",
                "Why was I charged twice for this service?",
                "I need a refund for my subscription",
                "How much does the premium plan cost?",
                "There's an unexpected charge on my card",
                "Billing issues and payment problems"
            ],
            "recommendation_request": [
                "Asking for suggestions, recommendations, advice",
                "Can you recommend some good books about AI?",
                "What's the best way to learn programming?",
                "Could you suggest a better alternative?",
                "Which option would you recommend?",
                "Looking for advice and suggestions"
            ],
            "feedback": [
                "User opinions, reviews, complaints, praise, satisfaction",
                "This service is absolutely amazing!",
                "I'm really disappointed with the quality",
                "The onboarding process was super smooth",
                "This is ridiculous and frustrating",
                "Customer feedback and user experience reviews"
            ],
            "technical_support": [
                "Technical problems, bugs, errors, troubleshooting",
                "The app keeps crashing when I upload files",
                "I'm getting an error message that won't go away",
                "The software is not working properly",
                "Need help fixing technical issues",
                "Bug reports and technical problems"
            ],
            "product_inquiry": [
                "Questions about features, functionality, capabilities",
                "What features are included in the premium plan?",
                "How does the AI recommendation system work?",
                "Can this software integrate with other tools?",
                "What are the system requirements?",
                "Product questions and feature inquiries"
            ],
            "general_inquiry": [
                "General questions, information requests, how-to questions",
                "How do I get started with this platform?",
                "Where can I find the user manual?",
                "What are your business hours?",
                "How long does processing usually take?",
                "General questions and information requests"
            ],
            "other": [
                "Messages that don't fit other categories, unclear requests",
                "Random text that doesn't make sense",
                "Unclear or ambiguous messages",
                "Off-topic discussions",
                "Unrelated content"
            ]
        }
        
        self.category_embeddings = {}
        for category, texts in enhanced_categories.items():
            # Create multiple embeddings for each category and average them
            embeddings = self.sentence_model.encode(texts)
            # Use the average of all example embeddings for better representation
            avg_embedding = embeddings.mean(axis=0)
            self.category_embeddings[category] = avg_embedding
    
    def categorize_batch(self, messages: List[str], custom_categories: Optional[Dict[str, str]] = None) -> List[CategoryResult]:
        """Categorize a batch of messages"""
        categories = custom_categories or self.default_categories
        
        # If using custom categories and sentence transformers, update embeddings
        if custom_categories and self.sentence_model:
            self._update_category_embeddings(custom_categories)
        
        results = []
        for message in messages:
            result = self._categorize_single(message, categories)
            results.append(result)
        
        return results
    
    def _update_category_embeddings(self, categories: Dict[str, str]):
        """Update category embeddings for custom categories"""
        if not self.sentence_model:
            return
        
        self.category_embeddings = {}
        for category, description in categories.items():
            # For custom categories, create multiple variations for better embeddings
            category_texts = [
                description,
                f"{category.replace('_', ' ')}: {description}",
                f"This is about {category.replace('_', ' ')}",
                f"User message related to {description.lower()}",
                f"Category: {category.replace('_', ' ')}"
            ]
            
            # Create multiple embeddings and average them
            embeddings = self.sentence_model.encode(category_texts)
            avg_embedding = embeddings.mean(axis=0)
            self.category_embeddings[category] = avg_embedding
    
    def _categorize_single(self, message: str, categories: Dict[str, str]) -> CategoryResult:
        """Categorize a single message"""
        # Check for ambiguous messages first
        ambiguous_result = self.handle_ambiguous_message(message)
        if ambiguous_result is not None:
            return ambiguous_result
        
        # First try LLM if available
        if self.client:
            try:
                llm_result = self._categorize_with_llm(message, categories)
                if llm_result:
                    return llm_result
            except Exception as e:
                print(f"LLM failed: {e}, falling back to sentence transformers")
        
        # Second, try sentence transformers if available (lazy load)
        if self._load_sentence_transformer():
            try:
                st_result = self._categorize_with_sentence_transformers(message, categories)
                if st_result:
                    return st_result
            except Exception as e:
                print(f"Sentence transformers failed: {e}, falling back to rules")
        
        # Final fallback to rule-based classification
        return self._categorize_with_rules(message, categories)
    
    def _categorize_with_llm(self, message: str, categories: Dict[str, str]) -> Optional[CategoryResult]:
        """Categorize using LLM"""
        if not OPENAI_AVAILABLE:
            return None
            
        category_list = "\n".join([f"- {cat}: {desc}" for cat, desc in categories.items()])
        
        prompt = f"""
Classify the following user message into one of these categories:

{category_list}

Message: "{message}"

Return ONLY a JSON object with this exact format:
{{
    "category": "category_name",
    "confidence": 0.85,
    "explanation": "Brief explanation why this category was chosen"
}}

The confidence should be between 0.0 and 1.0. Be conservative with confidence scores.
"""
        
        try:
            if OPENAI_NEW_CLIENT and self.client:
                # New OpenAI client
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150
                )
                content = response.choices[0].message.content.strip()
            elif self.client == "legacy":
                # Legacy OpenAI API
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150
                )
                content = response.choices[0].message.content.strip()
            else:
                return None
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                
                # Validate category exists
                category = result_data.get('category', 'other')
                if category not in categories:
                    category = 'other'
                
                return CategoryResult(
                    message=message,
                    category=category,
                    confidence=min(1.0, max(0.0, result_data.get('confidence', 0.5))),
                    explanation=result_data.get('explanation', 'LLM classification'),
                    method='llm'
                )
        except Exception as e:
            print(f"LLM categorization failed: {e}")
            return None
    
    def _categorize_with_sentence_transformers(self, message: str, categories: Dict[str, str]) -> Optional[CategoryResult]:
        """Categorize using sentence transformers with improved confidence scoring"""
        if not self.sentence_model or not hasattr(self, 'category_embeddings'):
            return None
        
        try:
            # Encode the message
            message_embedding = self.sentence_model.encode([message])
            
            # Calculate similarities with all categories
            best_category = 'other'
            best_similarity = 0.0
            similarities = {}
            
            for category, category_embedding in self.category_embeddings.items():
                if category in categories:
                    similarity = cosine_similarity(
                        message_embedding.reshape(1, -1),
                        category_embedding.reshape(1, -1)
                    )[0][0]
                    similarities[category] = similarity
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_category = category
            
            # Improved confidence calculation with dynamic thresholds
            if best_similarity < 0.15:
                confidence = 0.2
                best_category = 'other'
                explanation = "Low semantic similarity to all categories, defaulting to 'other'"
            elif best_similarity < 0.25:
                # Low similarity - be more conservative
                confidence = 0.3 + (best_similarity - 0.15) * 2.0
                confidence = min(0.5, confidence)
                top_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:2]
                similarity_text = ", ".join([f"{cat}: {sim:.2f}" for cat, sim in top_similarities])
                explanation = f"Low semantic similarity classification. Similarities: {similarity_text}"
            else:
                # Good similarity - more confident scoring
                # Map similarity (0.25-1.0) to confidence (0.5-0.9)
                confidence = 0.5 + (best_similarity - 0.25) * 0.53
                confidence = min(0.9, confidence)
                
                # Show top similarities for explanation
                top_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:2]
                similarity_text = ", ".join([f"{cat}: {sim:.2f}" for cat, sim in top_similarities])
                explanation = f"Semantic similarity classification. Similarities: {similarity_text}"
            
            # Additional boost for very high similarities
            if best_similarity > 0.7:
                confidence = min(0.95, confidence + 0.1)
                explanation = f"High semantic similarity classification. Similarities: {similarity_text}"
            
            return CategoryResult(
                message=message,
                category=best_category,
                confidence=confidence,
                explanation=explanation,
                method='sentence_transformer'
            )
            
        except Exception as e:
            print(f"Sentence transformer categorization failed: {e}")
            return None
    
    def _categorize_with_rules(self, message: str, categories: Dict[str, str]) -> CategoryResult:
        """Categorize using rule-based patterns"""
        message_lower = message.lower()
        best_category = 'other'
        best_score = 0
        matched_patterns = []
        
        for category, patterns in self.rule_patterns.items():
            if category in categories:
                score = 0
                category_matches = []
                
                for pattern in patterns:
                    matches = re.findall(pattern, message_lower, re.IGNORECASE)
                    if matches:
                        score += len(matches)
                        # Flatten tuple matches to strings
                        for match in matches:
                            if isinstance(match, tuple):
                                category_matches.extend([m for m in match if m])
                            else:
                                category_matches.append(match)
                
                if score > best_score:
                    best_score = score
                    best_category = category
                    matched_patterns = category_matches
        
        # Calculate confidence based on pattern matches
        confidence = min(0.8, 0.3 + (best_score * 0.2)) if best_score > 0 else 0.1
        
        explanation = f"Rule-based classification"
        if matched_patterns:
            explanation += f" based on keywords: {', '.join(set(matched_patterns[:3]))}"
        else:
            explanation += " - no strong patterns found, defaulting to 'other'"
        
        return CategoryResult(
            message=message,
            category=best_category,
            confidence=confidence,
            explanation=explanation,
            method='rule'
        )
    
    def handle_ambiguous_message(self, message: str) -> CategoryResult:
        """Special handling for ambiguous messages"""
        ambiguous_indicators = [
            r"\b(this|that|it)\s+is\s+(ridiculous|crazy|insane|unbelievable)\b",
            r"\b(what|why|how)\s+is\s+this\b",
            r"\b(seriously|really)\??\s*$",
        ]
        
        message_lower = message.lower()
        for pattern in ambiguous_indicators:
            if re.search(pattern, message_lower):
                return CategoryResult(
                    message=message,
                    category="feedback",  # Default for ambiguous negative sentiment
                    confidence=0.4,
                    explanation="Ambiguous message with negative sentiment, likely feedback",
                    method="ambiguous_handler"
                )
        
        # If not detectably ambiguous, return None to indicate normal processing
        return None

def load_custom_categories(file_path: str) -> Dict[str, str]:
    """Load custom categories from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading custom categories: {e}")
        return {}

def main():
    """CLI interface for the text categorizer"""
    parser = argparse.ArgumentParser(description="Smart Text Categorizer")
    parser.add_argument('--messages', nargs='+', help='Messages to categorize')
    parser.add_argument('--file', help='File containing messages (one per line)')
    parser.add_argument('--categories', help='JSON file with custom categories')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize categorizer
    categorizer = TextCategorizer(model=args.model)
    
    # Get messages
    messages = []
    if args.messages:
        messages = args.messages
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                messages = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        # Default test messages
        messages = [
            "I lost access to my account",
            "Can you recommend some books about AI?",
            "Why was I charged twice?",
            "The onboarding was super smooth!",
            "This is ridiculous",
            "The app keeps crashing when I try to upload files",
            "What features are included in the premium plan?",
            "How do I reset my password?"
        ]
        print("Using default test messages...")
    
    # Load custom categories if provided
    custom_categories = None
    if args.categories:
        custom_categories = load_custom_categories(args.categories)
    
    # Categorize messages
    print(f"\n🧠 Categorizing {len(messages)} messages...\n")
    results = categorizer.categorize_batch(messages, custom_categories)
    
    # Format output - ensure all numeric values are JSON serializable
    output_data = {
        "results": [
            {
                "message": r.message,
                "category": r.category,
                "confidence": float(r.confidence),  # Convert to Python float
                "explanation": r.explanation,
                "method": r.method
            }
            for r in results
        ],
        "summary": {
            "total_messages": len(messages),
            "categories_used": list(set(r.category for r in results)),
            "avg_confidence": float(sum(r.confidence for r in results) / len(results))  # Convert to Python float
        }
    }
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output_data, indent=2))
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"Total messages: {output_data['summary']['total_messages']}")
    print(f"Categories used: {', '.join(output_data['summary']['categories_used'])}")
    print(f"Average confidence: {output_data['summary']['avg_confidence']:.2f}")

if __name__ == "__main__":
    main() 