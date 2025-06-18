# 🧠 Smart Text Categorizer

**AI Challenge Solution: Intelligent text classification with 3-tier fallback system**

A robust text categorization system that classifies user messages into high-level categories using a 3-tier hybrid approach: Large Language Models (LLM) → Sentence Transformers → Rule-based patterns for maximum reliability and speed.

## ✨ Features

- **3-Tier Hybrid Classification**: OpenAI LLM → Sentence Transformers → Rule-based patterns
- **Local Model Fallback**: Works offline with sentence-transformers when LLM unavailable
- **Auto-Download Models**: Automatically downloads and caches models on first use
- **Configurable Categories**: Custom taxonomy support via JSON files
- **Ambiguous Message Handling**: Special logic for unclear messages
- **Confidence Scoring**: Returns confidence levels for each classification
- **Multiple Interfaces**: CLI tool and REST API endpoint
- **Batch Processing**: Handles multiple messages efficiently
- **Robust Fallbacks**: Works even without internet/API access
- **Detailed Explanations**: Shows reasoning for each classification

## 🎥 Demo Video

Watch the complete demo and testing of Smart Text Categorizer:

[![Smart Text Categorizer Demo](https://img.youtube.com/vi/VFaKGubkerc/0.jpg)](https://youtu.be/VFaKGubkerc?si=mQa61qGXO4NKbyTU)

_Click the image above to watch the demo video_

## 🚀 Quick Start

### Complete Setup

1. **Create and activate virtual environment (recommended):**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or use the provided activation script:

```bash
source activate_env.sh
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Download models (optional - recommended for faster startup):**

```bash
python3 setup_models.py
```

4. **Set up OpenAI API (optional for LLM features):**

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

5. **Test the system:**

```bash
python3 text_categorizer.py --messages "I lost my password" "Great service"
```

### Installation Only

For minimal setup without model pre-download:

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The ~90MB `all-MiniLM-L6-v2` model will be downloaded automatically on first use. This may take a few minutes on the first run but will be cached locally for subsequent uses.

### Environment Setup

#### Virtual Environment (Recommended)

To avoid conflicts with system packages:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

Use the provided activation script for convenience:

```bash
source activate_env.sh
```

#### OpenAI API Key

For LLM features, set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
OPENAI_API_KEY=your-api-key-here
```

### Basic Usage

Run with default test messages:

```bash
python3 text_categorizer.py
```

Categorize custom messages:

```bash
python3 text_categorizer.py --messages "I can't log in" "Great product!" "How much does it cost?"
```

Process messages from file:

```bash
python3 text_categorizer.py --file test_messages.txt
```

Use custom categories:

```bash
python3 text_categorizer.py --categories example_categories.json --file test_messages.txt
```

## 📊 Example Output

### With LLM (OpenAI API Available)

```
🧠 Categorizing 10 messages...

{
  "results": [
    {
      "message": "I lost access to my account",
      "category": "account_issue",
      "confidence": 0.85,
      "explanation": "The message explicitly mentions losing access to the account, indicating a problem with user accounts and access.",
      "method": "llm"
    },
    {
      "message": "Can you recommend some books about AI?",
      "category": "recommendation_request",
      "confidence": 0.9,
      "explanation": "The message explicitly asks for recommendations on books about AI, indicating a clear request for suggestions.",
      "method": "llm"
    },
    {
      "message": "Why was I charged twice?",
      "category": "billing",
      "confidence": 0.85,
      "explanation": "The message indicates a concern about being charged twice, which falls under financial concerns and billing issues.",
      "method": "llm"
    },
    {
      "message": "The onboarding was super smooth!",
      "category": "feedback",
      "confidence": 0.85,
      "explanation": "The message expresses a positive opinion about the onboarding process, indicating feedback on the user experience.",
      "method": "llm"
    },
    {
      "message": "This is ridiculous",
      "category": "feedback",
      "confidence": 0.4,
      "explanation": "Ambiguous message with negative sentiment, likely feedback",
      "method": "ambiguous_handler"
    },
    {
      "message": "The app keeps crashing when I try to upload files",
      "category": "technical_support",
      "confidence": 0.85,
      "explanation": "The message indicates a technical problem with the app crashing, which falls under technical support.",
      "method": "llm"
    },
    {
      "message": "What features are included in the premium plan?",
      "category": "product_inquiry",
      "confidence": 0.9,
      "explanation": "The user is inquiring about the features included in a specific plan, indicating a product-related question.",
      "method": "llm"
    },
    {
      "message": "How do I reset my password?",
      "category": "account_issue",
      "confidence": 0.85,
      "explanation": "The message specifically mentions resetting a password, which falls under account issues related to user accounts and login.",
      "method": "llm"
    },
    {
      "message": "The new update broke my workflow",
      "category": "technical_support",
      "confidence": 0.85,
      "explanation": "The message indicates a technical issue caused by the new update, suggesting a problem that needs troubleshooting.",
      "method": "llm"
    },
    {
      "message": "Could you suggest a better alternative?",
      "category": "recommendation_request",
      "confidence": 0.85,
      "explanation": "The user is explicitly asking for a suggestion, indicating a high likelihood that this falls under the recommendation_request category.",
      "method": "llm"
    }
  ],
  "summary": {
    "total_messages": 10,
    "categories_used": [
      "billing",
      "product_inquiry",
      "recommendation_request",
      "feedback",
      "technical_support",
      "account_issue"
    ],
    "avg_confidence": 0.8150000000000001
  }
}

📊 Summary:
Total messages: 10
Categories used: billing, product_inquiry, recommendation_request, feedback, technical_support, account_issue
Average confidence: 0.82
```

### With Sentence Transformers (Local Model Fallback)

```
🧠 Categorizing 10 messages...

🔄 LLM failed, loading cached sentence transformer model...
✅ Sentence transformer model loaded successfully
{
  "results": [
    {
      "message": "I lost access to my account",
      "category": "account_issue",
      "confidence": 0.9121598207950592,
      "explanation": "High semantic similarity classification. Similarities: account_issue: 0.84, billing: 0.32",
      "method": "sentence_transformer"
    },
    {
      "message": "Can you recommend some books about AI?",
      "category": "recommendation_request",
      "confidence": 0.6296978944540024,
      "explanation": "Semantic similarity classification. Similarities: recommendation_request: 0.49, product_inquiry: 0.30",
      "method": "sentence_transformer"
    },
    {
      "message": "Why was I charged twice?",
      "category": "billing",
      "confidence": 0.673310580253601,
      "explanation": "Semantic similarity classification. Similarities: billing: 0.58, account_issue: 0.23",
      "method": "sentence_transformer"
    },
    {
      "message": "The onboarding was super smooth!",
      "category": "feedback",
      "confidence": 0.6543543761968613,
      "explanation": "Semantic similarity classification. Similarities: feedback: 0.54, product_inquiry: 0.13",
      "method": "sentence_transformer"
    },
    {
      "message": "This is ridiculous",
      "category": "feedback",
      "confidence": 0.4,
      "explanation": "Ambiguous message with negative sentiment, likely feedback",
      "method": "ambiguous_handler"
    },
    {
      "message": "The app keeps crashing when I try to upload files",
      "category": "technical_support",
      "confidence": 0.6703310310840607,
      "explanation": "Semantic similarity classification. Similarities: technical_support: 0.57, account_issue: 0.22",
      "method": "sentence_transformer"
    },
    {
      "message": "What features are included in the premium plan?",
      "category": "product_inquiry",
      "confidence": 0.7083256375789643,
      "explanation": "Semantic similarity classification. Similarities: product_inquiry: 0.64, billing: 0.43",
      "method": "sentence_transformer"
    },
    {
      "message": "How do I reset my password?",
      "category": "account_issue",
      "confidence": 0.8572519969940186,
      "explanation": "High semantic similarity classification. Similarities: account_issue: 0.74, billing: 0.26",
      "method": "sentence_transformer"
    },
    {
      "message": "The new update broke my workflow",
      "category": "technical_support",
      "confidence": 0.5284292989969254,
      "explanation": "Semantic similarity classification. Similarities: technical_support: 0.30, account_issue: 0.20",
      "method": "sentence_transformer"
    },
    {
      "message": "Could you suggest a better alternative?",
      "category": "recommendation_request",
      "confidence": 0.7370416831970215,
      "explanation": "Semantic similarity classification. Similarities: recommendation_request: 0.70, feedback: 0.26",
      "method": "sentence_transformer"
    }
  ],
  "summary": {
    "total_messages": 10,
    "categories_used": [
      "account_issue",
      "billing",
      "product_inquiry",
      "technical_support",
      "feedback",
      "recommendation_request"
    ],
    "avg_confidence": 0.6770902319550514
  }
}

📊 Summary:
Total messages: 10
Categories used: account_issue, billing, product_inquiry, technical_support, feedback, recommendation_request
Average confidence: 0.68
```

### Without LLM and Sentence Transformers (Rule-based Fallback)

```
🧠 Categorizing 10 messages...

⚠️  Sentence transformers not available, falling back to rule-based classification
{
  "results": [
    {
      "message": "I lost access to my account",
      "category": "account_issue",
      "confidence": 0.8,
      "explanation": "Rule-based classification based on keywords: account, access, lost access",
      "method": "rule"
    },
    {
      "message": "Can you recommend some books about AI?",
      "category": "recommendation_request",
      "confidence": 0.7,
      "explanation": "Rule-based classification based on keywords: can you recommend, recommend",
      "method": "rule"
    },
    {
      "message": "Why was I charged twice?",
      "category": "billing",
      "confidence": 0.7,
      "explanation": "Rule-based classification based on keywords: charged, why was i charged",
      "method": "rule"
    },
    {
      "message": "The onboarding was super smooth!",
      "category": "feedback",
      "confidence": 0.5,
      "explanation": "Rule-based classification based on keywords: smooth",
      "method": "rule"
    },
    {
      "message": "This is ridiculous",
      "category": "feedback",
      "confidence": 0.4,
      "explanation": "Ambiguous message with negative sentiment, likely feedback",
      "method": "ambiguous_handler"
    },
    {
      "message": "The app keeps crashing when I try to upload files",
      "category": "technical_support",
      "confidence": 0.5,
      "explanation": "Rule-based classification based on keywords: crashing",
      "method": "rule"
    },
    {
      "message": "What features are included in the premium plan?",
      "category": "product_inquiry",
      "confidence": 0.8,
      "explanation": "Rule-based classification based on keywords: premium, plan, features",
      "method": "rule"
    },
    {
      "message": "How do I reset my password?",
      "category": "account_issue",
      "confidence": 0.5,
      "explanation": "Rule-based classification based on keywords: password",
      "method": "rule"
    },
    {
      "message": "The new update broke my workflow",
      "category": "other",
      "confidence": 0.1,
      "explanation": "Rule-based classification - no strong patterns found, defaulting to 'other'",
      "method": "rule"
    },
    {
      "message": "Could you suggest a better alternative?",
      "category": "recommendation_request",
      "confidence": 0.5,
      "explanation": "Rule-based classification based on keywords: suggest",
      "method": "rule"
    }
  ],
  "summary": {
    "total_messages": 10,
    "categories_used": [
      "billing",
      "recommendation_request",
      "other",
      "product_inquiry",
      "technical_support",
      "feedback",
      "account_issue"
    ],
    "avg_confidence": 0.55
  }
}

📊 Summary:
Total messages: 10
Categories used: billing, recommendation_request, other, product_inquiry, technical_support, feedback, account_issue
Average confidence: 0.55
```

## 🔧 Default Categories

- **account_issue**: Login, password, access problems
- **billing**: Payments, charges, refunds, pricing
- **recommendation_request**: Asking for suggestions/advice
- **feedback**: Opinions, reviews, complaints, praise
- **technical_support**: Technical problems, bugs, errors
- **product_inquiry**: Questions about features/functionality
- **general_inquiry**: General questions and information requests
- **other**: Uncategorized messages

## 🌐 API Server

Start the REST API:

```bash
python3 api_server.py
```

The server runs on port 5050 by default. You can change it with:

```bash
PORT=8080 python3 api_server.py
```

### API Endpoints

**POST /categorize**

```bash
curl -X POST http://localhost:5050/categorize \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      "I lost access to my account",
      "Can you recommend some books?"
    ]
  }'
```

Example response:

```json
{
  "results": [
    {
      "category": "account_issue",
      "confidence": 0.85,
      "explanation": "The message indicates a problem with accessing the user account, which falls under the account_issue category.",
      "message": "I lost access to my account",
      "method": "llm"
    },
    {
      "category": "recommendation_request",
      "confidence": 0.9,
      "explanation": "The message explicitly asks for book recommendations, indicating a high likelihood that it falls under the recommendation_request category.",
      "message": "Can you recommend some books?",
      "method": "llm"
    }
  ],
  "summary": {
    "avg_confidence": 0.875,
    "categories_used": ["account_issue", "recommendation_request"],
    "total_messages": 2
  }
}
```

**GET /health** - Health check  
**GET /categories** - Get available categories

## ⚙️ CLI Options

- `--messages`: List of messages to categorize
- `--file`: File containing messages (one per line)
- `--categories`: JSON file with custom categories
- `--model`: OpenAI model to use (default: gpt-3.5-turbo)
- `--output`: Save results to file

## 🎯 Architecture

### 3-Tier Classification Strategy

1. **Primary**: OpenAI LLM for nuanced understanding (highest accuracy)
2. **Secondary**: Sentence Transformers for semantic similarity (lazy-loaded fallback)
3. **Tertiary**: Rule-based patterns for reliability (fastest fallback)
4. **Special**: Ambiguous message detection (runs first)

**Smart Resource Management**: Sentence Transformers are only loaded when LLM fails, saving memory and startup time when LLM is available.

### Classification Methods

**Sentence Transformers (Local Model)**

- Uses `all-MiniLM-L6-v2` model for semantic similarity
- Enhanced category embeddings with multiple examples per category
- Compares message embeddings with averaged category embeddings
- Dynamic similarity thresholds: 0.15 minimum, 0.25 for higher confidence
- Confidence range: 0.2-0.95 with similarity-based mapping

**Rule-Based Patterns**

- Regex patterns for key categories:
  - Account issues: login, password, access keywords
  - Billing: charge, payment, refund patterns
  - Recommendations: suggest, recommend, advice patterns
  - Feedback: sentiment and opinion indicators
  - Technical: error, bug, broken patterns

### Confidence Scoring

- **LLM**: Uses model's internal confidence (0.85-0.9 for clear matches)
- **Sentence Transformers**: Enhanced similarity-based mapping:
  - 0.15-0.25 similarity → 0.3-0.5 confidence (conservative)
  - 0.25-1.0 similarity → 0.5-0.9 confidence (confident)
  - 0.7+ similarity → up to 0.95 confidence (very high boost)
- **Rules**: Based on pattern match strength (0.5-0.8 depending on keywords)
- **Ambiguous**: Lower confidence (0.4) for unclear messages
- **No match**: Very low confidence (0.1-0.2) when defaulting to 'other'

## 🚦 Performance Notes

- **With OpenAI API**: Highest accuracy (~0.82 avg confidence), ~2-3s per batch, ~50MB memory
- **Sentence Transformers**: Good accuracy (~0.68 avg confidence), ~1-2s per batch, works offline, ~200MB memory
- **Rule-based only**: Fast (<1s), moderate accuracy (~0.55 avg confidence) on clear messages, ~50MB memory
- **Smart loading**: Sentence transformers only load when needed (LLM fails)
- **Graceful degradation**: Full offline capability with local models

## 🧪 Testing

### Testing 3-Tier Fallback System

You can test each classification tier individually:

#### 1. Test with LLM (Highest Accuracy)

Requires valid OpenAI API key in `.env`:

```bash
# Ensure API key is set
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Test LLM classification
python3 text_categorizer.py --file test_messages.txt
```

Expected output: `"method": "llm"` with confidence ~0.85-0.9

#### 2. Test with Local Model (Semantic Similarity)

Disable LLM to test sentence transformers:

```bash
# Temporarily disable API key
mv .env .env.backup

# Test with sentence transformers (ensure model is downloaded)
python3 setup_models.py
python3 text_categorizer.py --file test_messages.txt

# Restore API key
mv .env.backup .env
```

Expected output: `"method": "sentence_transformer"` with confidence ~0.2-0.85

#### 3. Test with Rule-Based Only (Fastest)

Disable both LLM and sentence transformers:

```bash
# Temporarily disable API key and uninstall sentence-transformers
mv .env .env.backup
pip3 uninstall sentence-transformers -y

# Test with rules only
python3 text_categorizer.py --file test_messages.txt

# Restore everything
pip3 install sentence-transformers
mv .env.backup .env
```

Expected output: `"method": "rule"` with confidence ~0.1-0.8

### Test API server:

```bash
# Start server on default port 5050
python3 api_server.py &

# Wait for server to start (3-5 seconds)
sleep 5

# Test health endpoint first
curl -X GET http://localhost:5050/health

# Test categorization endpoint
curl -X POST http://localhost:5050/categorize -H "Content-Type: application/json" -d '{"messages": ["Test message"]}'

# Stop server when done
pkill -f api_server.py
```

### Expected Performance by Method

| Method                | Accuracy | Speed | Offline | Confidence Range | Avg Confidence |
| --------------------- | -------- | ----- | ------- | ---------------- | -------------- |
| LLM                   | Highest  | 2-3s  | ❌ No   | 0.85-0.9         | ~0.82          |
| Sentence Transformers | Good     | 1-2s  | ✅ Yes  | 0.2-0.95         | ~0.68          |
| Rule-based            | Moderate | <1s   | ✅ Yes  | 0.1-0.8          | ~0.55          |

## 📈 Possible Improvements

- **Caching**: Store LLM results for repeated messages
- **Batch API calls**: Process multiple messages in single LLM request
- **Local models**: Support for local LLMs (Ollama, etc.)
- **Active learning**: Improve rules based on LLM classifications
- **Metrics**: Track accuracy, response times, category distribution

## 🔍 Edge Cases Handled

- Empty/whitespace messages
- Very long messages
- Non-English text (basic support)
- API failures/rate limits
- Invalid custom categories
- Ambiguous sentiment expressions

---

**Built for the AI Challenge - Smart Text Categorizer**  
_Classification made simple, reliable, and explainable_
