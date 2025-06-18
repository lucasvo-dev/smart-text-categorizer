#!/usr/bin/env python3
"""
Simple API server for the Smart Text Categorizer
"""

from flask import Flask, request, jsonify
from text_categorizer import TextCategorizer
import os

app = Flask(__name__)
categorizer = TextCategorizer()

@app.route('/categorize', methods=['POST'])
def categorize_messages():
    """API endpoint to categorize messages"""
    try:
        data = request.get_json()
        
        if not data or 'messages' not in data:
            return jsonify({
                'error': 'Missing "messages" field in request body'
            }), 400
        
        messages = data['messages']
        if not isinstance(messages, list):
            return jsonify({
                'error': '"messages" must be a list'
            }), 400
        
        custom_categories = data.get('categories', None)
        
        # Categorize messages
        results = categorizer.categorize_batch(messages, custom_categories)
        
        # Format response
        response = {
            'results': [
                {
                    'message': r.message,
                    'category': r.category,
                    'confidence': r.confidence,
                    'explanation': r.explanation,
                    'method': r.method
                }
                for r in results
            ],
            'summary': {
                'total_messages': len(messages),
                'categories_used': list(set(r.category for r in results)),
                'avg_confidence': sum(r.confidence for r in results) / len(results) if results else 0
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'llm_available': categorizer.client is not None
    })

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get available categories"""
    return jsonify(categorizer.default_categories)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"Starting Text Categorizer API on port {port}")
    print(f"LLM Available: {categorizer.client is not None}")
    
    app.run(host='0.0.0.0', port=port, debug=debug) 