#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Show that we're in the virtual environment
echo "✅ Virtual environment activated"
echo "📦 Python location: $(which python3)"
echo "📋 Installed packages:"
pip list | grep -E "(openai|sentence-transformers|flask|scikit-learn)"

echo ""
echo "🚀 Ready to use Smart Text Categorizer!"
echo ""
echo "Usage examples:"
echo "  python3 text_categorizer.py"
echo "  python3 text_categorizer.py --messages 'I lost my password' 'Great service'"
echo "  python3 api_server.py"
echo ""
echo "To deactivate: deactivate" 