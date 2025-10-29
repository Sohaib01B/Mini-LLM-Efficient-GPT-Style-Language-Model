from flask import Flask, render_template, request, jsonify, session
import torch
from model.model_loader import StoryGenerator
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🎮 Using device: {device}")

try:
    model_generator = StoryGenerator('saved_models/best_model.pt', device=device)
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    MODEL_LOADED = False

@app.route('/')
def index():
    return render_template('index.html', model_loaded=MODEL_LOADED)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/stories')
def stories():
    return render_template('stories.html', model_loaded=MODEL_LOADED)

@app.route('/generate', methods=['POST'])
def generate_story():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        max_length = data.get('max_length', 200)
        temperature = data.get('temperature', 0.8)
        top_k = data.get('top_k', 50)
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Generate story
        generated_text = model_generator.generate_story(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k
        )
        
        # Store in session for history
        if 'story_history' not in session:
            session['story_history'] = []
        
        story_entry = {
            'prompt': prompt,
            'story': generated_text,
            'timestamp': datetime.now().isoformat(),
            'params': {
                'temperature': temperature,
                'top_k': top_k,
                'max_length': max_length
            }
        }
        
        session['story_history'].append(story_entry)
        # Keep only last 10 stories
        session['story_history'] = session['story_history'][-10:]
        
        return jsonify({
            'story': generated_text,
            'prompt': prompt
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(session.get('story_history', []))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('story_history', None)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)