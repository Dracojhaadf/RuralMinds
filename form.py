import os
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- CONFIGURATION ---
DATA_FILE = 'student_interactions.json'

def load_student_data():
    """Loads all student interaction records from the JSON file."""
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # Handle empty or corrupted JSON file
            return []

def save_student_data(data):
    """Saves all student interaction records to the JSON file."""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# --- FLASK ROUTES ---

@app.route('/student_form')
def student_form_page():
    """Renders the HTML page for the student interaction form."""
    return render_template('student_form.html')

@app.route('/submit_interaction', methods=['POST'])
def submit_interaction():
    """Handles submission of the student interaction form."""
    try:
        data = request.get_json()
        
        # 1. Extract and sanitize data (basic sanitization here)
        name = data.get('name', '').strip().title()
        age = data.get('age')
        major = data.get('major', '').strip().title()
        preferred_method = data.get('preferredMethod', '').strip()
        favorite_subject = data.get('favoriteSubject', '').strip().title()
        
        if not name or not preferred_method:
             return jsonify({'success': False, 'message': 'Name and Preferred Interaction Method are required.'}), 400

        # 2. Structure the data
        new_record = {
            'timestamp': os.stat(DATA_FILE).st_mtime if os.path.exists(DATA_FILE) else None,
            'name': name,
            'age': int(age) if age and str(age).isdigit() else 'Unknown',
            'major': major if major else 'Undeclared',
            'interaction_preference': preferred_method,
            'favorite_subject': favorite_subject if favorite_subject else 'Not specified'
        }

        # 3. Load, append, and save
        all_data = load_student_data()
        all_data.append(new_record)
        save_student_data(all_data)

        message = (
            f"Thank you, {name}! Your profile has been submitted. "
            f"You are looking to connect over: {preferred_method} (Topic: {new_record['favorite_subject']})."
        )
        
        return jsonify({'success': True, 'message': message, 'data': new_record})

    except Exception as e:
        return jsonify({'success': False, 'message': f"Server error during submission: {str(e)}"}), 500

@app.route('/student_list')
def list_students():
    """Renders a simple page to view submitted student data."""
    students = load_student_data()
    return render_template('student_list.html', students=students)


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Initialize the data file if it doesn't exist
    if not os.path.exists(DATA_FILE):
        save_student_data([])
        
    print(f"Starting Student Form Backend. Access at: http://127.0.0.1:5001/student_form")
    print(f"View submitted forms at: http://127.0.0.1:5001/student_list")
    app.run(debug=True, port=5001)