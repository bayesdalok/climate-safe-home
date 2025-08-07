#!/usr/bin/env python3
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Add this basic route to test
@app.route('/')
def home():
    return jsonify({"status": "Working!", "message": "Welcome to Climate Safe Home"})

# Keep your existing routes if needed
@app.route('/api/health')
def health():
    return jsonify({"status": "OK"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)