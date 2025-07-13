from app import app  # This should match your __init__.py structure

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)