from flask import jsonify
from app import app
from app.models.builder import get_builder_data

@app.route('/api/builder/<builder_name>', methods=['GET'])
def get_builder_report(builder_name):
    """Return builder performance report"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT name, location, certification_status, complaints, success_rate, last_updated
            FROM builders
            WHERE LOWER(name) = LOWER(?)
        ''', (builder_name,))
        row = cursor.fetchone()
        conn.close()

        if row:
            builder_data = {
                'name': row[0],
                'location': row[1],
                'certification_status': row[2],
                'complaints': row[3],
                'success_rate': row[4],
                'last_updated': row[5]
            }
            return jsonify({'success': True, 'data': builder_data})
        else:
            return jsonify({'success': False, 'error': 'Builder not found'}), 404

    except Exception as e:
        logger.error(f"Error fetching builder: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    pass