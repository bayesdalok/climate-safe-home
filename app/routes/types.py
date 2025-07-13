from flask import jsonify
from app import app

@app.route('/api/structure-types', methods=['GET'])
def get_structure_types():
    """Get available structure types"""
    structure_types = [
        {'value': 'mud_brick', 'label': 'Mud Brick', 'description': 'Traditional mud brick construction'},
        {'value': 'concrete', 'label': 'Concrete', 'description': 'Reinforced concrete structure'},
        {'value': 'wood', 'label': 'Wood', 'description': 'Wooden frame construction'},
        {'value': 'bamboo', 'label': 'Bamboo', 'description': 'Bamboo pole construction'},
        {'value': 'thatch', 'label': 'Thatch', 'description': 'Traditional thatch roofing'},
        {'value': 'tin_sheet', 'label': 'Tin Sheet', 'description': 'Metal sheet construction'}
    ]
    return jsonify({'success': True, 'data': structure_types})

@app.route('/api/foundation-types', methods=['GET'])
def get_foundation_types():
    """Get available foundation types"""
    foundation_types = [
        {'value': 'concrete', 'label': 'Concrete Foundation'},
        {'value': 'stone', 'label': 'Stone Foundation'},
        {'value': 'earth', 'label': 'Earth/Mud Foundation'},
        {'value': 'raised', 'label': 'Raised/Stilts Foundation'}
    ]
    return jsonify({'success': True, 'data': foundation_types})

@app.route('/api/roof-types', methods=['GET'])
def get_roof_types():
    """Get available roof types"""
    roof_types = [
        {'value': 'sloped', 'label': 'Sloped/Pitched Roof'},
        {'value': 'flat', 'label': 'Flat Roof'},
        {'value': 'curved', 'label': 'Curved/Dome Roof'}
    ]
    return jsonify({'success': True, 'data': roof_types})
