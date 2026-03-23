"""
This code is for testing preprocess_data.py.
"""

import json
def test_json():
    with open("preprocessed_data.json") as f:
        data = json.load(f)
    
    assert len(data)>0, "No windows generated!"
    for i, window in enumerate(data):
        assert len(window) == 10, f"Length of windoe {i} is not 10!"
        for event in window:
            assert "event_id" in event
            assert "time" in event
            assert "description" in event
    
    print("All tests are passed!")

test_json()