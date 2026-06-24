import app
import json
import base64

# Create a 1x1 black image in base64
valid_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

with app.app.test_client() as client:
    response = client.post('/api/analyze', 
                           data=json.dumps({"image": valid_b64, "patientId": "TEST-1"}),
                           content_type='application/json')
    print("STATUS:", response.status_code)
