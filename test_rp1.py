import os
import cv2
import json

import app

path = os.path.join(r"e:\V500\Dataset\Original Dataset\Retinitis Pigmentosa", "Retinitis Pigmentosa1.jpg")
image = cv2.imread(path)
if image is not None:
    try:
        # We need to mock Flask request
        app.app.test_request_context().push()
        res = app.analyze_retinal_scan(image, {}, "Retinitis Pigmentosa1.jpg")
        print("Success!", str(res)[:100])
    except Exception as e:
        print("Failed!", type(e), str(e))
