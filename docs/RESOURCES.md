# Project Resources

This file lists useful references for working with RetinaGuard V500.

## Key Technologies
- Flask: REST API backend framework
- OpenCV: medical image processing and feature extraction
- TensorFlow/Keras: deep learning model integration (planned)
- MongoDB: optional report storage for clinical tracking
- Express: frontend API proxy and static content delivery

## Useful References
- Flask documentation: https://flask.palletsprojects.com/
- OpenCV Python tutorials: https://opencv.org/
- MongoDB documentation: https://www.mongodb.com/docs/
- TensorFlow Keras guide: https://www.tensorflow.org/guide/keras

## Notes
- The backend is designed to run without MongoDB if the database is unavailable.
- The system includes a fallback rule-based mode for image analysis when TensorFlow is not installed.
