'''
    Name:   Javier S. Guerrero
    Date:   August 12th, 2026

    Detects a human face and runs a callback.
'''

class FaceDetection():
    def __init__(self, session, on_face_detection=lambda is_in_view : is_in_view):
        self.__on_detection_callback = on_face_detection
        self.__face_detection = session.service("ALFaceDetection")
        self.__memory = session.service("ALMemory")
        self.__subscriber = self.__memory.subscriber("FaceDetected")
        self.__subscriber_id = self.__subscriber.signal.connect(self.__on_face_detected)
        self.__got_face = False
        # Initiate face detection from sensors every 500ms at 80% accuracy
        self.__face_detection.subscribe("FaceDetectionAndTracker", 500, 0.8)
    def __del__(self):
        self.__subscriber.signal.disconnect(self.__on_face_detected, self.__subscriber_id)
        self.__face_detection.unsubscribe("FaceDetectionAndTracker")
    
    def __on_face_detected(self, face_info):
        '''Event handler for when face is detected.'''
        if face_info == []:
            self.__got_face = False
            self.__on_detection_callback(False)
        elif not self.__got_face:
            self.__got_face = True
            self.__on_detection_callback(True)

    def enableFaceTracking(self, enable):
        self.__face_detection.setTrackingEnabled(enable)