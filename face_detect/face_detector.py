# -*- coding: utf-8 -*-
import numpy as np
import pickle
import cv2
import os

class FaceDetector:

    def __init__(self, detector_folder, face_embeddings_model, recognizer, label_encoder, confidence = 0.6):
        protoPath = os.path.sep.join([detector_folder, "deploy.prototxt"])
        modelPath = os.path.sep.join([detector_folder, "res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        self.face_embeddings_model = cv2.dnn.readNetFromTorch(face_embeddings_model)
        self.recognizer = pickle.loads(open(recognizer, "rb").read())
        self.label_encoder = pickle.loads(open(label_encoder, "rb").read())
        self.confidence = confidence

    def detect_and_recognize(self, frame):

        (h, w) = frame.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.face_embeddings_model.setInput(faceBlob)
                vec = self.face_embeddings_model.forward()

                # perform classification to recognize the face
                preds = self.recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                probability = preds[j]
                name = self.label_encoder.classes_[j]
                confidence = "{:.2f}%".format(probability * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                #cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                #cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                return name, confidence, (startX, startY, endX, endY)
            else:
                return None, None