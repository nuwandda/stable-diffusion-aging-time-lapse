import random
from gender_detection.gender_detection import predict_gender
import face_recognition


def set_seed():
    seed = random.randint(42,4294967295)
    return seed


def gender(image):
    boxes_face = face_recognition.face_locations(image)
    if len(boxes_face) == 1:
        x0,y1,x1,y0 = boxes_face[0]
        face_image = image[x0:x1,y0:y1]
        
        return predict_gender(face_image)
    else:
        return 'Multiple faces', 1
