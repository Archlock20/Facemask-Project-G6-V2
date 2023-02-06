
import numpy as np
import cv2
import tensorflow as tf

from scipy.special import softmax

# Importar el modelo de deteccion de rostros
face_detection_model = cv2.dnn.readNetFromCaffe('./Models/deploy.prototxt.txt',
                                                './Models/res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Importar el modelo de reconocimiento de mascarillas
model = tf.keras.models.load_model('modelo_cnn_deteccion_mascarillas_AG_MT_V3')

# Crear las etiquetas y asignarles un color respectivo a cada una de ellas
etiquetas = ['Con Mascarilla', 'Sin Mascarilla']

def getcolor(etiqueta):
    if etiqueta == "Con Mascarilla":
        color = (0,255,0)
    else: 
        color = (0,0,255)
    return color


######EN FORMA DE FUNCION#####
def face_mask_prediction(img):
    # Paso 1 - Deteccion de rostros
    image = img.copy()
    mask_count = 0
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image,1,(300,300),(104,117,123),swapRB=True)
    #
    face_detection_model.setInput(blob)
    detection = face_detection_model.forward()
    for i in range(0,detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.5:
            box = detection[0,0,i,3:7]*np.array([w,h,w,h])
            box = box.astype(int)
            pt1 = (box[0],box[1])
            pt2 = (box[2],box[3])
            cv2.rectangle(image,pt1,pt2,(0,255,0),1)    
            # Paso 2 - Preprosesamiento de los datos
            face = image[box[1]:box[3],box[0]:box[2]]
            face_blob = cv2.dnn.blobFromImage(face,1,(100,100),(104,117,123),swapRB=True)
            face_blob_squeeze = np.squeeze(face_blob).T
            face_blob_rotate = cv2.rotate(face_blob_squeeze,cv2.ROTATE_90_CLOCKWISE)
            face_blob_flip = cv2.flip(face_blob_rotate,1)
            # Paso 3 - Normalizacion
            img_norm = np.maximum(face_blob_flip,0)/face_blob_flip.max()
            # Paso 4 - Aprendizaje Profundo (CNN)
            img_input = img_norm.reshape(1,100,100,3)
            result = model.predict(img_input)
            result = softmax(result)[0]
            confidence_index = result.argmax()
            confidence_score = result[confidence_index]
            label = etiquetas[confidence_index]
            label_text = '{}: {:,.0f}%'.format(label,confidence_score*100)
            if label == "Con Mascarilla":######################
                mask_count += 1################################
            #print(label_text)
            color = getcolor(label)
            cv2.rectangle(image,pt1,pt2,color,1)
            cv2.putText(image,label_text,pt1,cv2.FONT_HERSHEY_PLAIN,1,color,1)
            
    return image, mask_count

