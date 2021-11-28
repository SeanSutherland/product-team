from django.db import models
import tensorflow as tf
import json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import os
model = None
# Create your models here.
class Image(models.Model):
    picture = models.ImageField()
    classified = models.CharField(max_length=200, blank=True)
    upload = models.DateTimeField(auto_now_add=True)
    label_file = open("/home/seansutherland24/CV-App/backend/images/labels.json", "r")
    labels = json.loads(label_file.read())
    label_file.close()
    def fetchModel(self):
        if model is None:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            print("Probably about to fail")
            return tf.keras.models.load_model("/home/seansutherland24/CV-App/backend/images/inceptionv3")
        else: 
            return model
    def getLabels(self):
        label_file = open("/home/seansutherland24/CV-App/backend/images/labels.json", "r")
        labels = json.loads(label_file.read())
        label_file.close()
        return labels
    def __str__(self) -> str:
        return "Image classified at {}".format(self.upload.strftime('%Y-%m-%d %H:%M'))
            
    def save(self, *args, **kwargs):
        try:
            img = load_img(str("/home/seansutherland24/CV-App/media_root/" +str(self.picture)), color_mode="rgb", target_size=(299,299,3))
            img_array = img_to_array(img)
            img_array = np.array( [img_array,])
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)


        
            #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            #model = tf.keras.models.load_model("/home/seansutherland24/CV-App/backend/images/inceptionv3")
            model = self.fetchModel()
            preds = model.predict(img_array)[0]
            max_index = np.argmax(preds)
            labels = self.getLabels()
            mClass = labels[str(max_index)]
            self.classified = mClass
            print(self.classified)
            print('success')
        except Exception as err:
            print(type(err))
            print('classification failed')
        super().save(*args, **kwargs)
