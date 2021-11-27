from django.db import models
import tensorflow as tf
import json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np

# Create your models here.

class Image(models.Model):
    picture = models.ImageField()
    classified = models.CharField(max_length=200, blank=True)
    upload = models.DateTimeField(auto_now_add=True)
    label_file = open("labels.json", "r")
    labels = json.loads(label_file.read())
    label_file.close()

    def fetchModel():
        if model is None:
            return tf.keras.models.load_model("inceptionv3")
        else: 
            return model


    def __str__(self) -> str:
        return "Image classified at {}".format(self.upload.strftime('%Y-%m-%d %H:%M'))
            

    def save(self, *args, **kwargs):
        try:
            img = load_img(self.picture, color_mode="rgb", target_size=(299,299,3))
            img_array = img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

            model = fetchModel()
            preds = model.predict(img_array)[0]

            max_index = np.argmax(preds)
            self.classified = str(labels[str(max_index)])
            print('success')
        except:
            print('classification failed')
        super().save(*args, **kwargs)
