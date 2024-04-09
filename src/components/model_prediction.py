from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from src.entity.config_entity import ModelPredictionConfig
from tensorflow import keras
from typing import Dict


class ModelPrediction:
    def __init__(self, model=None, image_file=None, config: ModelPredictionConfig = None):
        self.model = model,
        self.image_file = image_file,
        self.config = config
    

    def read_image_data(self) -> np.ndarray:
        self.image = np.array(Image.open(BytesIO(self.image_file)))

    
    def preprocess_image(self):
        # Resize image
        resized_img = tf.image.resize(images=self.image, size=self.config.params_image_size)

        # Convert image to nparray
        image_array = keras.utils.img_to_array(resized_img)

        # Batch image
        self.image_array = np.array([image_array])

    
    def predict(self) -> Dict[str, str | int]:
        # Predict image
        predictions = self.model.predict(self.image_array)

        # Identify the class
        prediction_class = self.config.params_model_classes[np.argmax(predictions[0])]

        # Compute confidence score
        confidence_score = round(100 * (np.max(predictions[0])), 2)

        result = {'class': prediction_class, 'confidence_score': confidence_score}

        return result

        