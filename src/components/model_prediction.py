from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from src.entity.config_entity import ModelPredictionConfig
from tensorflow import keras
from typing import Dict
from fastapi import HTTPException, status
from utils import logger

class ModelPrediction:
    def __init__(self, model=None, image_file=None, config: ModelPredictionConfig = None):
        self.model = model
        self.image_file = image_file
        self.config = config


    def read_image_data(self) -> np.ndarray:
        # Read image data
        self.image = np.array(Image.open(BytesIO(self.image_file)))

    
    def preprocess_image(self) -> None:
        # Resize image
        resized_img = tf.image.resize(images=self.image, size=self.config.params_image_size)

        # Convert image to nparray
        image_array = keras.utils.img_to_array(resized_img)

        # Batch image
        self.image_array = np.array([image_array])

    
    def predict(self) -> Dict[str, str | int]:
        try:
            # Predict image
            predictions = self.model.predict(self.image_array)

            # Identify the class
            prediction_class = self.config.params_model_classes[np.argmax(predictions[0])]

            # Compute confidence score
            confidence_score = round(100 * (np.max(predictions[0])), 2)
            
            # Format output
            full_name = prediction_class.split('_')[-1]

            # Check if fullname is more than 1
            disease_name = name.split('-') if len(name) > 1 else name
            health_status = 'Healthy' if disease_name[-1] == 'healthy' else ' '.join(disease_name).title()
            result = {'health_status': health_status, 'confidence_score': confidence_score}

            return result
        except Exception as err:
            logger.exception(err)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        