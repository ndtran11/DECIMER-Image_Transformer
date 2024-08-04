from singleton_decorator import singleton 
import pystow
import tensorflow as tf
from DECIMER import utils
import pickle
import os

default_path = pystow.join("DECIMER-V2")


def get_models(model_urls: dict):
    """Download and load models from the provided URLs.

    This function downloads models from the provided URLs to a default location,
    then loads tokenizers and TensorFlow saved models.

    Args:
        model_urls (dict): A dictionary containing model names as keys and their corresponding URLs as values.

    Returns:
        tuple: A tuple containing loaded tokenizer and TensorFlow saved models.
            - tokenizer (object): Tokenizer for DECIMER model.
            - DECIMER_V2 (tf.saved_model): TensorFlow saved model for DECIMER.
            - DECIMER_Hand_drawn (tf.saved_model): TensorFlow saved model for DECIMER HandDrawn.
    """
    # Download models to a default location
    model_paths = utils.ensure_models(default_path=default_path, model_urls=model_urls)

    # Load tokenizers
    tokenizer_path = os.path.join(
        model_paths["DECIMER"], "assets", "tokenizer_SMILES.pkl"
    )
    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    # Load DECIMER models
    DECIMER_V2 = tf.saved_model.load(model_paths["DECIMER"])
    # DECIMER_Hand_drawn = tf.saved_model.load(model_paths["DECIMER_HandDrawn"])

    return tokenizer, DECIMER_V2 #, DECIMER_Hand_drawn

HERE = os.path.dirname(os.path.abspath(__file__))

model_urls = {
    "DECIMER": "https://zenodo.org/record/8300489/files/models.zip",
    # "DECIMER_HandDrawn": "https://zenodo.org/records/10781330/files/DECIMER_HandDrawn_model.zip",
}

@singleton
class SingletonInferenceInterace(object):
    def __init__(self, *args, **kwargs):
        # Set the absolute path

        self._tokenizer, self._DECIMER_V2 = get_models(model_urls)
        
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def model(self):
        return self._DECIMER_V2
    
    @property
    def handwritten_model(self):
        return self._DECIMER_Hand_drawn
    
    def get_model(self, for_handrawn: bool = False):
        if for_handrawn:
            return self._DECIMER_Hand_drawn

        return self._DECIMER_V2