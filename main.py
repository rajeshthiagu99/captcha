import base64
import io
from PIL import Image 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
img_width = 100
img_height = 40

# convert base64 string to image
def convert_base64_to_image(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def encode_single_sample(img_path,model_loaded,num_to_char_1):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)
    # 6. Map the characters in label to numbers
    preds = model_loaded.predict(img)
    pred_texts = decode_batch_predictions(preds,num_to_char_1)
    return pred_texts

# A utility function to decode the output of the network
def decode_batch_predictions(pred,num_to_char_1):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    print(input_len)

    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, : 4]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char_1(res)).numpy().decode("utf-8")
        output_text.append(res)
    print(output_text)
    return output_text

def get_recognized_text(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    model_path = 'final_model.h5'
    # load model from github
    
    model_loaded = keras.models.load_model(model_path)
    
    # Mapping integers back to original characters
    num_to_char_1 = layers.experimental.preprocessing.StringLookup(
        vocabulary=['I','N','Q','Y','T','F','Z','D','X','E','P','8','2','S','L','H','M','A','V','W','5','6','C','B','R','1','7','3','9','G','U','J','4','K'], mask_token=None, invert=True,num_oov_indices=0,oov_token='',
    )
    request_json = request.get_json()
    # save base64 string to image
    base64_string = eval(request_json['image'])

    image = convert_base64_to_image(base64_string)
    # convert to tf io 
    image.save('/tmp/test.png')
    result = encode_single_sample('/tmp/test.png',model_loaded,num_to_char_1)
    try: 
        result = result[0]
    except:
        result = 'error'
    return {'prediction':result}
