# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np

# model = load_model("./server/Potato_model_v1.h5")
# class_names = ["Early Blight","Late Blight","Healthy"]

# def prepare_image(img_path,target_size=(256,256)):
#     img = image.load_img(img_path,target_size=target_size)
#     img_array = image.img_to_array(img)/255.0
#     img_array = np.expand_dims(img_array,0)
#     return img_array

# img_path = "leaf.JPG" # path to sample image
# img = prepare_image(img_path)
# pred = model.predict(img)
# pred_class = class_names[np.argmax(pred)]
# confidence = round(100*np.max(pred[0]),2)
# print(f"Predicted: {pred_class}, Confidence: {confidence}%")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("./Potato_model_v2.h5")
class_names = ["Early Blight", "Late Blight", "Healthy"]

def prepare_image(img_path, target_size=(256,256)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)  # ❌ no division by 255
    img_array = np.expand_dims(img_array, 0)  # (1, 256, 256, 3)
    return img_array

img_path = "leaf.JPG"
img = prepare_image(img_path)
pred = model.predict(img)
pred_class = class_names[np.argmax(pred)]
confidence = round(100*np.max(pred[0]), 2)

print(f"Predicted: {pred_class}, Confidence: {confidence}%")
