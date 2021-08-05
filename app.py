import streamlit as st
import tensorflow as tf
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('/content/drive/MyDrive/web_app_models/new_resnet_model.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # COVID PREDICTION
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
    img=image.load_img(path,target_size=(224,224))
    img=np.asarray(img)
    img=np.expand_dims(img,axis=0)
    output=rmodel.predict(img)
    show_img=image.load_img(path,target_size=(224,224))
    plt.imshow(show_img)
    plt.show()
    if output[0][0] > output[0][1]:
        print("covid")
    else:
        print("non-covid")

    return output


       # size = (224,224)    
        #image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        #image = np.asarray(image)
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        #img_reshape = img[np.newaxis,...]
    
        #prediction = model.predict(img_reshape)
        
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    
    st.write(prediction)
    st.write(score)
    #print(
    #"This image most likely belongs to {} with a {:.2f} percent confidence."
    #.format(class_names[np.argmax(score)], 100 * np.max(score))
)
