from numpy.lib.type_check import imag
import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.title('--- ID CARD EXTRACTION ---')
st.write('\n')

# image = Image('text_detection_faster/test.png')
# show = st.image(image, use_column_width=False, width=300)

# image = cv2.imread('text_detection_faster/test.png')
# show = st.image(image, use_column_width=False, width=300)



st.sidebar.title("Upload Image")

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.sidebar.file_uploader("",type=['png', 'jpg', 'jpeg'])

if st.sidebar.button("Click here to Extract Information"):
  if uploaded_file is None:
    st.sidebar.write("Please upload an Image to Extract Information")
  else:
    print(type(uploaded_file))
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    test = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.write(str(type(test)))
    show = st.image(test, use_column_width=True, width=300, channels="BGR")
    cv2.imwrite('chien.png', test)