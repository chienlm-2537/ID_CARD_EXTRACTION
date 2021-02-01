from numpy.lib.type_check import imag
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from completedModel import CompletedModel
import time



st.title('--- ID CARD EXTRACTION ---')
st.write('\n')

@st.cache(persist = True)
def load_model():
  model = CompletedModel()
  return model

st.sidebar.title("Upload Image")
model = CompletedModel()
# st.set_option('deprecation.showfileUploaderEncoding', False)
def main():
  # model = load_model()
  uploaded_file = st.sidebar.file_uploader("",type=['png', 'jpg', 'jpeg'])
  if uploaded_file is not None:
      print(type(uploaded_file))
      file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
      test = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
      show = st.image(test, use_column_width=True, width=300, channels="BGR")
  if st.sidebar.button("Click here to Extract Information"):
    if uploaded_file is None:
      st.sidebar.write("Please upload an Image to Extract Information")
    else:
      try:
        print("start extraction")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        test = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        start = time.time()
        model.predict(test)
        end = time.time()
        show1 = st.image(model.mask, use_column_width=True, width=300, caption="Mask image")
        show3 = st.image(model.id_card, use_column_width=True, width=300, channels="BGR", caption="ID_Card image")
        show2 = st.image(model.cropped_image, use_column_width=True, width=300, channels="BGR", caption="detect field")
        st.write("Time processing: {} s".format(round(end-start)))
        st.write("\n")
        st.write("White mask ratio {}".format(model.ratio))
        print(model.field_dict.keys())
        st.write("ID NUMBER: {}".format(model.field_dict['id']))
        st.write("\n")
        st.write("NAME: {}".format(str(model.field_dict["name"]).upper()))
        st.write("\n")
        st.write("BIRTH: {}".format(model.field_dict["birth"]))
        st.write("\n")
        st.write("HOME: {}".format(model.field_dict["home"]))
        st.write("\n")
        st.write("ADDRESS: {}".format(model.field_dict["add"]))
        pass
      except:
        st.write("ERROR WHEN EXTRACT INFORMATION FROM IDCARD")
        pass

      # try:
      #   print("start extraction")
      #   file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
      #   test = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
      #   start = time.time()
      #   model.predict(test)
      #   end = time.time()
      #   show1 = st.image(model.cropped_image, use_column_width=True, width=300, channels="BGR")
      #   st.write("Time processing: {} s".format(round(end-start)))
      #   print(model.field_dict.keys())
      #   st.write("ID NUMBER: {}".format(model.field_dict['id']))
      #   st.write("\n")
      #   st.write("NAME: {}".format(model.field_dict["name"]))
      #   st.write("\n")
      #   st.write("BIRTH: {}".format(model.field_dict["birth"]))
      #   st.write("\n")
      #   st.write("HOME: {}".format(model.field_dict["home"]))
      #   st.write("\n")
      #   st.write("ADDRESS: {}".format(model.field_dict["add"]))
      # except:
      #   st.write("ERROR WHEN EXTRACT INFORMATION FROM IDCARD")
      #   pass

if __name__ == "__main__":
    # execute only if run as a script
    main()
    # cv2.imwrite('chien.png', test)