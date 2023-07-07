import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import tensorflow as tf
from vit.model import ViT, ViTBase, ViTHuge, ViTLarge
import numpy as np
import shutil

import os # inbuilt module
import random # inbuilt module
import webbrowser # inbuilt module

#=================================== Title ===============================
st.title("""
Cat 🐱 Or Dog 🐶
	""")

#================================= Title Image ===========================
st.text("""""")
img_path_list = ["static/image_1.jpg",
				"static/image_2.jpg"]
index = random.choice([0,1])
image = Image.open(img_path_list[index])
st.image(
	        image,
	        use_column_width=True,
	    )

#================================= About =================================
st.write("""
## 1️⃣ GIỚI THIỆU 
	""")
st.write("""
         Xin chào, chào mừng bạn đến với web của chúng tôi.
         Đây là một trang web sẽ giúp bạn phân loại chó hay mèo thông qua hình ảnh mà bạn đưa vào.
	""")
st.write("""
         Bạn chỉ cần đơn giản là upload file lên web của chúng tôi, 
sau đó hệ thống sẽ trả về kết quả cho bạn hình ảnh mà bạn đưa vào là hình ảnh chó hay mèo.
	""")

st.write("""
#### ©️ SẢN PHẨM ĐƯỢC PHÁT TRIỂN BỞI NHÓM 12
	""")
st.text("""20120061	Phạm Dương Trường Đức""")
st.text("""20120210	Trần Thị Kim Tiến""")
st.text("""20120238	Nguyễn Ngọc Khánh Vy""")
st.text("""20120307	Phạm Gia Khiêm""")
st.text("""20120328	Hoàng Đức Nhật Minh""")

#============================ How To Use It ===============================
st.write("""
## 2️⃣ HƯỚNG DẪN SỬ DỤNG
	""")
st.write("""
Để có thể sử dụng trang web này, bạn chỉ cần các thao tác đơn giản như sau:
- Trước tiên, bạn phải xác định rõ hình ảnh mà bạn upload lên bắt buộc phải chỉ có chó hay chỉ có mèo, không được có cả chó và mèo trong cùng một bức ảnh. Nếu không kết quả sẽ không chính xác.
- Tiếp theo, sau khi có hình ảnh chó hoặc mèo, bạn chỉ cần chọn nút **"Browse files"** hoặc kéo thả hình ảnh vào khung có dòng chữ **"Drag and drop file here"**.
- Hình ảnh mà bạn vừa upload lên sẽ được hiện trên màn hình. 
- Cuối cùng, nhấn nút **👉🏼 Predict** để xem kết quả.

🔘 **CHÚ Ý :** *Nếu bạn upload nhiều hơn 1 file hình ảnh, thì nó sẽ xảy ra lỗi nếu bạn nhấn vào nút* **👉🏼 Predict**
	""")

#========================= What It Will Predict ===========================
st.write("""
## 3️⃣ THỰC HIỆN CHƯƠNG TRÌNH
	""")
st.write("""
         Chương trình sẽ thực hiện phân loại hình ảnh chó 🐕 hay mèo 🐈 mà bạn upload lên trang web.
	""")

#======================== Time To See The Magic ===========================
st.write("""
## 👁️‍🗨️ Time To See The Magic 🌀
	""")

#========================== File Uploader ===================================
img_file_buffer = st.file_uploader("Upload an image here 👇🏻")

try:
	image = Image.open(img_file_buffer)
	img_array = np.array(image)
	st.write("""
		Preview 👀 Of Given Image!
		""")
	if image is not None:
	    st.image(
	        image,
	        use_column_width=True
	    )
	st.write("""
		Bây giờ, chỉ còn bước cuối cùng là phân loại xem đây là hình ảnh chó hay mèo.
		""")
	st.write("""
        **Chỉ cần nhấn vào nút '👉🏼 Predict' thì kết quả phân loại sẽ hiện lên màn hình**
		""")
except:
	st.write("""
		### ❗ Không có hình ảnh nào được chọn!!! Mời chọn hình ảnh!!!
		""")

#================================= Predict Button ============================
st.text("""""")
submit = st.button("👉🏼 Predict")

#==================================== Model ==================================
def generate_result(predictions, prediction):
    st.write("""
             ## 🎯 RESULT
             """)
    predict = np.round(predictions[0][prediction] * 100, 2)
    if prediction == 0:
        st.write(f"""
	    	## Chương trình của chúng tôi dự đoán {predict}% đây là **MỘT CHÚ MÈO 🐱**!!!
	    	""")
    else:
        st.write(f"""
	    	## Chương trình của chúng tôi dự đoán {predict}% đây là **MỘT CHÚ CHÓ 🐶**!!!
	    	""")

#=========================== Predict Button Clicked ==========================
if submit:
	try:
		# save image on that directory
		save_img("temp_dir/test_image.jpg", img_array)
		
		image_path = "temp_dir/test_image.jpg"
		# Predicting
		st.write("👁️ Predicting...")

		# Loading Model
		model_folder = './model/'
		
		model = load_model(model_folder, compile=False)
		# Loading image
		test_image = load_img(image_path, target_size = (150, 150)) 
		test_image = img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		
		# Prediction
		predictions = model.predict(test_image)
		prediction = np.argmax(predictions)		
  
		generate_result(predictions, prediction)

	except:
		st.write("""
		### ❗ Oops... Something Is Going Wrong
			""")

#=============================== Copy Right ==============================
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")