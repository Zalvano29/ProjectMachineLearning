import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Beranda","Tentang Kami","Pengenalan Penyakit"])

#Main Page
if(app_mode=="Beranda"):
    st.header("SISTEM PENGENALAN PENYAKIT TANAMAN APEL")
    image_path = "home_page.jpeg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Selamat datang di Sistem Pengenalan Penyakit Tanaman Apel! 🌿🔍
    
    Misi kami adalah membantu mengidentifikasi penyakit apel secara efisien. Unggah gambar daun pohon apel, dan sistem kami akan menganalisisnya untuk mendeteksi tanda-tanda penyakit. Bersama-sama, mari kita lindungi tanaman kita dan pastikan panen yang lebih sehat!

    ### Cara Kerjanya
    1. **Unggah Gambar:** Buka halaman **Pengenalan Penyakit** dan unggah gambar apel yang diduga memiliki penyakit.
    2. **Analisis:** Sistem kami akan memproses gambar menggunakan algoritma canggih untuk mengidentifikasi potensi penyakit.
    3. **Hasil:** Lihat hasil.

    ### Mengapa Memilih Kami?
    - **Akurasi:** Sistem kami menggunakan teknik pembelajaran mesin canggih untuk deteksi penyakit yang akurat.
    - **Ramah Pengguna:** Antarmuka yang sederhana dan intuitif untuk pengalaman pengguna yang lancar.
    - **Cepat dan Efisien:** Terima hasil dalam hitungan detik, yang memungkinkan pengambilan keputusan yang cepat.

    ### Mulai
    Klik halaman **Pengenalan Penyakit** di bilah sisi untuk mengunggah gambar dan rasakan kekuatan Sistem Pengenalan Penyakit Apel kami!

    ### Tentang Kami
    Pelajari lebih lanjut tentang proyek, tim kami, dan tujuan kami di halaman **Tentang**.
    """)

#About Project
elif(app_mode=="Tentang Kami"):
    st.header("Tentang Kami")
    st.markdown("""
                #### Tentang Dataset
                Dataset ini dibuat ulang menggunakan augmentasi offline dari dataset asli.
                Dataset ini terdiri dari sekitar 3171 gambar rgb daun pohon apel yang sehat dan sakit yang dikategorikan ke dalam 4 kelas berbeda.
                Direktori baru yang berisi 4 gambar uji dibuat kemudian untuk tujuan prediksi.
                #### Konten
                1. train (2536 gambar)
                2. test (4 gambar)
                3. validation (635 gambar)
                """)

#Prediction Page
elif(app_mode=="Pengenalan Penyakit"):
    st.header("Pengenalan Penyakit")
    test_image = st.file_uploader("Pilih gambar:")
    if(st.button("Lihat gambar")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Prediksi")):
        st.snow()
        st.write("Prediksi Kami")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']
        st.success("Model memprediksi {}".format(class_name[result_index]))