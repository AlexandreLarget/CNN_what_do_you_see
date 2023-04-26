import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

@st.cache_resource
def load_models():
    model1 = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    model2 = VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
    return model1, model2

model1, model2 = load_models()

layer_output1 = [layer.output for layer in model1.layers if "conv2" in layer.name]
MODEL_trained = tf.keras.Model(model1.input, layer_output1)

layer_output2 = [layer.output for layer in model2.layers if "conv2" in layer.name]
MODEL_raw = tf.keras.Model(model2.input, layer_output2)

def img_preprocessing(image):
  """
  Preprocess an image for our model by resizing it and normalizing it.
  """
  img = np.asarray(image)
  img = tf.image.resize(img, size=[224,224])
  img = img / 255.0
  img = tf.expand_dims(img, 0)
  return img

def get_matrix_list(num_layer, activation, n_plot):

    """
    :param num_layer: (int) which convolutional layer of the model is to target
    :param activation: (list) predict from the model
    :param n_plot: (int) number of plots to display
    :return: plots of images after been through the selected layer of the model.
    """
    # Create matrix and list of idx
    mat_idx = []
    mat_sum = []
    matrix = activation[num_layer][0]
    for i in range(len(matrix[0][0])):
        mat_idx.append(i)
        mat_sum.append(matrix[:,:,i].sum())
    df = pd.DataFrame([mat_idx, mat_sum], index=["mat_idx", "mat_sum"]).T

    # Select matrix with the highest sum
    list_mat = df.sort_values("mat_sum", ascending=False).head(n_plot).mat_idx.values.astype("int")
    return list_mat, matrix


with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("CNN, CNN, what do you see?")

st .markdown('''
            <br/>
            <p class='head'>This app will make you dive inside the "black box" of the VGG16. The VGG16 is a very popular convolutional neural network used for images recognition.<br/>
            It is composed of 5 convolutional blocks through which each image submitted will pass.<br/>
            Inside these blocks, each neuron of each layer will transform the image by slidding a kernel (or filter) over all the pixels of it.<br/>
            The neuron output image will be the result of these multiplications between the initial image and the kernel.</p>
            To use the app: 
            <ul>- upload an image you want</ul>
            <ul>- select the model you want to use between a VGG16 pretrained on "imagenet" and a VGG16 not trained</ul>
            <ul>- choose the number of plot you want to display per neuron (between 6 and 18)</ul>
            <ul>- hit the activation button</ul>
            <p class='head'>For each layer, you will see your image after it went through the referred convolutional block.<br/>
            The <b>aggregated matrix image</b> is an addition of all the matrices displayed by the convolutional block mentionned.</p>
            Enjoy !      
            <br/><br/>      
             ''', unsafe_allow_html=True)

with st.sidebar:
    image = st.file_uploader("Choose an image", type="jpg")
    model_selected = st.selectbox("model", ("VGG pretrained on imagenet", "VGG not trained"))
    num_plot = st.selectbox("number of plot per layer to display", (6, 12, 18))
    if image is not None:
        image = Image.open(image)

        # Display image
        col1, col2, col3 = st.columns([1,4,1]) #to center the image

        with col1:
            st.write(' ')

        with col2:
            img = np.array(image)
            st.image(img)

        with col3:
            st.write(' ')


        # Send image through the model
        processed_img = img_preprocessing(image)


    # Model selection
    if image and model_selected == "VGG pretrained on imagenet":
        activation = MODEL_trained.predict(processed_img)
    elif image:
        activation = MODEL_raw.predict(processed_img)

    activation_button = st.button("ACTIVATION", use_container_width=True)

if image and activation_button:
    for i in range(len(activation)):

        list_mat, matrix = get_matrix_list(num_layer=i, activation=activation, n_plot=num_plot)
        shift_idx = int(num_plot / 6) #Used to shift the index fromm the list_mat between each column

        # Subplots for each layers
        with st.expander(f"Image after been processed in the convolutional block {i+1}", expanded=True):
            st.subheader(f"Image shape: {matrix.shape[0]}x{matrix.shape[1]} | Number of images at this step: {matrix.shape[2]}")
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2,2,2,2,2,2,1,2])

            with col1:
                for i in list_mat[ : shift_idx]:
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(matrix[:,:,i])
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)

            with col2:
                for i in list_mat[shift_idx : 2* shift_idx]:
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(matrix[:,:,i])
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)


            with col3:
                for i in list_mat[2*shift_idx : 3* shift_idx]:
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(matrix[:, :, i])
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)

            with col4:
                for i in list_mat[3*shift_idx : 4* shift_idx]:
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(matrix[:, :, i])
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)

            with col5:
                for i in list_mat[4*shift_idx : 5* shift_idx]:
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(matrix[:, :, i])
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)

            with col6:
                for i in list_mat[5*shift_idx : 6* shift_idx]:
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(matrix[:, :, i])
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)

            with col7:
                st.write(' ')

            with col8:

                matrix_agg = np.zeros([matrix.shape[0], matrix.shape[1]])
                for mat in range(matrix.shape[2]):
                    matrix_agg += (matrix[:, :, mat]) ** 2
                matrix_agg *= 255.0 / matrix_agg.max()
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(matrix_agg)
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                st.markdown("<h4 class='subt'>Aggregated matrix</h4>", unsafe_allow_html=True)


    st.markdown(
        '''
        <p class="tail">
        Note the difference between the pretrained and not trained VGG16: <br/></p>
        <p class="head">
        On the 3 first blocks, the model that was not trained display images that were less modified, closer from the original one.<br/>
        It is because the kernels have issue to sort the information and prioritize the parts of the image that will allow the model to recognize identify it.<br/>
        On the last 2 blocks, this lake of prioritization leads to blurry results.<br/><br/>
        For the pretrained model on the opposite, the 2 first blocks quickly identify the sharps and edges of the image 
        which helps the 3 last blocks to focus on the most pertinent areas.<br/><br/>
        This allows us to conclude that a CNN trained with "imagenet" learns to identify patterns more than specific images.<br/>
        This is why we can use these pretrained models (transfer-learning) and still have great results with images they were not trained on.</p>
        <p class="ccl">
        <b>The model learns to see things in a way !</b> 
        </p> 
        ''', unsafe_allow_html=True)