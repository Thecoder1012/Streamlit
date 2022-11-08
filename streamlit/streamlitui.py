"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st

#set background

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://us.123rf.com/450wm/apostrophe/apostrophe1711/apostrophe171100111/90402746-sfondo-blu-scuro-con-trama-blu-solido.jpg");
             background-attachment: fixed;
             background-size: cover;
             height:100%;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

    
add_bg_from_url() 
   
# set title of app
st.title("Simple Image Classification Application")
st.write("")

with open( "./app/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

# enable users to upload images for the model to make predictions
upload_title = '<p style="font-family:Courier; color:Red; font-size: 40px;"><b>Upload an Image</b></p>'
st.markdown(upload_title, unsafe_allow_html=True)
file_up = st.file_uploader("", type = "jpg")

option = st.selectbox('Which option do you like?', ('Preprocessing', 'Testing'))
st.write('You selected:', option)

def predict(image):
    """Return top 5 predictions ranked by highest probability.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # create a ResNet model
    resnet = models.resnet101(pretrained = True)

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )])

    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    out = resnet(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)        
    if option == 'Testing':
        # display image that user uploaded
        st.write("Just a second ...")
        labels = predict(file_up)
        for i in labels:
            st.write('<p style=color:#ffffff; font-size: 20px;>Prediction (index, name)"', i[0], ",   Score: ", i[1],'</p>', unsafe_allow_html=True)
