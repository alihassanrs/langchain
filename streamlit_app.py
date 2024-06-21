import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import PIL.Image
import os  
import tempfile 

load_dotenv()


def Image_Processing(text, image_path):
    llm = ChatGoogleGenerativeAI(model='gemini-pro-vision', api_key=os.getenv('GOOGLE_API_KEY'))
    message = HumanMessage(
        content=[
            {
                'type': 'text',
                'text': text,
            },
            {'type': 'image_url', 'image_url': image_path},
        ]
    )
    result = llm.invoke([message])
    return result.content


def main():

   


    st.title('LangChain Image Processing Model')
    user_input = st.text_input('Enter Text')
    image = st.file_uploader('Upload File', type=['jpg', 'png', 'jpeg'])
    submit = st.button('Ask Question')

    if submit and image is not None:  
        user_image = PIL.Image.open(image)
        st.image(user_image, caption='Image Uploaded', use_column_width=True)

    # Image Save            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            user_image.save(temp_file.name)
            image_path = temp_file.name



        response = Image_Processing(user_input, image_path) 
        st.subheader('Response')
        st.write(response)
        os.remove(image_path)


def Chat():
    st.title('LangChain Chat Model')
    user_input = st.text_input('Enter Text')
    submit = st.button('Ask Question')
    if submit:
        llm = ChatGoogleGenerativeAI(model='gemini-pro', api_key=os.getenv('GOOGLE_API_KEY'))
        response = llm.invoke(user_input)
        st.subheader('Response')
        st.write(response.content)



st.sidebar.title('Chose Langchain Model')
pages = st.sidebar.selectbox(
   
    "Chose Langchain Model",
    ("Chat Model", "Image Processing Model")
    )
st.sidebar.caption('Streamlit web application that integrates with LangChain and Google Generative AI for processing text alongside images')
st.sidebar.subheader('Follow Me')
st.sidebar.link_button("Connect LinkedIn",'https://www.linkedin.com/in/alihassanml')
st.sidebar.link_button("Connect On Github",'https://github.com/alihassanml',type="secondary")


if pages == 'Image Processing Model':
    main()

if pages == 'Chat Model':
    Chat()
