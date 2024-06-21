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

    st.title('LangChain Google Generative AI')
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


if __name__ == '__main__':
    main()
