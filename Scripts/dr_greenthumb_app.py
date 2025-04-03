from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st
import re
from keras.models import load_model
import base64
import numpy as np
import os
import cv2
import tensorflow as tf
import streamlit as st



# Avoid OOM errors by setting GPU Memory Consumption Growth

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    except RuntimeError as e:
        print(e)


# llm
@st.cache_resource
def init_llm():
  return LlamaCpp(model_path = r"C:\Users\Marvin\Documents\WBS\Final_Project\LLM\mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                  max_tokens = 1000,
                  temperature = 0.1,
                  top_p = 0.9,
                  n_gpu_layers = 15,
                  n_ctx = 2048)
llm = init_llm()


# embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = r"C:\Users\Marvin\Documents\WBS\Final_Project\LLM\embeddings"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)


# # load vector database
# vector_db = Chroma(persist_directory=r"C:\Users\Marvin\Documents\WBS\Final_Project\LLM\vector_db\chroma_db", embedding_function=embeddings)

vector_db = Chroma(
    persist_directory=r"C:\Users\Marvin\Documents\WBS\Final_Project\LLM\vector_db\chroma_db",
    embedding_function=embeddings,
)

# retriever
retriever = vector_db.as_retriever(search_type="mmr",search_kwargs={"k": 2, "reduction_factor": 0.2})


# memory
@st.cache_resource
def init_memory(_llm):
    return ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
memory = init_memory(llm)

# prompt
template = """
<s> [INST]
You are polite and professional question-answering AI assistant. You must provide a helpful response to the user. You are a doctor who has a sense of humor, your goal is to provide helpful and precise responses and solutions to users.

In your response, PLEASE ALWAYS:
  (0) Be a detail-oriented reader: read the question and context and understand both before answering
  (1) Start your answer with a friendly, polite and funny tone, and reiterate the question so the user is sure you understood it
  (2) Provide a detailed and understandable answer with a touch of humor if appropriate. If you can't find the answer, respond with a funny explanation, starting with: "I think the answer is playing hide-and-seek in my clinic!".
  (3) Ensure your answer addresses the question and offers practical solutions.
  (4) Incorporate humor to keep the conversation engaging and enjoyable.
[/INST]
[INST]
Answer the following question using the context provided.
The question is surrounded by the tags <q> </q>.
The context is surrounded by the tags <c> </c>.
<q>
{question}
</q>
<c>
{context}
</c>
[/INST]
</s>
[INST]
Helpful Answer:
[INST]
"""
prompt = PromptTemplate(template=template, input_variables=["question", "context"])


# chain
chain = ConversationalRetrievalChain.from_llm(llm,
                                              retriever=retriever,
                                              memory=memory,
                                              return_source_documents=True,
                                              combine_docs_chain_kwargs={"prompt": prompt})


################
# Streamlit app
################

st.title("Welcome to Dr. Greenthumb's clinic!")

# Initialise chat history
# Chat history saves the previous messages to be displayed
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Dr. Greenthumb: How can I help you?"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Dr. Greenthumb is on the case. Check back for the solution!"):
        try:
            # Assuming chain is defined somewhere else in your code
            answer = chain(prompt)
            response = answer["answer"]
            #Clean up response by removing HTML-like tags
            clean_response = re.sub(r'<.*?>', '', response)
            # Display chatbot response in chat message container
            with st.chat_message("assistant"):
            #st.markdown(answer["answer"])
                st.markdown(clean_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            # Handle the error gracefully
            st.error(f"An error occurred: {e}")

#############################


# Update the image_path variable with the correct path to your image file
image_path = r'C:\Users\Marvin\Documents\WBS\Final_Project\Data\streamlit_upload_pics\dr_greenthumb.png'

# Read the image file and encode it to base64
try:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    # Display the image in Streamlit sidebar using HTML
    st.sidebar.markdown(
        f"""
        <div style="display: table; margin-top: -5%; margin-left: 10%;">
            <img src="data:image/png;base64,{data}" width="256" height="256">
        </div>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.error("Image file not found. Please check the file path.")


# # Update the image_path variable with the correct path to your image file
# image_path = r'C:\Users\Marvin\Documents\WBS\Final_Project\Data\streamlit_upload_pics\logo.png'

# # Read the image file and encode it to base64
# try:
#     with open(image_path, "rb") as f:
#         data = base64.b64encode(f.read()).decode("utf-8")

#     # Display the image in Streamlit sidebar using HTML
#     st.sidebar.markdown(
#         f"""
#         <div style="position: fixed; bottom: 10px; left: 10px;">
#             <img src="data:image/png;base64,{data}" width="400" height="100">
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
# except FileNotFoundError:
#     st.error("Image file not found. Please check the file path.")


##########################################################


# Load the Keras model
model = load_model(os.path.join(r'C:\Users\Marvin\Documents\WBS\Final_Project\Dr.-Greenthumb-Decoding-Nature-s-Needs\Models','multi_diseases_classifier_tomato_98.h5'))

# Define the class names based on your model
# class_names = [
#     'Tomato suffering of bacterial spots',
#     'Tomato suffering of early blight',
#     'This tomato looks healthy',
#     'Tomato suffering of late blight',
#     'Your Tomato has some leaf curl',
#     'Tomato suffering of leaf mold',
#     'Tomato suffering of mosaic virus... oh oh!',
#     'Tomato suffering of septoria leaf spots',
#     'Tomato infested by spider mites',
#     'Tomato with target spots'
# ]

class_names = [
    'Tomato bacterial spots',
    'Tomato early blight',
    'Tomato healthy',
    'Tomato late blight',
    'Tomato leaf curl',
    'Tomato leaf mold',
    'Tomato mosaic virus',
    'Tomato septoria leaf spots',
    'Tomato spider mites',
    'Tomato target spots'
]

# Allow users to upload an image
uploaded_file = st.sidebar.file_uploader("Dr. Greenthumb: Upload or Drag-and-Drop Image Here", type=["jpg", "jpeg", "png"])

# Process the uploaded image and display it
if uploaded_file is not None:
    try:
        # Read the image file
        img_bytes = uploaded_file.read()

        # Encode the image bytes to base64
        data = base64.b64encode(img_bytes).decode("utf-8")

        # Display the image in Streamlit sidebar using HTML
        st.sidebar.markdown(
            f"""
            <div style="display: table; margin-top: 10px; margin-left: auto; margin-right: auto;">
                <img src="data:image/png;base64,{data}" alt="Uploaded Image" style="max-width: 100%; height: auto;">
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Convert the image bytes to a numpy array
        img_array = np.frombuffer(img_bytes, np.uint8)

        # Decode the image data into an OpenCV format
        img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Resize the image to match model input shape (assuming your model expects (256, 256) input)
        img_resized = tf.image.resize(img_cv2, (256, 256))

        # Normalize pixel values (assuming model was trained with normalized data)
        img_normalized = img_resized / 255.0

        # Add batch dimension
        img_input = np.expand_dims(img_normalized, axis=0)

        # Perform prediction
        prediction_array = model.predict(img_input)

        # Get prediction probabilities for each class
        prediction_probabilities = prediction_array[0]

        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(prediction_probabilities)

        # Get the predicted class label
        predicted_class_label = class_names[predicted_class_index-1]

        # Get the probability of the predicted class
        predicted_class_probability = prediction_probabilities[predicted_class_index]

        # Define a threshold for confidence
        confidence_threshold = 0.99  # You can adjust this threshold as needed

        # Display initial response based on the image
        if predicted_class_probability >= confidence_threshold:
            initial_response = f"Based on the uploaded image, I detected: {predicted_class_label}"
            st.chat_message("assistant").markdown(initial_response)
            st.session_state.messages.append({"role": "assistant", "content": initial_response})

            # Process user input with the conversational retrieval chain
            with st.spinner("Dr. Greenthumb is on the case. Check back for the solution!"):
                prompt = f"What you can do against {initial_response}: "
                answer = chain(prompt)  # Send question to the conversational retrieval chain
                response = answer["answer"]  # Extract answer from the response dictionary

                # Display chatbot response for the treatment recommendation
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # If the predicted probability is low, indicate uncertainty
            initial_response = "Hmm... that should be a plant? Please try again."
            st.chat_message("assistant").markdown(initial_response)
            st.session_state.messages.append({"role": "assistant", "content": initial_response})

    except Exception as e:
        st.error(f"Error processing image: {e}")


######WEBCAM###

# Prompt the user to use their camera and capture an image
user_prompt = "Please use your camera to capture an image."
captured_image = st.sidebar.camera_input(user_prompt)

# Process the captured image and make predictions
if captured_image is not None:
    try:
       
        # Convert captured image data to bytes
        img_bytes = captured_image.getvalue()

        # Decode the image data into an OpenCV format
        img_cv2 = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Resize the image to match model input shape (assuming your model expects (256, 256) input)
        img_resized = tf.image.resize(img_cv2, (256, 256))

        # Normalize pixel values (assuming model was trained with normalized data)
        img_normalized = img_resized / 255.0

        # Add batch dimension
        img_input = np.expand_dims(img_normalized, axis=0)
        
        # Perform prediction
        prediction_array = model.predict(img_input)

        # Get prediction probabilities for each class
        prediction_probabilities = prediction_array[0]

        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(prediction_probabilities)

        # Get the predicted class label
        predicted_class_label = class_names[predicted_class_index-1]

        # Get the probability of the predicted class
        predicted_class_probability = prediction_probabilities[predicted_class_index]

        # Define a threshold for confidence
        confidence_threshold = 0.99  # You can adjust this threshold as needed

        # Display initial response based on the image
        if predicted_class_probability >= confidence_threshold:
            initial_response = f"Based on the uploaded image, I detected: {predicted_class_label}"
            st.write(initial_response)

            # If you want to integrate this with the conversational retrieval chain, provide the user's prompt
            with st.spinner("Dr. Greenthumb is on the case. Check back for the solution!"):
                prompt = f"What you can do against {initial_response}: "
                answer = chain(prompt)  # Send question to the conversational retrieval chain
                response = answer["answer"]  # Extract answer from the response dictionary
                st.write(response)
        else:
            # If the predicted probability is low, indicate uncertainty
            initial_response = "Hmm... that should be a plant? Please try again."

    except Exception as e:
        st.error(f"Error processing image: {e}")

st.stop()