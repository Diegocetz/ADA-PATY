import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde el archivo ..env
load_dotenv()

# Configurar el token de API de Hugging Face Hub
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Verificar si el token de API está configurado

# Configurar la página de la aplicación
st.set_page_config(page_title="OpenAssistant Powered Chat App")

# Contenido de la barra lateral
with st.sidebar:
    st.title('🤗💬 HuggingChat App')
    st.markdown('''
    ## Acerca de
    Esta aplicación es un chatbot alimentado por LLM construido usando:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5](https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5) modelo LLM

    ''')
    add_vertical_space(3)
    st.write('Hecho con ❤️ por [Prompt Engineer](https://youtube.com/@engineerprompt)')

st.header("Hola soy tu asistente personal HuggingChat")


def main():
    # Generar listas vacías para las respuestas generadas y el usuario.
    ## Respuesta del Asistente
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hola, ¿En qué puedo ayudarte?"]

    ## Pregunta del usuario
    if 'user' not in st.session_state:
        st.session_state['user'] = ['¡Hola!']

    # Diseño de los contenedores de entrada/respuesta
    response_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    input_container = st.container()

    # Obtener la entrada del usuario
    def get_text():
        input_text = st.text_input("Tú: ", "", key="input")
        return input_text

    ## Aplicar el cuadro de entrada del usuario
    with input_container:
        user_input = get_text()

    def chain_setup():

        template = """{question}
        """

        prompt = PromptTemplate(template=template, input_variables=["question"])

        llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
                             model_kwargs={"max_new_tokens": 1200}, huggingfacehub_api_token=huggingfacehub_api_token)

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt
        )
        return llm_chain

    # Generar respuesta
    def generate_response(question, llm_chain):
        response = llm_chain.run(question)
        return response

    ## Cargar LLM
    llm_chain = chain_setup()

    # Bucle principal
    with response_container:
        if user_input:
            response = generate_response(user_input, llm_chain)
            st.session_state.user.append(user_input)
            st.session_state.generated.append(response)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['user'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))


if __name__ == '__main__':
    main()
