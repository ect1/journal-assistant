import os
import tempfile
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

neon_db_password = st.secrets["neon_db_password"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]

st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.title("🦜 LangChain: Chat with Documents")


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = TextLoader(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    #print(docs)
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #embeddings = HuggingFaceHubEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = OpenAIEmbeddings()
    #vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    CONNECTION_STRING = f"postgresql://ect1:{neon_db_password}@ep-aged-wind-10740449.ap-southeast-1.aws.neon.tech/journal-vector?sslmode=require"
    COLLECTION_NAME = "journal-collection"

    vectordb = PGVector.from_documents(
        embedding=embeddings,
        documents=splits,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


#openai_api_key = os.
#if not openai_api_key:
#    st.info("Please add your OpenAI API key to continue.")
#    st.stop()

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["txt"], accept_multiple_files=True
)
#if not uploaded_files:
#    st.info("Please upload PDF documents to continue.")
#    st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

sys_prompt = """You are tasked with analyzing a dataset containing blood sugar level records of a pregnant patient. 
    Your goal is to extract insights from the data to assess the patient's glucose control and overall health during pregnancy. The analysis should focus on the following aspects:Identify patterns and trends in the blood sugar levels throughout the day.Assess the effectiveness of the medication and treatment plan in managing blood sugar levels.
    Evaluate the impact of diet, meal timing, and lifestyle factors on blood sugar control.Detect episodes of hypoglycemia or hyperglycemia and investigate potential contributing factors.Consider symptoms, observations, and other remarks provided to gain a comprehensive understanding of the patient's health status.
    Provide a detailed analysis based on the data, highlighting any noteworthy findings, trends, or areas for further investigation.
    --------------------
    {context}
    """

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

qa_chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(sys_prompt)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])