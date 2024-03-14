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

st.set_page_config(page_title="Chat with Documents", page_icon="🦜")
st.title("🦜 LangChain: Chat with Documents")


#@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files, collection_persona, pre_delete_collection=False):
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

    default_collection_name = "journal-collection"

    COLLECTION_NAME = default_collection_name

    if collection_persona == "Budget Secretary":
        COLLECTION_NAME = "budget-secretary-collection"
    elif collection_persona == "Crypto Buddy":
        COLLECTION_NAME = "crypto-buddy-collection"

    print(COLLECTION_NAME)

    vectordb = PGVector.from_documents(
        embedding=embeddings,
        documents=splits,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=pre_delete_collection
    )

    # Define retriever
    #retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 1, "fetch_k": 4})
    retriever = vectordb.as_retriever(search_type="similarity")

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
        
collection_persona = st.sidebar.selectbox('Collection/Persona',
    ('OB', 'Budget Secretary', 'Crypto Buddy'))

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF filesss", type=["txt"], accept_multiple_files=True
)

pre_delete_collection = st.sidebar.checkbox("Pre-delete collection")

# "gpt-3.5-turbo"
# gpt-4-0125-preview

model_name = 'gpt-3.5-turbo'
if st.sidebar.checkbox('GPT4'):
    model_name = 'gpt-4-0125-preview'



print(model_name)


if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

go = st.sidebar.button('Go', on_click=click_button)


if not st.session_state.clicked:
    st.stop()
#if not uploaded_files:
#    st.info("Please upload PDF documents to continue.")
#    st.stop()

retriever = configure_retriever(uploaded_files, collection_persona, pre_delete_collection)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

if go:
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

sys_prompt_ob = """You are tasked with analyzing a dataset containing blood sugar level records of a pregnant patient. 
    Your goal is to extract insights from the data to assess the patient's glucose control and overall health during pregnancy. The analysis should focus on the following aspects:Identify patterns and trends in the blood sugar levels throughout the day.Assess the effectiveness of the medication and treatment plan in managing blood sugar levels.
    Evaluate the impact of diet, meal timing, and lifestyle factors on blood sugar control.Detect episodes of hypoglycemia or hyperglycemia and investigate potential contributing factors.Consider symptoms, observations, and other remarks provided to gain a comprehensive understanding of the patient's health status.
    Provide a detailed analysis based on the data, highlighting any noteworthy findings, trends, or areas for further investigation.

    I may ask for data either specific data or all data, show it in tabular form.

    --------------------
    Answer the question based only on the following context:
    {context}
    """

sys_prompt_budget = """"
    You are a helpful budget assistant. Your goal is to track my daily expenses, including food, transportation, bills, rent, utilities, and unexpected expenditures like unplanned hangouts with friends or impulsive gadget purchases. Additionally, I would like you to generate reports, especially detailing the overall expenses on a weekly and monthly basis.

    I may also ask you to show the data in daily, weekly, monthly and yearly basis. show them in tabular form with total expenses in row and column.
    --------------------
    Answer the question based only on the following context:
    {context}
    """
sys_prompt_crypto_buddy = """
    
    You're tasked with analyzing a cryptocurrency trader's trade journal to provide insights and guidance. Each entry in the journal contains:
    1. Trade rationale: Explanation of the reasoning behind each trade.
    2. Risk management: Details on strategies used to manage risk.
    3. Trade execution: Information on entry and exit points, timing, and execution challenges.
    4. Performance evaluation: Outcomes of each trade and assessment of strategy effectiveness.
    5. Emotional and psychological factors: Insights into emotional challenges and coping mechanisms.
    6. Learning and improvement: Reflection on lessons learned and areas for development.
    7. Future planning: Discussion of plans for future trades and strategy adjustments.
    Your objective is to analyze the data and offer recommendations for improvement. Provide a detailed analysis highlighting patterns, trends, and areas for optimization to enhance trading performance.
    You have 2 tasks.
    1. Remind me when I'm going to enter a trade about the last performance of that coin, remind me of my rules, and let me know if I'm in good physical and emotional condition.
    2. Analyze based on the dataset. These are the fields of the dataset below:
    Date, Coin, Long/Short, Confluences,  Entry reason,  Exit reason, spot/derivative, Strategy, Entry Price, Exit Price, Quantity, Amount, Duration, pnl


    --------------------
    Answer the question based only on the following context:
    Question: {question}

    Context: {context}

    Answer:
    """

sys_prompt = sys_prompt_ob

if collection_persona == "Budget Secretary":
    sys_prompt = sys_prompt_budget
elif collection_persona == "Crypto Buddy":
    sys_prompt = sys_prompt_crypto_buddy

# print(sys_prompt)

# "gpt-3.5-turbo"
# gpt-4-0125-preview
# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name=model_name, temperature=0, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True, chain_type="stuff"
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
