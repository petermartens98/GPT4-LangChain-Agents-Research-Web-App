import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper 
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import YouTubeSearchTool
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from youtubesearchpython import VideosSearch
from langchain.chains import VectorDBQA
from langchain.retrievers import SelfQueryRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.tools import PubmedQueryRun
from langchain import LLMMathChain
import sqlite3
import pandas as pd
import os

# TODO: Allow users to upload their own files

def create_research_db():
    with sqlite3.connect('MASTER.db') as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Research (
                research_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                introduction TEXT,
                quant_facts TEXT,
                publications TEXT,
                books TEXT,
                ytlinks TEXT,
                prev_ai_research TEXT
            )
        """)


def create_messages_db():
    pass


def read_research_table():
    with sqlite3.connect('MASTER.db') as conn:
        query = "SELECT * FROM Research"
        df = pd.read_sql_query(query, conn)
    return df


def insert_research(user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research):
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Research (user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research))


def generate_research(userInput):
    global tools
    llm=OpenAI(temperature=0.7)
    wiki = WikipediaAPIWrapper()
    DDGsearch = DuckDuckGoSearchRun()
    YTsearch = YouTubeSearchTool()
    pubmed = PubmedQueryRun()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    tools = [
        Tool(
            name = "Wikipedia Research Tool",
            func=wiki.run,
            description="Useful for researching information on wikipedia"
        ),
        Tool(
            name = "Duck Duck Go Search Results Tool",
            func = DDGsearch.run,
            description="Useful for search for information on the internet"
        ),
        Tool(
            name = "YouTube Search Tool",
            func = YTsearch.run,
            description="Useful for gathering links on YouTube"
        ),
        Tool(
            name ='Calculator and Math Tool',
            func=llm_math_chain.run,
            description='Useful for mathematical questions and operations'
        ),
        Tool(
            name='Pubmed Science and Medical Journal Research Tool',
            func=pubmed.run,
            description='Useful for Pubmed science and medical research\nPubMed comprises more than 35 million citations for biomedical literature from MEDLINE, life science journals, and online books. Citations may include links to full text content from PubMed Central and publisher web sites.'

        )
    ]
    if st.session_state.embeddings_db:
        qa = VectorDBQA.from_chain_type(llm=llm,
                                        vectorstore=st.session_state.embeddings_db)
        tools.append(
            Tool(
                name='Vector-Based Previous Resarch Database Tool',
                func=qa.run,
                description='Provides access to previous research results'
            )
        )
         
        
    memory = ConversationBufferMemory(memory_key="chat_history")
    runAgent = initialize_agent(tools, 
                                llm, 
                                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                                verbose=True, 
                                memory=memory,
                                )

    with st.expander("Generative Results", expanded=True):
        st.subheader("User Input:")
        st.write(userInput)

        st.subheader("Introduction:")
        with st.spinner("Generating Introduction"):
            intro = runAgent(f'Write an academic introduction about {userInput}')
            st.write(intro['output'])

        st.subheader("Quantitative Facts:")
        with st.spinner("Generating Statistical Facts"):

            quantFacts = runAgent(f'''
                Considering user input: {userInput} and the intro paragraph: {intro} 
                \nGenerate a list of 3 to 5 quantitative facts about: {userInput}
                \nOnly return the list of quantitative facts
            ''')
            st.write(quantFacts['output'])

        prev_ai_research = ""
        if st.session_state.embeddings_db:
            st.subheader("Previous Related AI Research:")
            with st.spinner("Researching Pevious Research"):
                qa = VectorDBQA.from_chain_type(llm=llm,
                                                vectorstore=st.session_state.embeddings_db)
                prev_ai_research = qa.run(f'''
                    \nReferring to previous results and information, write about: {userInput}
                ''')
                st.write(prev_ai_research)

        st.subheader("Recent Publications:")
        with st.spinner("Generating Recent Publications"):
            papers = runAgent(f'''
                Consider user input: "{userInput}".
                \nConsider the intro paragraph: "{intro}",
                \nConsider these quantitative facts "{quantFacts}"
                \nNow Generate a list of 2 to 3 recent academic papers relating to {userInput}.
                \nInclude Titles, Links, Abstracts. 
            ''')
            st.write(papers['output'])

        st.subheader("Reccomended Books:")
        with st.spinner("Generating Reccomended Books"):
            readings = runAgent(f'''
                Consider user input: "{userInput}".
                \nConsider the intro paragraph: "{intro}",
                \nConsider these quantitative facts "{quantFacts}"
                \nNow Generate a list of 5 relevant books to read relating to {userInput}.
            ''')
            st.write(readings['output'])

        st.subheader("YouTube Links:")
        with st.spinner("Generating YouTube Links"):
            search = VideosSearch(userInput)
            ytlinks = ""
            for i in range(1,6):
                ytlinks += (str(i) + ". Title: " + search.result()['result'][0]['title'] + "Link: https://www.youtube.com/watch?v=" + search.result()['result'][0]['id']+"\n")
                search.next()
            st.write(ytlinks)

        # TODO: Influential Figures

        # TODO: AI Scientists Perscpective

        # TODO: AI Philosophers Perspective

        # TODO: Possible Routes for Original Research
        
        insert_research(userInput, intro['output'], quantFacts['output'], papers['output'], readings['output'], ytlinks, prev_ai_research)
        research_text = [userInput, intro['output'], quantFacts['output'], papers['output'], readings['output'], ytlinks, prev_ai_research]
        embedding_function = OpenAIEmbeddings()
        vectordb = Chroma.from_texts(research_text, embedding_function, persist_directory="./chroma_db")
        vectordb.persist()
        st.session_state.embeddings_db = vectordb


class Document:
    def __init__(self, content, topic):
        self.page_content = content
        self.metadata = {"Topic": topic}


def init_ses_states():
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("prev_chat_history", [])
    st.session_state.setdefault("embeddings_db", None)
    st.session_state.setdefault('research', None)
    st.session_state.setdefault("prev_research", None)
    st.session_state.setdefault("books", None)
    st.session_state.setdefault("prev_books", None)

def main():
    st.set_page_config(page_title="Research Bot")
    create_research_db()
    llm=OpenAI(temperature=0.7)
    embedding_function = OpenAIEmbeddings()
    init_ses_states()
    if os.path.exists("./chroma_db"):
        st.session_state.embeddings_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    st.header("GPT-4 LangChain Agents Research Bot")
    deploy_tab, prev_tab = st.tabs(["Generate Research","Previous Research"])
    with deploy_tab:
        userInput = st.text_area(label="User Input")
        if st.button("Generate Report") and userInput:
            generate_research(userInput)
        st.subheader("Chat with Data")
        user_message = st.text_input(label="User Message", key="um1")
        if st.button("Submit Message") and user_message:
            memory = ConversationBufferMemory(memory_key="chat_history")
            chatAgent = initialize_agent(tools, 
                                        llm, 
                                        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                                        verbose=True, 
                                        memory=memory,
                                        )
    with prev_tab:
        st.dataframe(read_research_table())
        selected_input = st.selectbox(label="Previous User Inputs",options=[i for i in read_research_table().user_input])
        if st.button("Render Research") and selected_input:
            with st.expander("Rendered Previous Research",expanded=True):
                selected_df = read_research_table()
                selected_df = selected_df[selected_df.user_input == selected_input].reset_index(drop=True)
                
                st.subheader("User Input:")
                st.write(selected_df.user_input[0])

                st.subheader("Introduction:")
                st.write(selected_df.introduction[0])

                st.subheader("Quantitative Facts:")
                st.write(selected_df.quant_facts[0])

                st.subheader("Previous Related AI Research:")
                st.write(selected_df.prev_ai_research[0])

                st.subheader("Recent Publications:")
                st.write(selected_df.publications[0])

                st.subheader("Recommended Books:")
                st.write(selected_df.books[0])

                st.subheader("YouTube Links:")
                st.write(selected_df.ytlinks[0])

            st.subheader("Chat with Data")
            prev_user_message = st.text_input(label="User Message", key="um2")
            

if __name__ == '__main__':
    load_dotenv()
    main()

