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
from youtubesearchpython import VideosSearch
import sqlite3
import pandas as pd
import os


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
                ytlinks TEXT
            )
        """)


def read_research_table():
    with sqlite3.connect('MASTER.db') as conn:
        query = "SELECT * FROM Research"
        df = pd.read_sql_query(query, conn)
    return df


def insert_research(user_input, introduction, quant_facts, publications, books, ytlinks):
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Research (user_input, introduction, quant_facts, publications, books, ytlinks)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_input, introduction, quant_facts, publications, books, ytlinks))


def generate_research(userInput):
    wiki = WikipediaAPIWrapper()
    DDGsearch = DuckDuckGoSearchRun()
    YTsearch = YouTubeSearchTool()
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
        )
    ]
    if st.session_state.embeddings_db:
        tools.append(
            Tool(
                name = 'Previous Research Chroma Database Tool',
                func = st.session_state.embeddings_db.similarity_search(userInput),
                description="Useful for looking up similar previous research/information in chromaDB"
            )
        )
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm=OpenAI(temperature=TEMP)
    runAgent = initialize_agent(tools, 
                                llm, 
                                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                                verbose=True, 
                                memory=memory)

    with st.expander("Generative Results", expanded=True):
        st.subheader("User Input:")
        st.write(userInput)

        # ToDo
        # Determine Category
        # Image Generation
        # Store Image?

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
        
        insert_research(userInput, 
                        intro['output'], 
                        quantFacts['output'], 
                        papers['output'], 
                        readings['output'], 
                        ytlinks)


class Document:
    def __init__(self, content, topic):
        self.page_content = content
        self.metadata = {"Topic": topic}


def main():
    global TEMP
    st.set_page_config(page_title="Research Bot")
    create_research_db()
    embedding_function = OpenAIEmbeddings()
    st.session_state.setdefault("embeddings_db", None)
    if os.path.exists("/chroma_db"):
        st.session_state.embeddings_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    st.header("GPT-4 LangChain Agents Research Bot")
    st.caption("Powered by OpenAI, LangChain, ChromaDB, SQLite, YouTube, Wikepedia, DuckDuckGo, Streamlit")
    with st.sidebar:
        with st.expander("LLM Settings", expanded=True):
            TEMP = st.slider("Temperature",0.0,1.0,0.65)
    deploy_tab, prev_tab = st.tabs(["Generate Research","Previous Research"])
    with deploy_tab:
        userInput = st.text_area(label="User Input")
        if st.button("Generate Report") and userInput:
            generate_research(userInput)
            research_df = read_research_table().tail(1)
            documents = research_df.apply(lambda row: Document(' '.join([f'{idx}: {val}' for idx, val in zip(row.index, row.values.astype(str))]), row['user_input']), axis=1).tolist()
            #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            embeddings_db = Chroma.from_documents(documents, embedding_function, persist_directory="./chroma_db")
            embeddings_db.persist()
            st.session_state.embeddings_db = embeddings_db
    with prev_tab:
        st.dataframe(read_research_table())
        selected_input = st.selectbox(label="Previous User Inputs",options=[i for i in read_research_table().user_input])
        if st.button("Render Research") and selected_input:
            with st.expander("Rendered Previous Research",expanded=True):
                selected_df = read_research_table()
                selected_df = selected_df[selected_df.user_input == selected_input]
                st.subheader("User Input:")
                st.write(selected_df.user_input[0])
                st.subheader("Introduction:")
                st.write(selected_df.introduction[0])
                st.subheader("Quantitative Facts:")
                st.write(selected_df.quant_facts[0])
                st.subheader("Recent Publications:")
                st.write(selected_df.publications[0])
                st.subheader("Recommended Books:")
                st.write(selected_df.books[0])
                st.subheader("YouTube Links:")
                st.write(selected_df.ytlinks[0])
            

if __name__ == '__main__':
    load_dotenv()
    main()

