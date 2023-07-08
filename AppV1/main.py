import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper 
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import YouTubeSearchTool
from youtubesearchpython import VideosSearch
import requests
import urllib.parse
from bs4 import BeautifulSoup


def generateYouTubeLinks(userInput):
    search = VideosSearch(userInput)
    for i in range(1,6):
        st.caption(str(i) + ". Title: " + search.result()['result'][0]['title'] + "Link: https://www.youtube.com/watch?v=" + search.result()['result'][0]['id'])
        st.write()
        search.next()


def generate_research(userInput):
    wiki = WikipediaAPIWrapper()
    DDGsearch = DuckDuckGoSearchRun()
    YTsearch = YouTubeSearchTool()
    tools = [
        Tool(
            name = "Wikipedia Research Tool",
            func=wiki.run,
            description="Useful for looking up information on wikipedia"
        ),
        Tool(
            name = "Duck Duck Go Search Results Tool",
            func = DDGsearch.run,
            description="Useful for looking up information on the internet"
        ),
        Tool(
            name = "YouTube Search Tool",
            func = YTsearch.run,
            description="Useful for looking up links on YouTube"
        )
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm=OpenAI(temperature=TEMP)
    runAgent = initialize_agent(tools, 
                                llm, 
                                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                                verbose=True, 
                                memory=memory)

    with st.expander("Generative Results", expanded=True):
        st.write(userInput)

        st.write("Introduction:")
        with st.spinner("Generating Introduction"):
            intro = runAgent(f'Write an academic introduction about {userInput}')
            st.caption(intro['output'])

        st.write("Quantitative Facts:")
        with st.spinner("Generating Statistical Facts"):
            quantFacts = runAgent(f'''
                Considering user input: {userInput} and the intro paragraph: {intro}, 
                Generate only a list of 5 statistical and numerical facts about: {userInput}
            ''')
            st.caption(quantFacts['output'])

        st.write("Recent Publications:")
        with st.spinner("Generating Recent Publications"):
            papers = runAgent(f'''
                Consider user input: "{userInput}".
                \nConsider the intro paragraph: "{intro}",
                \nConsider these quantitative facts "{quantFacts}"
                \nNow Generate a list of 2 to 3 recent academic papers relating to {userInput}.
                \nInclude Titles, Links, Abstracts. 
            ''')
            st.caption(papers['output'])

        st.write("Reccomended Books:")
        with st.spinner("Generating Reccomended Books"):
            readings = runAgent(f'''
                Consider user input: "{userInput}".
                \nConsider the intro paragraph: "{intro}",
                \nConsider these quantitative facts "{quantFacts}"
                \nNow Generate a list of 5 relevant books to read relating to {userInput}.
            ''')
            st.caption(readings['output'])

        st.write("YouTube Links")
        with st.spinner("Generating YouTube Links"):
            generateYouTubeLinks(userInput)
        


def main():
    global TEMP
    st.set_page_config(page_title="Research Bot")
    st.header("GPT-4 LangChain Agents Research Bot")
    st.caption("Powered by OpenAI, LangChain, YouTube, Wikepedia, DuckDuckGo, Streamlit")
    with st.sidebar:
        with st.expander("Research Prompt",expanded=True):
            userInput = st.text_area(label="User Input")
        with st.expander("LLM Settings", expanded=True):
            TEMP = st.slider("Temperature",0.0,1.0,0.65)
    if st.button("Generate Report") and userInput:
        generate_research(userInput)


if __name__ == '__main__':
    load_dotenv()
    main()

