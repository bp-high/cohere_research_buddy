from typing import List
import requests
import streamlit as st
import cohere
from llmsherpa.readers import LayoutPDFReader
from llama_index import StorageContext, load_index_from_storage
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.llms import LiteLLM
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.readers.schema.base import Document
from url_processor import url_processor
from read_llm_sherpa_pdf import get_vector_index
import modal
import os


llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
MODERATION_THRESHOLD = st.secrets.MODERATION_THRESHOLD
COHERE_API_KEY = st.secrets.COHERE_API_KEY
S2_API_KEY = st.secrets.S2_API_KEY

llm = LiteLLM("command-nightly")
embed_model = CohereEmbedding(
    cohere_api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_query",
)
cohere_rerank = CohereRerank(api_key=COHERE_API_KEY, top_n=5)
st.set_page_config(page_title="Research Buddy: Insights and Q&A on AI Research Papers using Cohere", page_icon="🧐", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title(body="AI Research Copilot 📚🤖")


def initialize_session_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about the research paper"}
        ]

    if "paper_content" not in st.session_state:
        st.session_state.paper_content = None

    if "paper_insights" not in st.session_state:
        st.session_state.paper_insights = None


initialize_session_state()

st.header('Read with AI powered by Cohere', divider='rainbow')
st.info("""This Application currently only works with arxiv and acl anthology web links which belong to the format:- 

    1) Arxiv:- https://arxiv.org/abs/paper_unique_identifier

    2) ACL Anthology:- https://aclanthology.org/paper_unique_identifier/ 
    """, icon="ℹ️")
user_input_paper = st.text_input("Enter the arxiv or acl anthology url of the paper",
                           "https://arxiv.org/abs/2310.16787", key="input2")


def reset_conversation():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the research paper"}
    ]


def get_sherpa_paper_content(url: str) -> List:
    with st.spinner(text="Using LLM Sherpa's LayoutPDFReader to read the paper contents"):
        try:
            pdf_path = url_processor(url=url)
            pdf_reader = LayoutPDFReader(llmsherpa_api_url)
            doc = pdf_reader.read_pdf(path_or_url=pdf_path)
            internal_chunks_list = []
            for chunk in doc.chunks():
                internal_chunks_list.append(chunk.to_context_text())

            return internal_chunks_list

        except Exception as e:
            return []


if st.button("Chat"):
    chunks_list = get_sherpa_paper_content(url=user_input_paper)
    if chunks_list:
        with st.spinner(text="Indexing the documents"):
            index_dir = get_vector_index(chunks_list=chunks_list)
            if index_dir != "Error":
                st.session_state.vector_store = index_dir
            else:
                st.write("Error occurred while indexing the documents")

    else:
        st.write("Error occurred while parsing and reading the documents")

if st.session_state.vector_store is not None:
    co = cohere.Client(COHERE_API_KEY)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model
    )
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=st.session_state.vector_store),
        service_context=service_context
    )
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[cohere_rerank],
        response_mode="no_text"
    )

    custom_chat_history = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            custom_message = {"user_name": "User", "text": message["content"]}
            custom_chat_history.append(custom_message)
        elif message["role"] == "assistant":
            custom_message = {"user_name": "Chatbot", "text": message["content"]}
            custom_chat_history.append(custom_message)

    prompt = st.chat_input(placeholder="Your question")

    # Prompt for user input and save to chat history
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = query_engine.query(str(prompt))
                    documents_list = []
                    document_number = 0
                    for node in response.source_nodes:
                        local_dict = {"title": f"answer_candidate_{document_number}", "snippet": node.text}
                        documents_list.append(local_dict)

                    try:
                        response_cohere = co.chat(
                            prompt,
                            model="command-nightly",
                            temperature=0.3,
                            prompt_truncation='off',
                            documents=documents_list)
                    except Exception as e:
                        response_cohere = co.chat(
                            prompt,
                            model="command-nightly",
                            temperature=0.3,
                            prompt_truncation='AUTO',
                            documents=documents_list)

                    st.write(response_cohere.text)
                    response_message = {"role": "assistant", "content": response_cohere.text}
                    st.session_state.messages.append(response_message)

                except Exception as e:
                    st.write("Error Occurred")
                    response_message = {"role": "assistant", "content": "Error Occurred"}
                    st.session_state.messages.append(response_message)

    st.button('Reset Chat', on_click=reset_conversation)

st.divider()


tab1, tab2 = st.tabs(["Literature-Review", "Read Mode(Powered by Nougat:- Meta)"])

with tab1:
    st.header('Search powered by Cohere and Semantic Scholar', divider='rainbow')
    user_input_search = st.text_input("Enter any query you have related to research/academic topics",
                                      "Limitations of LLMs", key="input5")

    web = st.checkbox('Web')
    semantic_scholar = st.checkbox('Semantic Scholar')


    def query_search_semantic_scholar(query, result_limit=10):
        rsp = requests.get('https://api.semanticscholar.org/graph/v1/paper/search',
                           headers={'X-API-KEY': S2_API_KEY},
                           params={'query': query, 'limit': result_limit, 'fields': 'url,title,tldr,abstract'})
        try:

            rsp.raise_for_status()
            results = rsp.json()
            print(results)
            total = results["total"]
            if not total:
                return 'No matches found. Please try another query.'

            try:
                papers = results['data']
                results_dict = {}
                results_list = []

                for paper in papers:
                    local_dict = {}
                    local_dict["title"] = paper["title"]
                    local_dict["url"] = paper["url"]
                    if paper["tldr"] != None and paper["tldr"]["text"] and paper["tldr"]["text"] != "":
                        local_dict["snippet"] = paper["tldr"]["text"]
                    elif paper["abstract"] != None:
                        local_dict["snippet"] = paper["abstract"]
                    else:
                        local_dict["snippet"] = "No data found"
                    results_list.append(local_dict)

                results_dict["results"] = results_list

                return results_dict
            except Exception as e:
                return 'No matches found. Please try another query.'

        except Exception as e:
            return "Limit Exceeded"


    if st.button("Search"):
        with st.spinner(text="Searching for an Answer using LLMs powered by Cohere"):
            co = cohere.Client(COHERE_API_KEY)
            try:
                if web and semantic_scholar:
                    try:
                        response_search_cohere = co.chat(
                            user_input_search,
                            model="command-nightly",
                            temperature=0.3,
                            prompt_truncation='off',
                            citation_quality='accurate',
                            connectors=[{"id":"semanticscholar-08yt70"},
                                        {"id":"web-search"}]
                            )
                    except Exception as e:
                        response_search_cohere = co.chat(
                            user_input_search,
                            model="command-nightly",
                            temperature=0.3,
                            prompt_truncation='AUTO',
                            citation_quality='accurate',
                            connectors=[{"id": "semanticscholar-08yt70"},
                                        {"id": "web-search"}]
                            )
                elif web==False and semantic_scholar==True:
                    try:
                        response_search_cohere = co.chat(
                            user_input_search,
                            model="command-nightly",
                            temperature=0.3,
                            prompt_truncation='off',
                            citation_quality='accurate',
                            connectors=[{"id":"semanticscholar-08yt70"}]
                            )
                    except Exception as e:
                        response_search_cohere = co.chat(
                            user_input_search,
                            model="command-nightly",
                            temperature=0.3,
                            prompt_truncation='AUTO',
                            citation_quality='accurate',
                            connectors=[{"id": "semanticscholar-08yt70"}]
                            )
                elif web==True and semantic_scholar==False:
                    try:
                        response_search_cohere = co.chat(
                            user_input_search,
                            model="command-nightly",
                            temperature=0.3,
                            prompt_truncation='off',
                            citation_quality='accurate',
                            connectors=[{"id": "web-search"}]
                        )
                    except Exception as e:
                        response_search_cohere = co.chat(
                            user_input_search,
                            model="command-nightly",
                            temperature=0.3,
                            prompt_truncation='AUTO',
                            citation_quality='accurate',
                            connectors=[{"id": "web-search"}]
                            )

                else:
                    try:
                        response_search_cohere = co.chat(
                            user_input_search,
                            model="command-nightly",
                            temperature=0.3,
                            prompt_truncation='off',
                            )
                    except Exception as e:
                        response_search_cohere = co.chat(
                            user_input_search,
                            model="command-nightly",
                            temperature=0.3,
                            prompt_truncation='AUTO',
                            )

                st.write(response_search_cohere.text)
                with st.expander("See Citations"):
                    try:
                        st.write(response_search_cohere.citations)

                    except Exception as e:
                        print(str(e))
                        st.write("No Citations found")

                with st.expander("See Semantic Scholar results"):
                    try:
                        st.write(query_search_semantic_scholar(query=user_input_search))

                    except Exception as e:
                        print(str(e))
                        st.write("No Citations found")

            except Exception as e:
                print(str(e))
                st.write("Error Occurred")


with tab2:
    user_input = st.text_input("Enter the arxiv or acl anthology url of the paper",
                               "https://arxiv.org/abs/2310.16787", key="input1")


    def get_paper_content(url: str) -> tuple:
        with st.spinner(text="Using Nougat(https://facebookresearch.github.io/nougat/) to read the paper contents and get the markdown representation of the paper"):
            try:
                f = modal.Function.lookup("streamlit-hack", "main")
                output = f.call(url)
                st.session_state.paper_content = output

            except Exception as e:
                output = "Error Occurred"
                st.session_state.paper_content = output

            return output

    if st.button("Read and Index Paper"):
        paper_content = get_paper_content(url=user_input)

    if st.session_state.paper_content is not None:
        with st.expander("See Paper Contents"):
            st.write(st.session_state.paper_content)
