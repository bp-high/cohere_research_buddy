from typing import List
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.llms import LiteLLM
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.readers.schema.base import Document
from llama_index.node_parser import TokenTextSplitter, SentenceSplitter
import os
import uuid
import streamlit as st

COHERE_API_KEY = st.secrets.COHERE_API_KEY

llm = LiteLLM("command-nightly")
embed_model = CohereEmbedding(
    cohere_api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_document",
)


def get_vector_index(chunks_list: List) -> str:
    try:
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model
        )

        documents = [Document(text=t) for t in chunks_list]
        parser = SentenceSplitter(
                                    chunk_size=3000,
                                    chunk_overlap=20,
                                 )
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, service_context=service_context)

        persist_dir = uuid.uuid4().hex
        index.storage_context.persist(persist_dir=persist_dir)

        return persist_dir

    except Exception as e:
        print(f"Indexing Error:- {str(e)}")
        return "Error"


