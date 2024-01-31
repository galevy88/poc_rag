import os
import openai
import pinecone
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from typing import List
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=Warning)
def get_embeddings(texts, model="text-embedding-ada-002"):

    responses = openai.Embedding.create(input=texts, model=model)

    embeddings = [response['embedding'] for response in responses['data']]
    return embeddings

def construct_context(contexts: List[str], max_section_len=1000) -> str:
    chosen_sections = []
    chosen_sections_len = 0
    separator = "\n"

    for text in contexts:
        text = text.strip()
        chosen_sections_len += len(text) + 2
        if chosen_sections_len > max_section_len:
            break
        chosen_sections.append(text)

    concatenated_doc = separator.join(chosen_sections)
    return concatenated_doc

def main():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENV')

    openai.api_key = openai_api_key
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")

    # RELATED QUESTIONS
    question1 = "how many civilians were killed in the festival"
    question2 = "what happened in nova festival"
    question3 = "when the massacare of hamas took place"
    question4 = "where the massacare of hamas took place"
    question5 = "please tell what was israel response for hamas massacre"

    # RELATED QUESTIONS
    question6 = "how long apples stay fresh"
    question7 = "are oranges have a lot of calories"
    question8 = "how long it takes to oranges to grow"
    question9 = "are apples sweet?"

    question = question1

    index_name = "freetruth"
    index = pinecone.Index(index_name)

    # Retrieve context from Pinecone index
    query_vec = get_embeddings([question])[0]
    res = index.query(query_vec, top_k=5, include_metadata=True)
    contexts = [match['metadata']['text'] for match in res['matches']]
    context_str = construct_context(contexts)
    
    prompt_template = """
    Answer the following QUESTION based on the CONTEXT given. Please based on the CONTEXT only If you do not know the answer and the CONTEXT doesn't contain the answer truthfully say I don't know.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    text_input = prompt_template.format(context=context_str, question=question)
    result = llm.predict(text_input)
    print(result)

if __name__ == "__main__":
    main()
