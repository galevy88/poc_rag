import pandas as pd
import openai
import pinecone
from dotenv import load_dotenv
import os
from tqdm.auto import tqdm

def insert_into_pinecone(csv_file, index_name):
    df = pd.read_csv(csv_file, header=None, names=["Question", "Answer"])

    # Combine the question and answer
    df['Combined'] = df['Question'] + " " + df['Answer']

    def get_embeddings(texts, model="text-embedding-ada-002"):
        responses = openai.Embedding.create(input=texts, model=model)
        embeddings = [response['embedding'] for response in responses['data']]
        return embeddings

    # Get embeddings for the combined text
    embeddings = get_embeddings(df['Combined'].tolist())

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name)
    index = pinecone.Index(index_name)

    for i in tqdm(range(len(df)), desc="Inserting records"):
        combined_text = df.iloc[i]['Combined']
        
        embedding = embeddings[i]
        # Use combined text in metadata
        record = (str(i), embedding, {"text": combined_text})
        index.upsert(vectors=[record])

def main():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENV')

    openai.api_key = openai_api_key
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    insert_into_pinecone("massacre_data.csv", "freetruth")
    # insert_into_pinecone("apple_orange_data.csv", "freetruth")

if __name__ == "__main__":
    main()
