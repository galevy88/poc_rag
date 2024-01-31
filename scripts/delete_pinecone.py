import pinecone
import os
from dotenv import load_dotenv

def delete_all_vectors_from_index(index_name, api_key, environment):

    pinecone.init(api_key=api_key, environment=environment)
    index = pinecone.Index(index_name)

    vector_ids = set()


    while True:
    
        query_vector = [0] * 1536

        response = index.query(query_vector, top_k=1000)
        new_ids = {match["id"] for match in response["matches"]}

        if not new_ids.difference(vector_ids):
            break

        vector_ids.update(new_ids)

        last_vector_id = response["matches"][-1]["id"]

    for vector_id in vector_ids:
        index.delete([vector_id])

    print(f"Deleted {len(vector_ids)} vectors from the index '{index_name}'.")



if __name__=="__main__":

    load_dotenv()
    api_key = os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENV')

    index_name = 'freetruth'
    delete_all_vectors_from_index(index_name, api_key, environment)
