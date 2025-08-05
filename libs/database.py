import logging
from chromadb.api import ClientAPI
from chromadb.api.types import (
    QueryResult,
)

logger = logging.getLogger("RAG")


def search(
    client: ClientAPI, collection_name: str, query: str, embedding_function
) -> QueryResult:
    logger.debug(
        f"Searching collection '{
            collection_name}' with query '{query}'"
    )

    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    # https://docs.trychroma.com/docs/overview/getting-started
    results = collection.query(
        query_texts=[query],  # Chroma will embed this for you
        n_results=4,  # how many results to return
    )

    return results
