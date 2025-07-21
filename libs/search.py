import os
import logging
from openai import OpenAI
from .embedding import set_embedding_function
from .utils import format_footnotes, print_fancy_markdown

logger = logging.getLogger("RAG")


def create_llm_client(llm):
    """Create LLM client based on the provider"""
    if llm == "ollama":
        logger.info(f"Using Ollama as LLM")
        return OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',  # required, but unused
        )
    elif llm == "gemini":
        logger.info(f"Using Gemini as LLM")
        return OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    else:
        logger.info(f"Using OpenAI as LLM")
        return OpenAI()


def search(client_llm, model, client, collection_name, query, dry_run, embedding_function):
    """Search for documents in the collection and generate response"""
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    # https://docs.trychroma.com/docs/overview/getting-started
    results = collection.query(
        query_texts=[query],  # Chroma will embed this for you
        n_results=4  # how many results to return
    )

    system_prompt = """
    ### Example
    User: How can a coding buddy help with learning Rust?

    Answer:
    A coding buddy can help you improve by providing feedback through code reviews and engaging in pair programming sessions [1]. They can also help you solidify your understanding of Rust by encouraging you to explain concepts aloud [2].

    Footnotes:
    [1] "Find A Coding Buddy" in "Flattening Rust's Learning Curve | corrode Rust Consulting", `markdown.md`
    [2] "Explain Rust Code To Non-Rust Developers" in "Flattening Rust's Learning Curve | corrode Rust Consulting", `markdown.md`

    ---

    You are a helpful assistant that answers questions strictly based on the provided documents.

    Each document includes:
    - `text`: the paragraph or chunk content
    - `resolved_title`: the section this chunk belongs to
    - `page_title`: the title of the entire markdown document
    - `source`: the filename (e.g., markdown.md)

    When answering:
    - Use only the provided document text — do not invent or guess.
    - Use full sentences and clearly explained reasoning.
    - Every factual statement must be annotated with a footnote reference like [1], [2], etc.

    Footnotes must:
    - Be deduplicated
    - Follow the format:
      [1] "{section}" in "{page}", `{source}`

    VERY IMPORTANT:
    - yor response must be nicely formatted in valid Markdown, with footnotes at the end of the answer
    - Use text formatting like bold, italics, and code blocks as needed.
    - use quotes for direct citations from the documents.
    - Use headings, lists, and other Markdown features to improve readability.

    If the documents do not answer the question, respond with: I don't know.
    """

    footnotes_metadata = []

    for i, item in enumerate(results["documents"][0]):
        metadata_entry = results["metadatas"][0][i]
        footnotes_metadata.append(metadata_entry)

        system_prompt += f"{i + 1}. \"{item}\"\n"
        system_prompt += f"metadata: {metadata_entry}\n\n"

    # Format footnotes from metadata
    footnotes = format_footnotes(footnotes_metadata)
    system_prompt += f"Footnotes:\n{footnotes}\n"

    # print(system_prompt)
    print_fancy_markdown(system_prompt, "📝 System Prompt", border_style="blue")

    if dry_run:
        exit(0)

    response = client_llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    if len(response.choices) > 0:
        # print(response.choices[0].message.content)
        print_fancy_markdown(response.choices[0].message.content, "📝 Agent Reply")
    else:
        print("No response from OpenAI API")
        print(response)


def process_search(client, collection, query, llm, model, dry_run, embedding_model, embedding_llm):
    """Process search operation"""
    logger.info(f"Searching collection '{collection}' with query '{query}'")

    client_llm = create_llm_client(llm)
    embedding_function = set_embedding_function(embedding_llm, embedding_model)

    search(client_llm, model, client, collection, query, dry_run, embedding_function)
    logger.info("Search completed")
