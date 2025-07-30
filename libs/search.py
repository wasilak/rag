import os
import logging
from chromadb.api import ClientAPI
from openai import OpenAI
from .embedding import set_embedding_function
from .utils import format_footnotes, print_fancy_markdown
from .models import get_best_model
from .search_orchestrator import SearchOrchestrator

logger = logging.getLogger("RAG")


def create_llm_client(
    llm: str, embedding_ollama_host: str, embedding_ollama_port: int
) -> OpenAI:
    """Create LLM client based on the provider"""
    if llm == "ollama":
        logger.debug("Using Ollama as LLM")
        return OpenAI(
            base_url=f"http://{embedding_ollama_host}:{embedding_ollama_port}/v1",
            api_key="ollama",  # required, but unused
        )
    elif llm == "gemini":
        logger.debug("Using Gemini as LLM")
        return OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        logger.debug("Using OpenAI as LLM")
        return OpenAI()


def search(
    client_llm: OpenAI,
    model: str,
    client: ClientAPI,
    collection_name: str,
    query: str,
    dry_run: bool,
    embedding_function,
) -> None:
    # Initialize search orchestrator
    orchestrator = SearchOrchestrator(
        client=client,
        llm_client=client_llm,
        collection_name=collection_name,
        embedding_function=embedding_function,
        model=model,
        debug=logger.isEnabledFor(logging.DEBUG),
    )

    # Perform iterative search
    search_result = orchestrator.perform_iterative_search(query)

    if logger.isEnabledFor(logging.DEBUG):
        for iteration in search_result.iterations:
            logger.debug(f"Iteration {iteration.iteration}:")
            logger.debug(f"Query: {iteration.query}")
            logger.debug(f"Score: {iteration.relevance_score}")
            logger.debug(f"Analysis: {iteration.analysis}")

    results = search_result.best_results

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
    - deduplicate footnotes! that means if you have a footnote [1] already, don't repeat it in response as [2] and so on.
    - if it makes sense, use quotes for direct citations from the documents.
    - if it makes sense, use headings, lists, tables, code blocks and other Markdown features to improve readability, especilayy to highlight important information, sections, topics, etc.
    - all response content (headings, paragraphs, lists, tables, code blocks, etc.) should be left-aligned, not centered!
    - if you use quotes, make sure to use them correctly, with proper spacing and punctuation.


    Footnotes must:
    - Be deduplicated
    - Follow the format:
      [1] "{section}" in "{page}", `{source}`

    If the documents do not answer the question, respond with: I don't know, unless it is about code generation.
    If document refers to external resources, you can use them.
    If you don't know the answer, BUT! user requested code examples, do not scope your answert to only documents provided, but reach out to your knowledge to figure out how to generate code examples, but they have to based only on information provided in the documents.
    """

    footnotes_metadata = []

    # Check if documents and metadatas are not None
    if results["documents"] and results["metadatas"]:
        for i, item in enumerate(results["documents"][0]):
            metadata_entry = results["metadatas"][0][i]
            footnotes_metadata.append(metadata_entry)

            system_prompt += f'{i + 1}. "{item}"\n'
            system_prompt += f"metadata: {metadata_entry}\n\n"

    # Format footnotes from metadata
    footnotes = format_footnotes(footnotes_metadata)
    system_prompt += f"Footnotes:\n{footnotes}\n"

    # print(system_prompt)
    if logger.isEnabledFor(logging.DEBUG):
        print_fancy_markdown(
            system_prompt,
            "📝 System Prompt",
            border_style="blue",
            borders_only="top_bottom",
        )

    if dry_run:
        exit(0)

    response = client_llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )

    if len(response.choices) > 0:
        markdown_content = (
            response.choices[0].message.content
            if response.choices[0].message.content
            else "No content returned"
        )
        print_fancy_markdown(
            markdown_content, "🤖 Agent Reply", borders_only="top_bottom"
        )
    else:
        print("No response from OpenAI API")
        print(response)


def process_search(
    client: ClientAPI,
    collection: str,
    query: str,
    llm: str,
    model: str,
    dry_run: bool,
    embedding_model: str,
    embedding_llm: str,
    embedding_ollama_host: str,
    embedding_ollama_port: int,
) -> None:
    """Process search operation"""
    logger.debug(f"Searching collection '{collection}' with query '{query}'")

    # Validate and get best available model
    validated_model = get_best_model(
        llm, embedding_ollama_host, embedding_ollama_port, model, "chat"
    )

    client_llm = create_llm_client(llm, embedding_ollama_host, embedding_ollama_port)
    embedding_function = set_embedding_function(
        embedding_llm, embedding_model, embedding_ollama_host, embedding_ollama_port
    )

    search(
        client_llm,
        validated_model,
        client,
        collection,
        query,
        dry_run,
        embedding_function,
    )
    logger.debug("Search completed")
