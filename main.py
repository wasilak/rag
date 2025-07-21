import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import argparse
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader, AsyncHtmlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme


def main():
  parser = argparse.ArgumentParser(description="Run ChromaDB with Ollama Embedding Function")
  parser.add_argument("--collection", type=str, default="RAG", help="Name of the collection to use or create")
  parser.add_argument("--db-path", type=str, default="db", help="Path to the ChromaDB database")
  parser.add_argument("--dry-run", action='store_true', help="Run in dry-run mode without making changes")

  subparsers = parser.add_subparsers(dest='subparser')

  # data fill subcommand
  data_subparser = subparsers.add_parser("data-fill", help="Data fill subcommand")
  data_subparser.add_argument("source_path", type=str, default=None, help="Path to the source data file or URL")
  data_subparser.add_argument("--source-type", type=str, default="file", choices=["file", "url"], help="Type of the source data: 'file' for local file, 'url' for remote URL")
  data_subparser.add_argument("--mode", type=str, default="single", choices=["single", "elements"], help="Mode for processing the data: 'single' for single file, 'elements' for multiple elements in a file")
  data_subparser.add_argument("--embedding-model", type=str, default="nomic-embed-text", help="Embedding model to use with Ollama (default: bge-m3)") # nomic-embed-text, bge-m3, mxbai-embed-large

  # seach subcommand
  search_parser = subparsers.add_parser("search", help="Search for documents in the collection")
  search_parser.add_argument("query", type=str, help="Query text to search for in the collection")
  search_parser.add_argument("--llm", type=str, default="ollama", choices=["openai", "ollama"], help="LLM to use for processing the query")
  search_parser.add_argument("--model", type=str, default="qwen3:8b", help="Model to use for the LLM")
  # model="gpt-4o",
  # model="qwen3:8b",
  # model="deepseek-r1:14b",

  args = parser.parse_args()

  # print(args)
  # exit(0)

  client = chromadb.PersistentClient(path=args.db_path)

  if args.subparser == "data-fill":

    if args.source_type == "file":
      full_path = os.path.abspath(args.source_path)

      loader = UnstructuredMarkdownLoader(full_path, mode=args.mode)
      raw_documents = loader.load()

    else:
      # https://python.langchain.com/docs/how_to/document_loader_markdown/
      urls = [args.source_path]
      loader = AsyncHtmlLoader(urls)
      docs = loader.load()

      # https://python.langchain.com/docs/integrations/document_transformers/markdownify/
      md = MarkdownifyTransformer()
      raw_documents = md.transform_documents(docs)

    bootstrap_db(client, args.collection, raw_documents, args.embedding_model, args.mode)

    print(f"Collection '{args.collection}' has been created and filled with data.")
    exit(0)

  elif args.subparser == "search":

    if args.llm == "ollama":
      client_llm = OpenAI(
          base_url = 'http://localhost:11434/v1',
          api_key='ollama', # required, but unused
      )
    else:
      client_llm = OpenAI()

    search(client_llm, args.model, client, args.collection, args.query, args.dry_run)
    exit(0)

def bootstrap_db(client, collection, raw_documents, embedding_model, mode):

  ollama_ef = OllamaEmbeddingFunction(
      url="http://localhost:11434",
      model_name=embedding_model,
  )

  try:
      client.delete_collection(name=collection)
  except Exception as e:
      print(f"Error deleting collection {collection}: {e}")
  finally:
      print(f"Creating collection {collection} with Ollama embedding function...")
      collection = client.get_or_create_collection(
          name=collection,
          embedding_function=ollama_ef
      )

  # splitter = MarkdownTextSplitter(
  #     # chunk_size=300,
  #     # chunk_overlap=100,
  #     # length_function=len,
  #     # is_separator_regex=False,
  # )

  splitter = RecursiveCharacterTextSplitter(
      # chunk_size=300,
      # chunk_overlap=100,
      # length_function=len,
      # is_separator_regex=False,
  )

  chunks = splitter.split_documents(raw_documents)

  documents, metadata, ids = process_markdown_documents(chunks, mode)

  collection.upsert(
      documents=documents,
      metadatas=metadata,
      ids=ids
  )


def process_markdown_documents(chunks, mode):
  documents, metadata, ids = [], [], []

  if mode == "single":
    i = 0
    for chunk in chunks:
        documents.append(chunk.page_content)
        ids.append("ID_"+str(i))
        metadata.append(chunk.metadata)
        i += 1

  if mode == "elements":

    # Build element_id map
    elements_by_id = {doc.metadata["element_id"]: doc for doc in chunks}

    def get_ancestor_chain(doc_id, elements_by_id, max_levels=100):
        chain = []
        current_id = doc_id
        levels = 0
        while levels < max_levels:
            doc = elements_by_id.get(current_id)
            if not doc or "parent_id" not in doc.metadata:
                break
            parent_id = doc.metadata["parent_id"]
            parent = elements_by_id.get(parent_id)
            if not parent:
                break
            chain.insert(0, parent)
            current_id = parent_id
            levels += 1
        return chain



    for i, chunk in enumerate(chunks):
        element_id = chunk.metadata["element_id"]
        parent_chain_docs = get_ancestor_chain(element_id, elements_by_id)

        documents.append(chunk.page_content)
        ids.append(f"ID_{i}")

        temp_metadata = {}
        for key, value in chunk.metadata.items():
            temp_metadata[key] = ', '.join(value) if isinstance(value, list) else value

        if parent_chain_docs:
            temp_metadata["page_title"] = parent_chain_docs[0].page_content
            temp_metadata["resolved_title"] = chunk.page_content if chunk.metadata.get("category") == "Title" else parent_chain_docs[-1].page_content
        else:
            temp_metadata["page_title"] = chunk.page_content
            temp_metadata["resolved_title"] = chunk.page_content

        metadata.append(temp_metadata)

  return documents, metadata, ids

def format_footnotes(metadatas: list[dict]) -> str:
    """
    Deduplicates and formats footnotes from a list of metadata dicts.
    Assumes each metadata contains 'resolved_title' and 'source' fields.
    """
    seen = set()
    numbered = []
    for meta in metadatas:
        title = meta.get("resolved_title") or meta.get("top_title") or "Untitled"
        source = meta.get("source", "unknown.md").split("/")[-1]  # Extract filename
        key = (title, source)
        if key not in seen:
            seen.add(key)
            numbered.append((title.strip(), source.strip()))

    # Format into footnotes
    result = "\n".join([f"[{i+1}] “{title}”, `{source}`" for i, (title, source) in enumerate(numbered)])
    return result

def search(client_llm, model, client, collection_name, query, dry_run):
  collection = client.get_or_create_collection(name=collection_name)

  # https://docs.trychroma.com/docs/overview/getting-started
  results = collection.query(
      query_texts=[query], # Chroma will embed this for you
      n_results=4 # how many results to return
  )
  # print(results)
  # exit(0)

  system_prompt = """
  ### Example
  User: How can a coding buddy help with learning Rust?

  Answer:
  A coding buddy can help you improve by providing feedback through code reviews and engaging in pair programming sessions [1]. They can also help you solidify your understanding of Rust by encouraging you to explain concepts aloud [2].

  Footnotes:
  [1] “Find A Coding Buddy” in “Flattening Rust's Learning Curve | corrode Rust Consulting”, `markdown.md`
  [2] “Explain Rust Code To Non-Rust Developers” in “Flattening Rust's Learning Curve | corrode Rust Consulting”, `markdown.md`

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
    [1] “{section}” in “{page}”, `{source}`

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
      messages = [
          {"role":"system","content":system_prompt},
          {"role":"user","content":query}
      ]
  )

  if len(response.choices) > 0:
    # print(response.choices[0].message.content)
    print_fancy_markdown(response.choices[0].message.content, "📝 Agent Reply")
  else:
    print("No response from OpenAI API")
    print(response)


def print_fancy_markdown(md: str, title: str, border_style: str = "green", code_theme: str = "monokai"):
    """
    Render markdown with:
    - Styled headings
    - Syntax-highlighted code blocks
    - Wrapped in a panel for emphasis
    """
    # Define a default custom theme for markdown
    custom_theme = Theme({
        "markdown.h1": "bold cyan",
        "markdown.h2": "bold magenta",
        "markdown.h3": "bold green",
        "markdown.code": "bright_white on dark_green",
        "markdown.block_quote": "italic yellow",
        "markdown.list_item": "white",
    })

    console = Console(theme=custom_theme, highlight=True)

    md_render = Markdown(md, code_theme=code_theme)

    # Wrap in a panel with a title
    console.print(Panel(md_render, title=title, border_style=border_style, expand=True))

if __name__ == "__main__":
  load_dotenv()
  main()
