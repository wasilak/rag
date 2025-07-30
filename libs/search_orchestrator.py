import logging
from typing import List, Optional, Any, Tuple, cast
from dataclasses import dataclass
from openai import OpenAI
from chromadb.api import ClientAPI
from chromadb.api.types import QueryResult


logger = logging.getLogger("RAG")


@dataclass
class SearchIteration:
    """Represents a single search iteration"""

    query: str
    results: QueryResult
    relevance_score: float
    iteration: int
    refined_query: Optional[str] = None
    analysis: Optional[str] = None


@dataclass
class SearchResult:
    """Final search result with iteration history"""

    best_results: QueryResult
    best_score: float
    iterations: List[SearchIteration]
    final_query: str
    original_query: str


class SearchOrchestrator:
    def __init__(
        self,
        client: ClientAPI,
        llm_client: OpenAI,
        collection_name: str,
        embedding_function: Any,
        model: str,
        max_iterations: int = 3,
        min_relevance_score: float = 0.7,
        debug: bool = False,
    ):
        self.client = client
        self.llm_client = llm_client
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.model = model
        self.max_iterations = max_iterations
        self.min_relevance_score = min_relevance_score
        self.debug = debug
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function
        )

    def evaluate_results(
        self, results: QueryResult, query: str, iteration: int
    ) -> Tuple[float, str, str]:
        """
        Evaluate search results using LLM to determine relevance and suggest improvements
        Returns: (relevance_score, analysis, refined_query)
        """
        # Prepare context from results
        context = self._format_results_for_evaluation(results)

        evaluation_prompt = f"""
        You are an expert search quality analyst. Analyze these search results for the query: "{query}"

        Search iteration: {iteration}
        Context from search:
        {context}

        Please provide three things:
        1. A relevance score between 0.0 and 1.0 (where 1.0 is perfect relevance)
        2. Brief analysis of why this score was given
        3. A refined search query that would get better results (or None if current results are optimal)

        Format your response exactly as follows:
        SCORE: (number between 0.0-1.0)
        ANALYSIS: (your analysis)
        REFINED_QUERY: (your suggested query or None)
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search quality analyst. Provide exact format as requested.",
                    },
                    {"role": "user", "content": evaluation_prompt},
                ],
            )

            if not response.choices or not response.choices[0].message.content:
                logger.error("No response from LLM during evaluation")
                return 0.0, "Evaluation failed", query

            # Parse response
            content = response.choices[0].message.content
            lines = content.split("\n")

            score = 0.0
            analysis = "No analysis provided"
            refined_query = query

            for line in lines:
                if line.startswith("SCORE:"):
                    try:
                        score = float(line.replace("SCORE:", "").strip())
                    except ValueError:
                        score = 0.0
                elif line.startswith("ANALYSIS:"):
                    analysis = line.replace("ANALYSIS:", "").strip()
                elif line.startswith("REFINED_QUERY:"):
                    refined = line.replace("REFINED_QUERY:", "").strip()
                    if refined.lower() != "none":
                        refined_query = refined

            return score, analysis, refined_query

        except Exception as e:
            logger.error(f"Error during result evaluation: {e}")
            return 0.0, f"Evaluation error: {str(e)}", query

    def _format_results_for_evaluation(self, results: QueryResult) -> str:
        """Format search results into a string for LLM evaluation"""
        formatted = []

        if not results or not results.get("documents") or not results["documents"]:
            return "No results found"

        # Safely get documents
        documents_list = results.get("documents", [[]])
        documents = (
            documents_list[0] if documents_list and len(documents_list) > 0 else []
        )

        # Safely get metadata with multiple fallbacks
        metadatas_list = results.get("metadatas")
        if metadatas_list and len(metadatas_list) > 0:
            metadatas = metadatas_list[0]
        else:
            metadatas = []

        # Process documents with safe metadata access
        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            formatted.append(f"Document {i + 1}:")
            formatted.append(
                f"Content: {doc[:200]}..." if len(doc) > 200 else f"Content: {doc}"
            )
            formatted.append(f"Metadata: {metadata}")
            formatted.append("---")

        return "\n".join(formatted)

    def perform_iterative_search(self, query: str) -> SearchResult:
        """
        Perform iterative self-improving search
        Returns the best results along with search history
        """
        iterations: List[SearchIteration] = []
        current_query = query
        best_results = None
        best_score = 0.0
        results = None

        logger.info(f"Starting iterative search for query: '{query}'")
        logger.info(
            f"Max iterations: {self.max_iterations}, Min relevance score: {self.min_relevance_score}"
        )

        for iteration in range(self.max_iterations):
            try:
                logger.info(f"\n=== Iteration {iteration + 1} ===")
                logger.info(f"Current query: '{current_query}'")

                # Perform search
                results = self.collection.query(
                    query_texts=[current_query], n_results=4
                )

                # Evaluate results
                relevance_score, analysis, refined_query = self.evaluate_results(
                    results, current_query, iteration + 1
                )

                # Record iteration
                current_iteration = SearchIteration(
                    query=current_query,
                    results=results,
                    relevance_score=relevance_score,
                    iteration=iteration + 1,
                    refined_query=refined_query,
                    analysis=analysis,
                )
                iterations.append(current_iteration)

                if results and results["documents"] and len(results["documents"]) > 0:
                    logger.info(
                        f"Results found: {len(results['documents'][0])} documents"
                    )
                    if self.debug:
                        for i, doc in enumerate(results["documents"][0]):
                            logger.debug(f"Document {i + 1}: {doc[:100]}...")
                else:
                    logger.info("No results found")

                logger.info(f"Relevance score: {relevance_score:.2f}")
                logger.info(f"Analysis: {analysis}")
                logger.info(f"Refined query: '{refined_query}'")

                # Update best results if better
                if relevance_score > best_score:
                    best_results = results
                    best_score = relevance_score

                # Check if we should continue
                if relevance_score >= self.min_relevance_score:
                    logger.info(
                        f"✓ Reached sufficient relevance score: {relevance_score:.2f} >= {self.min_relevance_score}"
                    )
                    break

                if current_query == refined_query:
                    logger.info(
                        "✓ No further query refinement suggested, stopping iterations"
                    )
                    break

                current_query = refined_query

            except Exception as e:
                logger.error(f"❌ Error in search iteration {iteration + 1}: {e}")
                break

        logger.info("\n=== Search completed ===")
        logger.info(f"Best score achieved: {best_score:.2f}")
        logger.info(f"Final query: '{current_query}'")
        logger.info(f"Total iterations: {len(iterations)}")

        if best_results is None and results is None:
            # Perform one final search if no results were obtained
            try:
                results = self.collection.query(query_texts=[query], n_results=4)
            except Exception as e:
                logger.error(f"Error in final fallback search: {e}")
                # Create a properly typed empty QueryResult
                results = cast(
                    QueryResult,
                    {
                        "ids": [[]],
                        "embeddings": None,
                        "documents": [[]],
                        "metadatas": [[]],
                        "distances": [[]],
                    },
                )

        return SearchResult(
            best_results=cast(
                QueryResult, best_results if best_results is not None else results
            ),
            best_score=best_score,
            iterations=iterations,
            final_query=current_query,
            original_query=query,
        )
