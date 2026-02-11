"""
Query and chat engines for LlamaIndex
"""

import logging
from typing import List, Dict, Optional, Any

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine, SimpleChatEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)

class QueryEngineManager:
    """Manage query and chat engines"""
    
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.llm = OpenAI(
            model=settings.MODEL_NAME,
            temperature=0.1,
            api_key=settings.OPENAI_API_KEY
        )
        self.similarity_top_k = settings.SIMILARITY_TOP_K
        self.response_mode = settings.RESPONSE_MODE
    
    def create_query_engine(
        self,
        similarity_top_k: Optional[int] = None,
        response_mode: Optional[str] = None,
        use_postprocessor: bool = True
    ) -> RetrieverQueryEngine:
        """
        Create a query engine
        
        Args:
            similarity_top_k: Number of similar documents to retrieve
            response_mode: Response synthesis mode
            use_postprocessor: Whether to use similarity postprocessor
            
        Returns:
            Configured query engine
        """
        similarity_top_k = similarity_top_k or self.similarity_top_k
        response_mode = response_mode or self.response_mode
        
        logger.info(f"Creating query engine with top_k={similarity_top_k}, mode={response_mode}")
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k
        )
        
        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode=response_mode,
            llm=self.llm
        )
        
        # Create postprocessors
        postprocessors = []
        if use_postprocessor:
            postprocessors.append(
                SimilarityPostprocessor(similarity_cutoff=0.7)
            )
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors
        )
        
        return query_engine
    
    def create_chat_engine(
        self,
        chat_mode: str = "condense_question",
        similarity_top_k: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Create a chat engine
        
        Args:
            chat_mode: Chat mode ("condense_question", "simple", "context")
            similarity_top_k: Number of similar documents to retrieve
            verbose: Whether to show verbose output
            
        Returns:
            Configured chat engine
        """
        similarity_top_k = similarity_top_k or self.similarity_top_k
        
        logger.info(f"Creating chat engine with mode={chat_mode}")
        
        if chat_mode == "condense_question":
            # Create query engine for chat
            query_engine = self.create_query_engine(similarity_top_k=similarity_top_k)
            
            chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=query_engine,
                llm=self.llm,
                verbose=verbose
            )
        elif chat_mode == "simple":
            chat_engine = SimpleChatEngine.from_defaults(
                llm=self.llm,
                verbose=verbose
            )
        else:
            # Default to condense_question
            chat_engine = self.create_chat_engine(
                chat_mode="condense_question",
                similarity_top_k=similarity_top_k,
                verbose=verbose
            )
        
        return chat_engine
    
    def query(
        self,
        question: str,
        similarity_top_k: Optional[int] = None,
        response_mode: Optional[str] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the index
        
        Args:
            question: Query question
            similarity_top_k: Number of similar documents to retrieve
            response_mode: Response synthesis mode
            include_sources: Whether to include source information
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Querying: {question}")
        
        query_engine = self.create_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=response_mode
        )
        
        response = query_engine.query(question)
        
        result = {
            "question": question,
            "answer": str(response),
            "response": response
        }
        
        if include_sources and hasattr(response, 'source_nodes'):
            sources = []
            for node in response.source_nodes:
                source_info = {
                    "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "metadata": node.metadata,
                    "score": getattr(node, 'score', None)
                }
                sources.append(source_info)
            result["sources"] = sources
        
        return result
    
    def chat(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        chat_mode: str = "condense_question",
        similarity_top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Chat with the index
        
        Args:
            message: Chat message
            chat_history: Previous chat history
            chat_mode: Chat mode to use
            similarity_top_k: Number of similar documents to retrieve
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Chat message: {message}")
        
        chat_engine = self.create_chat_engine(
            chat_mode=chat_mode,
            similarity_top_k=similarity_top_k
        )
        
        # If chat history is provided, you might need to handle it differently
        # depending on the chat engine implementation
        response = chat_engine.chat(message)
        
        result = {
            "message": message,
            "answer": str(response),
            "response": response
        }
        
        # Add source information if available
        if hasattr(response, 'source_nodes'):
            sources = []
            for node in response.source_nodes:
                source_info = {
                    "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "metadata": node.metadata,
                    "score": getattr(node, 'score', None)
                }
                sources.append(source_info)
            result["sources"] = sources
        
        return result
    
    def get_query_suggestions(self, topic: str, num_suggestions: int = 5) -> List[str]:
        """
        Get query suggestions based on a topic
        
        Args:
            topic: Topic to generate suggestions for
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of suggested queries
        """
        prompt = f"""
        Based on the topic "{topic}", generate {num_suggestions} specific and useful questions 
        that would be good for querying a document database. Make the questions specific and actionable.
        
        Format each question on a new line.
        """
        
        try:
            response = self.llm.complete(prompt)
            suggestions = [line.strip() for line in str(response).split('\n') if line.strip()]
            return suggestions[:num_suggestions]
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []
    
    def explain_response(self, response) -> Dict[str, Any]:
        """
        Explain the response with detailed information
        
        Args:
            response: Query response object
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            "answer": str(response),
            "type": type(response).__name__,
        }
        
        if hasattr(response, 'source_nodes'):
            explanation["num_sources"] = len(response.source_nodes)
            explanation["sources"] = []
            
            for i, node in enumerate(response.source_nodes):
                source_info = {
                    "index": i,
                    "content_preview": node.text[:100] + "..." if len(node.text) > 100 else node.text,
                    "metadata": node.metadata,
                    "score": getattr(node, 'score', None)
                }
                explanation["sources"].append(source_info)
        
        return explanation
