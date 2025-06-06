from langchain_core.embeddings import Embeddings
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain import hub
from langchain_openai import AzureChatOpenAI
from typing import List, Tuple, Optional, Any
import cohere


class CohereEmbeddings(Embeddings):
    def __init__(self, keys_client: SecretClient):
        """Initialize Cohere embeddings with Azure Key Vault credentials."""
        # Get Cohere credentials from Azure Key Vault
        cohere_api_key = keys_client.get_secret("CO-API-KEY").value
        cohere_endpoint = keys_client.get_secret("AZURE-ML-COHERE-EMBED-ENDPOINT").value
        # Making embeddings client 
        self.co_embed = cohere.Client(
            api_key=cohere_api_key,
            base_url=cohere_endpoint,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.co_embed.embed(texts=texts, input_type="search_document")
        return response.embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.co_embed.embed(texts=[text], input_type="search_document")
        return response.embeddings[0]


class RAGService:
    def __init__(self, keys_client: SecretClient):
        """Initialize RAG service with Azure credentials."""
        # Initialize Azure credentials
        self.keys_client = keys_client
        
        # Get Azure OpenAI credentials
        api_base = keys_client.get_secret("archivai-openai-base").value
        api_key = keys_client.get_secret("archivaigpt4-key").value
        deployment_name = 'archivaigpt4'
        api_version = '2025-01-01-preview'

        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=api_base,
            azure_deployment=deployment_name,
            openai_api_version=api_version,
            openai_api_key=api_key,
        )

        # Initialize embeddings
        self.cohere_embeddings = CohereEmbeddings(keys_client)
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name="archivai_collection",
            embedding_function=self.cohere_embeddings,
            persist_directory="./app/rag/chroma_langchain_db"
        )
        
        # Load RAG prompt
        self.prompt = hub.pull("rlm/rag-prompt")

    def store_data(self, document_text: str, file_id: int) -> dict:
        """
        Stores the document text and its file_id into the Chroma vector database.

        Args:
            document_text: The text content of the document.
            file_id: The integer ID of the file.
            
        Returns:
            dict: Status message
        """
        try:
            doc = Document(page_content=document_text, metadata={"file_id": file_id})
            self.vector_store.add_documents([doc])
            return {
                "status": "success",
                "message": f"Document with file_id {file_id} added to the vector store.",
                "file_id": file_id
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to store document: {str(e)}",
                "file_id": file_id
            }

    def retrieve(self, question: str, k: int = 10) -> Tuple[List[Optional[int]], str]:
        """
        Retrieves the top k relevant file_ids and a response from the LLM
        based on the given question.

        Args:
            question: The question to ask the LLM.
            k: Number of documents to retrieve (default: 10)

        Returns:
            A tuple containing:
                - A list of top k file_ids (or fewer if not enough documents are found).
                - The response string from the LLM.
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.similarity_search(question, k=k)

            # Extract file_ids from metadata
            top_file_ids: List[Optional[int]] = []
            for doc in retrieved_docs:
                if doc.metadata and "file_id" in doc.metadata:
                    top_file_ids.append(doc.metadata["file_id"])
                else:
                    top_file_ids.append(None)

            # Prepare context for the LLM
            docs_content = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Invoke the prompt and LLM
            message_payload = self.prompt.invoke({
                "question": question,
                "context": docs_content
            })
            answer = self.llm.invoke(message_payload)
            
            return top_file_ids, answer.content
            
        except Exception as e:
            return [], f"Error during retrieval: {str(e)}"

    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store collection."""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            return {
                "status": "success",
                "document_count": count,
                "collection_name": "archivai_collection"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get collection stats: {str(e)}"
            }

    def clear_data(self, file_id: List[int] | int | None = None) -> dict:
        """
        Clear documents from the vector database.
        
        Args:
            file_id: If provided, only documents with this file_id will be deleted.
                    If None, all documents will be cleared.
                    
        Returns:
            dict: Status message with details about the deletion
        """
        try:
            collection = self.vector_store._collection
            
            if file_id is None:
                # Clear all documents
                initial_count = collection.count()
                if initial_count == 0:
                    return {
                        "status": "success",
                        "message": "No documents to clear from vector database.",
                        "deleted_count": 0,
                        "file_id": None
                    }
                
                # Get all document IDs
                ids_to_delete = collection.get(include=[])['ids']
                
                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)
                
                return {
                    "status": "success",
                    "message": f"All documents cleared from vector database.",
                    "deleted_count": initial_count,
                    "file_id": None
                }
            else:
                # if file_id is int, convert to list
                if isinstance(file_id, int):
                    file_id = [file_id]

                if len(file_id) == 0:
                    return {
                        "status": "warning",
                        "message": "No file_id provided to clear.",
                        "deleted_count": 0,
                        "file_id": None
                    }
                
                if len(file_id) > 1:
                    query = {"$or": [{"file_id": fid} for fid in file_id]}
                else:
                    query = {"file_id": file_id[0]}
                results = collection.get(where=query, include=[]) # include=[] for efficiency

                ids_to_delete = results["ids"]
                
                if not ids_to_delete:
                    return {
                        "status": "warning",
                        "message": f"No documents found with file_id {file_id}",
                        "deleted_count": 0,
                        "file_id": file_id
                    }
                
                # Delete documents with the specified file_id
                collection.delete(ids=ids_to_delete)
                deleted_count = len(ids_to_delete)
                
                return {
                    "status": "success",
                    "message": f"Deleted {deleted_count} documents with file_id {file_id}",
                    "deleted_count": deleted_count,
                    "file_id": file_id
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to clear documents: {str(e)}",
                "file_id": file_id
            }
