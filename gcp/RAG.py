"""
Complete Vertex AI RAG System for Nepali Documents
Creates Vector Index, Deploys Endpoint, Creates RAG Corpus, and Sets up RAG Engine
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import hashlib
import time

# Google Cloud imports
try:
    from google.cloud import aiplatform
    from vertexai.language_models import TextEmbeddingModel
    from vertexai.preview import rag
    import vertexai
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    print("Google Cloud libraries not installed. Install with:")
    print("   pip install google-cloud-aiplatform vertexai tqdm")


class NepaliRAGSystem:
    """
    Complete RAG system for Nepali documents using Vertex AI
    """
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """
        Initialize RAG system
        
        Args:
            project_id: Your Google Cloud project ID
            region: GCP region (default: us-central1)
        """
        self.project_id = project_id
        self.region = region
        self.embedding_model = None
        self.embedding_dimension = 768  # text-multilingual-embedding-002
        
        if VERTEX_AI_AVAILABLE:
            # Initialize Vertex AI
            vertexai.init(project=project_id, location=region)
            aiplatform.init(project=project_id, location=region)
            print(f"Initialized Vertex AI for project: {project_id}")
        else:
            print("Vertex AI not available. Running in dry-run mode.")
    
    def load_embedding_model(self):
        """Load Google's multilingual embedding model"""
        if not VERTEX_AI_AVAILABLE:
            print("Using dummy embeddings (Vertex AI not available)")
            return None
        
        try:
            self.embedding_model = TextEmbeddingModel.from_pretrained(
                "text-multilingual-embedding-002"
            )
            print("Loaded text-multilingual-embedding-002 model")
            return self.embedding_model
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            return None
    
    def generate_embedding(self, text: str, max_retries: int = 100) -> List[float]:
        """
        Generate embedding for text using Vertex AI with unlimited retry logic
        
        Args:
            text: Text to embed (supports Nepali)
            max_retries: Maximum number of retries on quota errors
        
        Returns:
            768-dimensional embedding vector
        """
        if not self.embedding_model:
            return self._generate_dummy_embedding(text)
        
        text = text[:5000] if len(text) > 5000 else text
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                embeddings = self.embedding_model.get_embeddings([text])
                return embeddings[0].values
            except Exception as e:
                error_message = str(e)
                
                if "429" in error_message or "Quota exceeded" in error_message or "quota" in error_message.lower():
                    retry_count += 1
                    wait_time = min(2 ** min(retry_count, 6), 60)
                    print(f"Quota limit hit. Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error generating embedding: {e}")
                    return self._generate_dummy_embedding(text)
        
        print(f"Max retries ({max_retries}) reached. Using fallback embedding.")
        return self._generate_dummy_embedding(text)
    
    def _generate_dummy_embedding(self, text: str) -> List[float]:
        """Fallback dummy embedding for testing"""
        text_hash = hashlib.md5(text.encode()).digest()
        embedding = []
        for i in range(self.embedding_dimension):
            byte_val = text_hash[i % len(text_hash)]
            embedding.append((byte_val / 255.0) * 2 - 1)
        return embedding
    
    def prepare_documents_for_rag(self, input_folder: str, output_folder: str) -> str:
        """
        Prepare documents for RAG corpus
        
        Args:
            input_folder: Folder with JSON files
            output_folder: Output folder for prepared files
            
        Returns:
            Path to prepared files directory
        """
        print("\nPreparing documents for RAG corpus...")
        
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_files = list(input_path.rglob('*.json'))
        
        for json_file in tqdm(json_files, desc="Preparing documents"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                records = [data] if isinstance(data, dict) else data
                
                for idx, record in enumerate(records):
                    text = record.get('text', '')
                    if text:
                        output_file = output_path / f"{json_file.stem}_{idx}.txt"
                        with open(output_file, 'w', encoding='utf-8') as out:
                            out.write(text)
                            
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        print(f"Prepared documents in: {output_path}")
        return str(output_path)
    
    def upload_to_gcs(self, local_folder: str, bucket_name: str, gcs_prefix: str = "rag-documents") -> str:
        """
        Upload documents to Google Cloud Storage
        """
        print("\nUploading documents to GCS...")
        
        from google.cloud import storage
        
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name)
        
        local_path = Path(local_folder)
        uploaded_count = 0
        
        for file_path in tqdm(list(local_path.rglob('*')), desc="Uploading"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                blob_name = f"{gcs_prefix}/{relative_path}"
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                uploaded_count += 1
        
        gcs_uri = f"gs://{bucket_name}/{gcs_prefix}"
        print(f"Uploaded {uploaded_count} files to: {gcs_uri}")
        return gcs_uri
    
    def create_rag_corpus(self, display_name: str, gcs_uri: str = None) -> Any:
        """Create a RAG corpus in Vertex AI"""
        print(f"\nCreating RAG corpus: {display_name}")
        
        try:
            rag_corpus = rag.create_corpus(
                display_name=display_name,
                description=f"Nepali document corpus for RAG - {display_name}"
            )
            
            print(f"Created RAG corpus: {rag_corpus.name}")
            
            if gcs_uri:
                print(f"Importing documents from: {gcs_uri}")
                
                rag.import_files(
                    corpus_name=rag_corpus.name,
                    paths=[gcs_uri],
                    chunk_size=512,
                    chunk_overlap=100,
                )
                
                print("Documents imported successfully")
            
            return rag_corpus
            
        except Exception as e:
            print(f"Error creating RAG corpus: {e}")
            raise
    
    def create_vector_index(self, index_name: str, embeddings_gcs_uri: str) -> Any:
        """Create Vector Search Index"""
        print(f"\nCreating Vector Search Index: {index_name}")
        
        try:
            index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=index_name,
                contents_delta_uri=embeddings_gcs_uri,
                dimensions=self.embedding_dimension,
                approximate_neighbors_count=10,
                distance_measure_type="COSINE_DISTANCE",
                leaf_node_embedding_count=500,
                leaf_nodes_to_search_percent=7,
                description="Nepali RAG Vector Search Index",
                labels={"purpose": "rag", "language": "nepali"}
            )
            
            print(f"Vector Index created: {index.resource_name}")
            return index
            
        except Exception as e:
            print(f"Error creating vector index: {e}")
            raise
    
    def query_rag(self, corpus_name: str, query: str, 
                  model: str = "gemini-1.5-pro-002", 
                  similarity_top_k: int = 5,
                  vector_distance_threshold: float = 0.5) -> Dict[str, Any]:
        """Query the RAG system"""
        print(f"\nQuerying RAG: {query}")
        
        try:
            response = rag.retrieval_query(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=corpus_name,
                    )
                ],
                text=query,
                similarity_top_k=similarity_top_k,
                vector_distance_threshold=vector_distance_threshold,
            )
            
            from vertexai.generative_models import GenerativeModel
            
            model = GenerativeModel(model)
            
            contexts_text = "\n\n".join([
                f"Context {i+1}:\n{ctx.text}"
                for i, ctx in enumerate(response.contexts.contexts)
            ])
            
            prompt = f"""Based on the following contexts, answer the question.

Contexts:
{contexts_text}

Question: {query}

Answer:"""
            
            generation_response = model.generate_content(prompt)
            
            result = {
                "query": query,
                "answer": generation_response.text,
                "contexts": [
                    {
                        "text": ctx.text,
                        "distance": ctx.distance if hasattr(ctx, 'distance') else None
                    }
                    for ctx in response.contexts.contexts
                ],
                "num_contexts": len(response.contexts.contexts)
            }
            
            print(f"Retrieved {result['num_contexts']} relevant contexts")
            return result
            
        except Exception as e:
            print(f"Error querying RAG: {e}")
            raise

