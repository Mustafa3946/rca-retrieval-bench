"""
Generate embeddings for log messages and index in Azure AI Search.
"""
import os
import json
import ssl
import httpx
import time
from openai import AzureOpenAI, RateLimitError
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, SearchFieldDataType,
    VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration
)
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Disable SSL verification for corporate proxy
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["AZURE_CORE_DISABLE_VERIFY_SSL"] = "1"

def create_search_index(search_endpoint=None, search_key=None):
    """Create Azure AI Search index with vector support."""
    endpoint = search_endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
    key = search_key or os.getenv("AZURE_SEARCH_ADMIN_KEY")
    index_name = "log-embeddings"
    
    # Use custom transport to disable SSL verification
    transport = RequestsTransport(connection_verify=False)
    index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key), transport=transport)
    
    # Define index schema
    fields = [
        SimpleField(name="log_id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="timestamp", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="node", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="message", type=SearchFieldDataType.String),
        SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                   searchable=True, vector_search_dimensions=1536, vector_search_profile_name="vector-profile")
    ]
    
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")],
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")]
    )
    
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    
    print(f"Creating index: {index_name}")
    index_client.create_or_update_index(index)
    print("Index created")

def generate_embeddings(conn_str=None, openai_key=None, openai_endpoint=None, search_endpoint=None, search_key=None):
    """Load logs, generate embeddings, upload to search index."""
    # Setup clients with SSL bypass
    http_client = httpx.Client(verify=False)
    openai_client = AzureOpenAI(
        api_key=openai_key or os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        azure_endpoint=openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
        http_client=http_client
    )
    
    blob_service = BlobServiceClient.from_connection_string(
        conn_str or os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        connection_verify=False
    )
    
    transport = RequestsTransport(connection_verify=False)
    search_client = SearchClient(
        endpoint=search_endpoint or os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name="log-embeddings",
        credential=AzureKeyCredential(search_key or os.getenv("AZURE_SEARCH_ADMIN_KEY")),
        transport=transport
    )
    
    # Download parsed logs
    print("Loading parsed logs...")
    blob_client = blob_service.get_blob_client(container="logs-raw", blob="hdfs/parsed.jsonl")
    data = blob_client.download_blob().readall().decode('utf-8')
    logs = [json.loads(line) for line in data.split('\n') if line.strip()]
    
    # Process all logs with increased capacity (120K TPM)
    print(f"Loaded {len(logs)} logs, processing all logs with 120K TPM capacity")
    
    # Process in batches with retry logic
    batch_size = 50  # Larger batch size for 120K TPM capacity
    documents = []
    
    print(f"Generating embeddings for {len(logs)} logs...")
    for i in tqdm(range(0, len(logs), batch_size)):
        batch = logs[i:i+batch_size]
        messages = [log["message"] for log in batch]
        
        # Get embeddings with retry logic
        max_retries = 5
        retry_delay = 60
        for attempt in range(max_retries):
            try:
                response = openai_client.embeddings.create(
                    input=messages,
                    model="text-embedding-3-small"
                )
                break
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"\nRate limit hit, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
        
        # Prepare documents
        for log, embedding_obj in zip(batch, response.data):
            documents.append({
                "log_id": log["log_id"],
                "timestamp": log["timestamp"],
                "node": log["node"],
                "message": log["message"],
                "embedding": embedding_obj.embedding
            })
        
        # Upload batch to search
        if len(documents) >= batch_size:
            search_client.upload_documents(documents)
            documents = []
        
        # Add small delay between batches to avoid rate limits
        time.sleep(5)
    
    # Upload remaining
    if documents:
        search_client.upload_documents(documents)
    
    print("Embeddings indexed successfully")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        conn_str, openai_key, openai_endpoint, search_endpoint, search_key = sys.argv[1:6]
        create_search_index(search_endpoint, search_key)
        generate_embeddings(conn_str, openai_key, openai_endpoint, search_endpoint, search_key)
    else:
        create_search_index()
        generate_embeddings()
