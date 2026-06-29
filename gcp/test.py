#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpus Inspector
See exactly what files are indexed in your RAG corpus
"""

import vertexai
from vertexai.preview import rag
from google.cloud import storage
import json
import sys

# Load deployment info
try:
    with open("rag_deployment_info.json", "r") as f:
        deployment = json.load(f)
except FileNotFoundError:
    print("Error: rag_deployment_info.json not found")
    sys.exit(1)

PROJECT_ID = deployment["project_id"]
REGION = deployment["region"]
CORPUS_NAME = deployment["corpus_name"]
BUCKET_NAME = deployment.get("bucket_name", "")

vertexai.init(project=PROJECT_ID, location=REGION)


def list_corpus_files():
    """List all files in the corpus"""
    print("=" * 70)
    print("Listing Files in RAG Corpus")
    print("=" * 70)
    print(f"Corpus: {CORPUS_NAME}\n")
    
    try:
        files = rag.list_files(corpus_name=CORPUS_NAME)
        file_list = list(files)
        
        print(f"Total files in corpus: {len(file_list)}\n")
        
        if len(file_list) == 0:
            print("WARNING: Corpus is EMPTY!")
            print("\nPossible reasons:")
            print("  1. Import still in progress (wait 10-15 minutes)")
            print("  2. Import failed")
            print("  3. Wrong corpus name")
            return []
        
        print("Files in corpus:")
        print("-" * 70)
        
        for i, file in enumerate(file_list, 1):
            print(f"{i}. {file.name}")
            if hasattr(file, 'display_name'):
                print(f"   Display name: {file.display_name}")
        
        return file_list
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def list_gcs_files():
    """List all files in GCS bucket"""
    print("\n" + "=" * 70)
    print("Listing Files in GCS Bucket")
    print("=" * 70)
    
    if not BUCKET_NAME:
        print("Bucket name not in deployment info")
        return []
    
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        prefixes = ["rag-documents", "rag-documents-v2"]
        all_blobs = []
        
        for prefix in prefixes:
            blobs = list(bucket.list_blobs(prefix=prefix))
            if blobs:
                print(f"\nPrefix: {prefix}/")
                print(f"Files: {len(blobs)}")
                all_blobs.extend(blobs)
                
                print("\nFirst 10 files:")
                for i, blob in enumerate(blobs[:10], 1):
                    print(f"  {i}. {blob.name} ({blob.size} bytes)")
        
        if not all_blobs:
            print("\nWARNING: No files found in GCS bucket!")
        
        return all_blobs
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def compare_local_vs_corpus(input_folder="nepali_ocr_data"):
    """Compare local files vs what's in corpus"""
    from pathlib import Path
    
    print("\n" + "=" * 70)
    print("Comparing Local Files vs Corpus")
    print("=" * 70)
    
    # Get local files
    if not Path(input_folder).exists():
        print(f"Local folder not found: {input_folder}")
        return
    
    json_files = list(Path(input_folder).rglob('*.json'))
    print(f"\nLocal JSON files: {len(json_files)}")
    
    # Get corpus files
    try:
        corpus_files = rag.list_files(corpus_name=CORPUS_NAME)
        corpus_file_list = list(corpus_files)
        print(f"Files in corpus: {len(corpus_file_list)}")
        
        # Compare
        if len(json_files) > 0 and len(corpus_file_list) == 0:
            print("\nPROBLEM: You have local files but corpus is empty!")
            print("Solution: Run the deployment script to upload and index")
        
        elif len(corpus_file_list) < len(json_files):
            print(f"\nWARNING: Only {len(corpus_file_list)} out of {len(json_files)} files indexed")
            print("Some files might not have been uploaded or indexed")
        
        else:
            print("\nAll files appear to be indexed")
        
    except Exception as e:
        print(f"Error: {e}")


def sample_corpus_content():
    """Sample actual content from corpus files"""
    print("\n" + "=" * 70)
    print("Sampling Content from Corpus Files")
    print("=" * 70)
    
    try:
        files = rag.list_files(corpus_name=CORPUS_NAME)
        file_list = list(files)[:5]  # First 5 files
        
        if not file_list:
            print("No files in corpus to sample")
            return
        
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        for i, file in enumerate(file_list, 1):
            print(f"\n[{i}] File: {file.name}")
            
            # Try to extract blob name from file.name
            # file.name format: projects/.../locations/.../ragCorpora/.../ragFiles/...
            # We need to find the actual GCS path
            
            # Try both prefixes
            for prefix in ["rag-documents", "rag-documents-v2"]:
                blobs = list(bucket.list_blobs(prefix=prefix, max_results=100))
                
                for blob in blobs:
                    if file.display_name in blob.name if hasattr(file, 'display_name') else False:
                        content = blob.download_as_text(encoding='utf-8')
                        print(f"    Content preview: {content[:300]}...")
                        break
        
    except Exception as e:
        print(f"Error: {e}")


def test_retrieval_coverage():
    """Test how many different files get retrieved"""
    print("\n" + "=" * 70)
    print("Testing Retrieval Coverage")
    print("=" * 70)
    
    test_queries = [
        "नेपाल",
        "इतिहास",
        "संविधान",
        "प्रान्त",
        "जनसंख्या",
        "राष्ट्रिय",
        "विकास",
        "सरकार",
    ]
    
    all_retrieved_files = set()
    
    for query in test_queries:
        try:
            response = rag.retrieval_query(
                rag_resources=[rag.RagResource(rag_corpus=CORPUS_NAME)],
                text=query,
                similarity_top_k=10,
                vector_distance_threshold=0.0,
            )
            
            num_contexts = len(response.contexts.contexts)
            print(f"\nQuery: '{query}' -> {num_contexts} contexts")
            
            # Try to extract file names from contexts
            for ctx in response.contexts.contexts:
                if hasattr(ctx, 'source'):
                    all_retrieved_files.add(ctx.source)
        
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n{'='*70}")
    print(f"Unique files retrieved across all queries: {len(all_retrieved_files)}")
    
    # Compare with total files
    try:
        files = rag.list_files(corpus_name=CORPUS_NAME)
        total_files = len(list(files))
        
        if len(all_retrieved_files) < total_files:
            percentage = (len(all_retrieved_files) / total_files) * 100
            print(f"Coverage: {percentage:.1f}% ({len(all_retrieved_files)}/{total_files} files)")
            print(f"\nWARNING: Only {len(all_retrieved_files)} out of {total_files} files are being retrieved!")
            print("Many files might not be reachable with current queries")
    
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main menu"""
    print("\n" + "=" * 70)
    print("Corpus Inspector Menu")
    print("=" * 70)
    print("\nOptions:")
    print("  1. List files in corpus")
    print("  2. List files in GCS bucket")
    print("  3. Compare local vs corpus")
    print("  4. Sample corpus content")
    print("  5. Test retrieval coverage")
    print("  6. Run all diagnostics")
    print("  7. Exit")
    
    choice = input("\nChoice: ").strip()
    
    if choice == "1":
        list_corpus_files()
    elif choice == "2":
        list_gcs_files()
    elif choice == "3":
        compare_local_vs_corpus()
    elif choice == "4":
        sample_corpus_content()
    elif choice == "5":
        test_retrieval_coverage()
    elif choice == "6":
        list_corpus_files()
        list_gcs_files()
        compare_local_vs_corpus()
        test_retrieval_coverage()
    elif choice == "7":
        return
    else:
        print("Invalid choice")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        print("Running full diagnostic...\n")
        list_corpus_files()
        list_gcs_files()
        compare_local_vs_corpus()
        test_retrieval_coverage()
    else:
        main()