#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resilient Nepali RAG System Deployment
- Auto-retry with exponential backoff for quota issues
- Progress tracking and resume capability
- Waits and retries until completion
"""

from google.cloud import storage
from vertexai.preview import rag
import vertexai
from pathlib import Path
import json
import sys
import os
import time
from tqdm import tqdm
from datetime import datetime

# Configuration
PROJECT_ID = "gen-lang-client-0000379298"
REGION = "us-east1"
BUCKET_NAME = "rag_nepali"
INPUT_FOLDER = "nepali_ocr_data"
CORPUS_NAME = "nepali-documents-resilient"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retry configuration
UPLOAD_BATCH_SIZE = 5  # Smaller batches for safer operation
MAX_RETRIES = 10  # Maximum retry attempts per batch
INITIAL_RETRY_DELAY = 60  # Start with 60 seconds
MAX_RETRY_DELAY = 600  # Cap at 10 minutes
BATCH_DELAY = 3  # Delay between successful batches
MAX_EMBEDDING_REQUESTS_PER_MIN = 200  # Very conservative

# Progress tracking file
PROGRESS_FILE = "deployment_progress.json"

print("=" * 70)
print("Initializing Vertex AI...")
print("=" * 70)
vertexai.init(project=PROJECT_ID, location=REGION)
print(f"Connected to project: {PROJECT_ID}")
print(f"Region: {REGION}\n")


def load_progress():
    """Load progress from previous run"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"uploaded_files": [], "imported_batches": [], "corpus_name": None}
    return {"uploaded_files": [], "imported_batches": [], "corpus_name": None}


def save_progress(progress):
    """Save progress for resume capability"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def wait_with_countdown(seconds, reason="Rate limit"):
    """Wait with a countdown display"""
    print(f"\n⏳ {reason} - Waiting {seconds} seconds...")
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        print(f"\r   Time remaining: {mins:02d}:{secs:02d}", end='', flush=True)
        time.sleep(1)
    print("\r   ✓ Wait complete" + " " * 20)


def retry_with_backoff(func, *args, max_retries=MAX_RETRIES, operation_name="Operation", **kwargs):
    """
    Execute function with exponential backoff retry on quota errors
    Returns: (success: bool, result: any, error: str)
    """
    retry_delay = INITIAL_RETRY_DELAY
    
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            return True, result, None
            
        except Exception as e:
            error_msg = str(e)
            is_quota_error = any(x in error_msg.lower() for x in 
                               ['quota exceeded', '429', 'rate limit', 'resourceexhausted'])
            
            if is_quota_error:
                if attempt < max_retries - 1:
                    print(f"\n⚠ {operation_name} hit rate limit (attempt {attempt + 1}/{max_retries})")
                    wait_with_countdown(retry_delay, "Rate limit cooldown")
                    
                    # Exponential backoff with jitter
                    retry_delay = min(retry_delay * 1.5, MAX_RETRY_DELAY)
                else:
                    print(f"\n✗ {operation_name} failed after {max_retries} attempts")
                    return False, None, error_msg
            else:
                # Non-quota error - fail immediately
                print(f"\n✗ {operation_name} failed: {error_msg[:150]}")
                return False, None, error_msg
    
    return False, None, "Max retries exceeded"


def prepare_and_upload_documents():
    """Prepare documents and upload to GCS"""
    print("=" * 70)
    print("Step 1: Preparing and uploading documents")
    print("=" * 70)
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"✗ Error: Input folder '{INPUT_FOLDER}' not found!")
        sys.exit(1)
    
    progress = load_progress()
    already_uploaded = set(progress.get("uploaded_files", []))
    
    storage_client = storage.Client(project=PROJECT_ID)
    
    try:
        bucket = storage_client.get_bucket(BUCKET_NAME)
        print(f"✓ Using bucket: gs://{BUCKET_NAME}")
    except Exception as e:
        print(f"Creating bucket: gs://{BUCKET_NAME}")
        bucket = storage_client.create_bucket(BUCKET_NAME, location=REGION)
        print(f"✓ Bucket created")
    
    local_path = Path(INPUT_FOLDER)
    gcs_prefix = "rag-documents-resilient"
    
    json_files = list(local_path.rglob('*.json'))
    
    if not json_files:
        print(f"✗ No JSON files found in {INPUT_FOLDER}")
        sys.exit(1)
    
    print(f"\n✓ Found {len(json_files)} JSON files")
    print(f"  Target: gs://{BUCKET_NAME}/{gcs_prefix}")
    print(f"  Already uploaded: {len(already_uploaded)} files")
    
    uploaded_files = list(already_uploaded)
    errors = []
    
    print("\nProcessing files...\n")
    
    for json_file in tqdm(json_files, desc="Processing"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = [data] if isinstance(data, dict) else data
            
            for idx, record in enumerate(records):
                text = record.get('text', '')
                if text and len(text.strip()) > 50:
                    blob_name = f"{gcs_prefix}/{json_file.stem}_{idx}.txt"
                    
                    # Skip if already uploaded
                    if blob_name in already_uploaded:
                        continue
                    
                    blob = bucket.blob(blob_name)
                    cleaned_text = text.strip()
                    
                    blob.metadata = {
                        'source_file': str(json_file),
                        'record_index': str(idx),
                        'upload_time': datetime.now().isoformat()
                    }
                    
                    blob.upload_from_string(cleaned_text, content_type='text/plain')
                    uploaded_files.append(blob_name)
                    
                    # Save progress periodically
                    if len(uploaded_files) % 10 == 0:
                        progress["uploaded_files"] = uploaded_files
                        save_progress(progress)
                    
        except Exception as e:
            errors.append((json_file, str(e)))
    
    # Final progress save
    progress["uploaded_files"] = uploaded_files
    save_progress(progress)
    
    print("\n" + "=" * 70)
    print("Upload Summary:")
    print(f"  Total files uploaded: {len(uploaded_files)}")
    if errors:
        print(f"  Errors: {len(errors)}")
    print(f"  Location: gs://{BUCKET_NAME}/{gcs_prefix}/")
    print("=" * 70)
    
    if len(uploaded_files) == 0:
        print("\n✗ No documents uploaded")
        sys.exit(1)
    
    return f"gs://{BUCKET_NAME}/{gcs_prefix}", uploaded_files


def create_rag_corpus_with_retry(gcs_uri, uploaded_files):
    """Create RAG corpus and import files with resilient retry logic"""
    print("\n" + "=" * 70)
    print("Step 2: Creating RAG Corpus")
    print("=" * 70)
    
    progress = load_progress()
    
    # Check for existing corpus
    corpus = None
    if progress.get("corpus_name"):
        print(f"Checking for existing corpus: {progress['corpus_name']}")
        try:
            # Try to use existing corpus
            existing_corpora = list(rag.list_corpora())
            for existing_corpus in existing_corpora:
                if existing_corpus.name == progress["corpus_name"]:
                    corpus = existing_corpus
                    print(f"✓ Found existing corpus: {corpus.display_name}")
                    break
        except Exception as e:
            print(f"⚠ Could not check existing corpus: {e}")
    
    # Create new corpus if needed
    if not corpus:
        try:
            existing_corpora = list(rag.list_corpora())
            for existing_corpus in existing_corpora:
                if existing_corpus.display_name == CORPUS_NAME:
                    response = input(f"\nCorpus '{CORPUS_NAME}' exists. Use it? (yes/no): ").strip().lower()
                    if response in ['yes', 'y']:
                        corpus = existing_corpus
                        break
        except Exception as e:
            print(f"⚠ Error checking corpora: {e}")
        
        if not corpus:
            print(f"Creating new corpus: {CORPUS_NAME}")
            corpus = rag.create_corpus(
                display_name=CORPUS_NAME,
                description=f"Nepali documents - resilient deployment"
            )
            print(f"✓ Created: {corpus.name}")
            
            progress["corpus_name"] = corpus.name
            save_progress(progress)
    
    print("\n" + "=" * 70)
    print("Step 3: Importing Documents (Resilient Mode)")
    print("=" * 70)
    print(f"Total files: {len(uploaded_files)}")
    print(f"Batch size: {UPLOAD_BATCH_SIZE}")
    print(f"Max retries per batch: {MAX_RETRIES}")
    print("\n⏳ This will take time but will complete automatically...")
    print("   (Safe to leave running - progress is saved)\n")
    
    file_paths = [f"gs://{BUCKET_NAME}/{blob_name}" for blob_name in uploaded_files]
    
    imported_batches = set(progress.get("imported_batches", []))
    total_imported = 0
    failed_batches = []
    
    num_batches = (len(file_paths) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE
    
    print(f"Starting import of {num_batches} batches...\n")
    
    for batch_idx in range(0, len(file_paths), UPLOAD_BATCH_SIZE):
        batch_num = batch_idx // UPLOAD_BATCH_SIZE
        
        # Skip already imported batches
        if batch_num in imported_batches:
            print(f"Batch {batch_num + 1}/{num_batches}: Already imported ✓")
            total_imported += min(UPLOAD_BATCH_SIZE, len(file_paths) - batch_idx)
            continue
        
        batch = file_paths[batch_idx:batch_idx + UPLOAD_BATCH_SIZE]
        
        print(f"\nBatch {batch_num + 1}/{num_batches}: Importing {len(batch)} files...")
        
        def import_batch():
            return rag.import_files(
                corpus_name=corpus.name,
                paths=batch,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                max_embedding_requests_per_min=MAX_EMBEDDING_REQUESTS_PER_MIN,
            )
        
        success, import_response, error = retry_with_backoff(
            import_batch,
            operation_name=f"Batch {batch_num + 1}",
            max_retries=MAX_RETRIES
        )
        
        if success:
            imported_count = import_response.imported_rag_files_count
            total_imported += imported_count
            imported_batches.add(batch_num)
            
            # Save progress
            progress["imported_batches"] = list(imported_batches)
            save_progress(progress)
            
            print(f"  ✓ Imported {imported_count} files")
            print(f"  Progress: {total_imported}/{len(file_paths)} files ({100*total_imported//len(file_paths)}%)")
            
            # Brief pause between successful batches
            if batch_num < num_batches - 1:
                time.sleep(BATCH_DELAY)
        else:
            failed_batches.append((batch_num, error))
            print(f"  ✗ Batch {batch_num + 1} failed after all retries")
            
            # Ask user if they want to continue
            response = input("\nContinue with remaining batches? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("\n⚠ Import paused. Run script again to resume.")
                break
    
    print("\n" + "=" * 70)
    print("Import Summary:")
    print(f"  Successfully imported: {total_imported}/{len(file_paths)} files")
    print(f"  Completed batches: {len(imported_batches)}/{num_batches}")
    if failed_batches:
        print(f"  Failed batches: {len(failed_batches)}")
        for batch_num, error in failed_batches[:3]:
            print(f"    - Batch {batch_num + 1}: {error[:80]}")
    print("=" * 70)
    
    if len(imported_batches) == num_batches:
        print("\n✓ ALL FILES IMPORTED SUCCESSFULLY!")
        # Clear progress file on complete success
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print("  (Progress file cleared)")
    else:
        print("\n⚠ Import incomplete. Run script again to resume from checkpoint.")
    
    print("\n⏳ Indexing in progress (5-10 minutes)...")
    
    return corpus


def test_rag_query_with_retry(corpus_name):
    """Test RAG with retry logic"""
    print("\n" + "=" * 70)
    print("Step 4: Testing RAG System")
    print("=" * 70)
    
    print("\nWaiting 15 seconds for indexing to start...")
    time.sleep(15)
    
    try:
        from vertexai.generative_models import GenerativeModel
        model = GenerativeModel("gemini-2.0-flash")
        print(f"✓ Using model: gemini-2.0-flash\n")
    except Exception as e:
        print(f"⚠ Could not load model: {e}")
        print("Skipping test queries")
        return
    
    test_queries = [
        ("नेपाल", "Simple Nepali word"),
        ("Nepal", "Simple English word"),
        ("नेपालको इतिहास", "Nepali phrase"),
    ]
    
    print("Testing retrieval with retry logic...\n")
    
    for query, description in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}' ({description})")
        print('='*70)
        
        def test_query():
            return rag.retrieval_query(
                rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
                text=query,
                similarity_top_k=5,
                vector_distance_threshold=0.5,
            )
        
        success, response, error = retry_with_backoff(
            test_query,
            operation_name="Retrieval",
            max_retries=3
        )
        
        if success:
            num_contexts = len(response.contexts.contexts)
            if num_contexts > 0:
                print(f"✓ Retrieved {num_contexts} contexts")
                print(f"\nFirst context preview:")
                print(f"{response.contexts.contexts[0].text[:200]}...")
            else:
                print("⚠ No contexts found - indexing may still be in progress")
        else:
            print(f"✗ Query failed: {error[:150]}")
            if "quota" in error.lower():
                print("⚠ Still hitting rate limits - wait longer before querying")
    
    print(f"\n{'='*70}")
    print("Testing completed")
    print('='*70)


def save_deployment_info(corpus):
    """Save deployment information"""
    print("\n" + "=" * 70)
    print("Step 5: Saving Deployment Information")
    print("=" * 70)
    
    info = {
        "project_id": PROJECT_ID,
        "region": REGION,
        "bucket_name": BUCKET_NAME,
        "corpus_name": corpus.name,
        "corpus_display_name": corpus.display_name,
        "gcs_path": f"gs://{BUCKET_NAME}/rag-documents-resilient/",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_embedding_requests_per_min": MAX_EMBEDDING_REQUESTS_PER_MIN,
        "deployment_timestamp": datetime.now().isoformat()
    }
    
    with open("rag_deployment_info.json", "w", encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved to: rag_deployment_info.json")
    print("\nDeployment Details:")
    for key, value in info.items():
        print(f"  {key}: {value}")


def main():
    """Main deployment flow with resume capability"""
    print("=" * 70)
    print("Resilient Nepali RAG System Deployment")
    print("=" * 70)
    print("\n🔄 Features:")
    print("  • Auto-retry on quota errors")
    print("  • Progress tracking (can resume if interrupted)")
    print("  • Smart exponential backoff")
    print("  • Waits and retries until completion")
    
    print(f"\n📋 Configuration:")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Region: {REGION}")
    print(f"  Bucket: {BUCKET_NAME}")
    print(f"  Input: {INPUT_FOLDER}")
    print(f"  Corpus: {CORPUS_NAME}")
    print(f"  Batch size: {UPLOAD_BATCH_SIZE} files")
    print(f"  Max retries: {MAX_RETRIES} per batch")
    print("=" * 70)
    
    # Check for previous progress
    if os.path.exists(PROGRESS_FILE):
        print("\n✓ Found previous progress - will resume from checkpoint")
    
    confirm = input("\nProceed with deployment? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Deployment cancelled")
        sys.exit(0)
    
    try:
        start_time = time.time()
        
        gcs_uri, uploaded_files = prepare_and_upload_documents()
        corpus = create_rag_corpus_with_retry(gcs_uri, uploaded_files)
        test_rag_query_with_retry(corpus.name)
        save_deployment_info(corpus)
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("✓ RAG SYSTEM DEPLOYMENT COMPLETE!")
        print("=" * 70)
        print(f"\nCorpus: {corpus.name}")
        print(f"Display Name: {corpus.display_name}")
        print(f"Files: {len(uploaded_files)}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        
        print("\n📝 Next Steps:")
        print("   1. Wait 5-10 minutes for full indexing")
        print("   2. Test queries: python query_rag_improved.py")
        print("   3. Check status: python diagnostic_tool.py")
        print("\n" + "=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        print("✓ Progress saved - run script again to resume")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Deployment failed: {e}")
        print("\n✓ Progress saved - run script again to retry")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()