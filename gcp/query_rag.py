#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resilient Nepali RAG System Deployment
- Ensures ALL files are imported with verification
- Auto-retry with exponential backoff for quota issues
- Progress tracking and resume capability
- Continuous retry until ALL files are imported
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
UPLOAD_BATCH_SIZE = 3  # Smaller batches for reliability
MAX_RETRIES = 20  # More retries for persistent issues
INITIAL_RETRY_DELAY = 60
MAX_RETRY_DELAY = 600
BATCH_DELAY = 5
VERIFICATION_DELAY = 10  # Time to wait before verification
MAX_EMBEDDING_REQUESTS_PER_MIN = 150  # Very conservative

# Progress tracking file
PROGRESS_FILE = "deployment_progress.json"
MISSING_FILES_FILE = "missing_files.json"  # Track which files are missing

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
            return {"uploaded_files": [], "successfully_imported": [], "corpus_name": None, "retry_attempts": 0}
    return {"uploaded_files": [], "successfully_imported": [], "corpus_name": None, "retry_attempts": 0}


def save_progress(progress):
    """Save progress for resume capability"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def load_missing_files():
    """Load missing files from previous runs"""
    if os.path.exists(MISSING_FILES_FILE):
        try:
            with open(MISSING_FILES_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"missing_files": [], "total_attempts": 0}
    return {"missing_files": [], "total_attempts": 0}


def save_missing_files(missing_data):
    """Save missing files data"""
    with open(MISSING_FILES_FILE, 'w') as f:
        json.dump(missing_data, f, indent=2)


def wait_with_countdown(seconds, reason="Rate limit"):
    """Wait with a countdown display"""
    print(f"\n[WAITING] {reason} - {seconds} seconds...")
    for remaining in range(seconds, 0, -5):
        mins, secs = divmod(remaining, 60)
        print(f"  Time remaining: {mins:02d}:{secs:02d}")
        time.sleep(min(5, remaining))
    print("  [DONE] Wait complete\n")


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
                    print(f"\n[WARNING] {operation_name} hit rate limit (attempt {attempt + 1}/{max_retries})")
                    wait_with_countdown(retry_delay, "Rate limit cooldown")
                    retry_delay = min(retry_delay * 1.5, MAX_RETRY_DELAY)
                else:
                    print(f"\n[ERROR] {operation_name} failed after {max_retries} attempts")
                    return False, None, error_msg
            else:
                print(f"\n[ERROR] {operation_name} failed: {error_msg[:150]}")
                return False, None, error_msg
    
    return False, None, "Max retries exceeded"


def prepare_and_upload_documents():
    """Prepare documents and upload to GCS"""
    print("=" * 70)
    print("Step 1: Preparing and uploading documents")
    print("=" * 70)
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input folder '{INPUT_FOLDER}' not found!")
        sys.exit(1)
    
    progress = load_progress()
    already_uploaded = set(progress.get("uploaded_files", []))
    
    storage_client = storage.Client(project=PROJECT_ID)
    
    try:
        bucket = storage_client.get_bucket(BUCKET_NAME)
        print(f"[OK] Using bucket: gs://{BUCKET_NAME}")
    except Exception as e:
        print(f"Creating bucket: gs://{BUCKET_NAME}")
        bucket = storage_client.create_bucket(BUCKET_NAME, location=REGION)
        print(f"[OK] Bucket created")
    
    local_path = Path(INPUT_FOLDER)
    gcs_prefix = "rag-documents-resilient"
    
    json_files = list(local_path.rglob('*.json'))
    
    if not json_files:
        print(f"[ERROR] No JSON files found in {INPUT_FOLDER}")
        sys.exit(1)
    
    print(f"\n[INFO] Found {len(json_files)} JSON files")
    print(f"[INFO] Target: gs://{BUCKET_NAME}/{gcs_prefix}")
    print(f"[INFO] Already uploaded: {len(already_uploaded)} files")
    
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
                    
                    if len(uploaded_files) % 10 == 0:
                        progress["uploaded_files"] = uploaded_files
                        save_progress(progress)
                    
        except Exception as e:
            errors.append((json_file, str(e)))
    
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
        print("\n[ERROR] No documents uploaded")
        sys.exit(1)
    
    return f"gs://{BUCKET_NAME}/{gcs_prefix}", uploaded_files


def verify_imported_files(corpus_name):
    """Verify which files are actually imported in the corpus"""
    print("\n[INFO] Verifying imported files...")
    try:
        imported_files = list(rag.list_files(corpus_name=corpus_name))
        imported_names = set([f.display_name for f in imported_files])
        print(f"[OK] Found {len(imported_names)} files in corpus")
        return imported_names
    except Exception as e:
        print(f"[WARNING] Could not verify files: {e}")
        return set()


def retry_import_missing_files(corpus, file_paths, file_basenames, already_imported):
    """
    Keep retrying to import missing files until ALL are imported
    Returns: (newly_imported_count, updated_imported_set)
    """
    progress = load_progress()
    missing_data = load_missing_files()
    
    # Find which files are missing
    missing_indices = [i for i, basename in enumerate(file_basenames) 
                      if basename not in already_imported]
    
    if not missing_indices:
        return 0, already_imported
    
    missing_paths = [file_paths[i] for i in missing_indices]
    missing_names = [file_basenames[i] for i in missing_indices]
    
    print(f"\n[RETRY] Attempting to import {len(missing_names)} missing files")
    print(f"[RETRY] Missing files: {', '.join(missing_names[:5])}{'...' if len(missing_names) > 5 else ''}")
    
    # Update missing files tracking
    missing_data["missing_files"] = missing_names
    missing_data["total_attempts"] = missing_data.get("total_attempts", 0) + 1
    save_missing_files(missing_data)
    
    total_imported = len(already_imported)
    newly_imported = set()
    
    # Try importing missing files in small batches
    for batch_idx in range(0, len(missing_paths), UPLOAD_BATCH_SIZE):
        batch_paths = missing_paths[batch_idx:batch_idx + UPLOAD_BATCH_SIZE]
        batch_names = missing_names[batch_idx:batch_idx + UPLOAD_BATCH_SIZE]
        
        batch_num = (batch_idx // UPLOAD_BATCH_SIZE) + 1
        total_batches = (len(missing_paths) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE
        
        print(f"\n  Retry batch {batch_num}/{total_batches}: {len(batch_paths)} files")
        print(f"    Files: {', '.join(batch_names)}")
        
        def import_batch():
            return rag.import_files(
                corpus_name=corpus.name,
                paths=batch_paths,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                max_embedding_requests_per_min=MAX_EMBEDDING_REQUESTS_PER_MIN,
            )
        
        success, import_response, error = retry_with_backoff(
            import_batch,
            operation_name=f"Retry batch {batch_num}",
            max_retries=MAX_RETRIES
        )
        
        if success:
            imported_count = import_response.imported_rag_files_count
            print(f"    [OK] Import API returned: {imported_count} files")
            
            # Wait before verifying this batch
            time.sleep(VERIFICATION_DELAY)
            
            # Verify which files were actually imported
            current_imported = verify_imported_files(corpus.name)
            newly_added = current_imported - already_imported
            
            if newly_added:
                print(f"    [VERIFIED] New files in corpus: {len(newly_added)}")
                for name in newly_added:
                    print(f"      ✓ {name}")
                
                already_imported.update(newly_added)
                newly_imported.update(newly_added)
                
                # Update progress
                progress["successfully_imported"] = list(already_imported)
                save_progress(progress)
                
                total_imported = len(already_imported)
                print(f"    [PROGRESS] Total imported: {total_imported}/{len(file_basenames)}")
            else:
                print(f"    [WARNING] No new files verified in corpus yet")
            
            time.sleep(BATCH_DELAY)
        else:
            print(f"    [ERROR] Batch failed: {error[:100]}")
    
    return len(newly_imported), already_imported


def create_rag_corpus_with_retry(gcs_uri, uploaded_files):
    """Create RAG corpus and import ALL files with continuous retry"""
    print("\n" + "=" * 70)
    print("Step 2: Creating RAG Corpus")
    print("=" * 70)
    
    progress = load_progress()
    missing_data = load_missing_files()
    
    # Get or create corpus
    corpus = None
    if progress.get("corpus_name"):
        print(f"Checking for existing corpus: {progress['corpus_name']}")
        try:
            existing_corpora = list(rag.list_corpora())
            for existing_corpus in existing_corpora:
                if existing_corpus.name == progress["corpus_name"]:
                    corpus = existing_corpus
                    print(f"[OK] Found existing corpus: {corpus.display_name}")
                    break
        except Exception as e:
            print(f"[WARNING] Could not check existing corpus: {e}")
    
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
            print(f"[WARNING] Error checking corpora: {e}")
        
        if not corpus:
            print(f"Creating new corpus: {CORPUS_NAME}")
            corpus = rag.create_corpus(
                display_name=CORPUS_NAME,
                description=f"Nepali documents - resilient deployment"
            )
            print(f"[OK] Created: {corpus.name}")
            
            progress["corpus_name"] = corpus.name
            save_progress(progress)
    
    print("\n" + "=" * 70)
    print("Step 3: Importing ALL Documents (Continuous Retry)")
    print("=" * 70)
    print(f"Total files to import: {len(uploaded_files)}")
    print(f"Batch size: {UPLOAD_BATCH_SIZE}")
    print(f"Max retries per batch: {MAX_RETRIES}")
    print(f"Previous retry attempts: {missing_data.get('total_attempts', 0)}")
    print("\nThis will keep retrying until ALL files are imported...")
    print("(Safe to leave running - progress is saved)\n")
    
    # Convert to full GCS paths
    file_paths = [f"gs://{BUCKET_NAME}/{blob_name}" for blob_name in uploaded_files]
    file_basenames = [blob_name.split('/')[-1] for blob_name in uploaded_files]
    
    # Get current imported files
    already_imported = set(progress.get("successfully_imported", []))
    current_imported = verify_imported_files(corpus.name)
    already_imported.update(current_imported)
    
    progress["successfully_imported"] = list(already_imported)
    save_progress(progress)
    
    print(f"\n[INITIAL] Files already in corpus: {len(already_imported)}/{len(file_paths)}")
    
    if len(already_imported) < len(file_paths):
        missing_count = len(file_paths) - len(already_imported)
        print(f"[MISSING] {missing_count} files need to be imported")
    
    max_total_attempts = 50  # Maximum total retry attempts across all rounds
    consecutive_failures = 0
    last_success_count = len(already_imported)
    
    for attempt in range(max_total_attempts):
        print(f"\n{'='*70}")
        print(f"Retry Attempt {attempt + 1}/{max_total_attempts}")
        print(f"{'='*70}")
        
        # Update current state
        current_imported = verify_imported_files(corpus.name)
        already_imported.update(current_imported)
        
        if len(already_imported) == len(file_paths):
            print(f"\n[SUCCESS] All {len(file_paths)} files are imported!")
            break
        
        print(f"\n[BEFORE] Files imported: {len(already_imported)}/{len(file_paths)}")
        print(f"[BEFORE] Files remaining: {len(file_paths) - len(already_imported)}")
        
        # Retry importing missing files
        newly_imported_count, already_imported = retry_import_missing_files(
            corpus, file_paths, file_basenames, already_imported
        )
        
        print(f"\n[AFTER] Files imported: {len(already_imported)}/{len(file_paths)}")
        
        if newly_imported_count > 0:
            consecutive_failures = 0
            last_success_count = len(already_imported)
            print(f"[PROGRESS] Imported {newly_imported_count} new files")
        else:
            consecutive_failures += 1
            print(f"[WARNING] No new files imported this attempt")
            print(f"[INFO] Consecutive failures: {consecutive_failures}")
            
            if consecutive_failures >= 3:
                print(f"\n[WARNING] Multiple consecutive failures - increasing wait time")
                wait_with_countdown(300, "Extended cooldown")
                consecutive_failures = 0
        
        # Check if we're done
        if len(already_imported) == len(file_paths):
            print(f"\n[SUCCESS] All {len(file_paths)} files are imported!")
            break
        
        # Wait before next attempt, with increasing delay
        wait_time = 30 + (attempt * 5)  # Increasing delay
        wait_time = min(wait_time, 300)  # Cap at 5 minutes
        
        if attempt < max_total_attempts - 1:
            print(f"\n[INFO] Waiting {wait_time} seconds before next attempt...")
            time.sleep(wait_time)
    
    # Final verification
    print(f"\n{'='*70}")
    print("Final Verification")
    print(f"{'='*70}")
    
    time.sleep(VERIFICATION_DELAY)
    final_imported = verify_imported_files(corpus.name)
    
    print(f"\n[FINAL] Files in corpus: {len(final_imported)}/{len(file_paths)}")
    
    if len(final_imported) == len(file_paths):
        print("[SUCCESS] ALL FILES VERIFIED IN CORPUS!")
        # Clean up progress files
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        if os.path.exists(MISSING_FILES_FILE):
            os.remove(MISSING_FILES_FILE)
    else:
        print(f"[WARNING] Only {len(final_imported)}/{len(file_paths)} files verified")
        print("Missing files:")
        missing = set(file_basenames) - final_imported
        for i, name in enumerate(sorted(missing), 1):
            print(f"  {i}. {name}")
        
        # Save missing files for next run
        missing_data = {
            "missing_files": list(missing),
            "total_attempts": missing_data.get("total_attempts", 0),
            "last_attempt": datetime.now().isoformat()
        }
        save_missing_files(missing_data)
        
        print(f"\n[ACTION REQUIRED] Run the script again to continue importing missing files")
        print(f"[TIP] Wait 5-10 minutes and run again - sometimes indexing takes time")
    
    print(f"\n[INFO] Indexing in progress (5-10 minutes)...")
    
    return corpus, len(final_imported)


def save_deployment_info(corpus, files_count, total_files):
    """Save deployment information"""
    print("\n" + "=" * 70)
    print("Step 4: Saving Deployment Information")
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
        "files_imported": files_count,
        "total_files": total_files,
        "completion_percentage": f"{(files_count/total_files*100):.1f}%",
        "deployment_timestamp": datetime.now().isoformat()
    }
    
    with open("rag_deployment_info.json", "w", encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved to: rag_deployment_info.json")


def main():
    """Main deployment flow"""
    print("=" * 70)
    print("Resilient Nepali RAG System Deployment")
    print("=" * 70)
    print("\nFeatures:")
    print("  - Auto-retry on quota errors")
    print("  - Verifies ALL files are imported")
    print("  - Continuous retry until complete")
    print("  - Progress tracking (resume if interrupted)")
    
    print(f"\nConfiguration:")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Region: {REGION}")
    print(f"  Bucket: {BUCKET_NAME}")
    print(f"  Input: {INPUT_FOLDER}")
    print(f"  Corpus: {CORPUS_NAME}")
    print(f"  Batch size: {UPLOAD_BATCH_SIZE} files")
    print(f"  Max retries: {MAX_RETRIES} per batch")
    print(f"  Max total attempts: 50")
    print("=" * 70)
    
    if os.path.exists(PROGRESS_FILE):
        print("\n[INFO] Found previous progress - will resume")
    
    missing_data = load_missing_files()
    if missing_data.get("missing_files"):
        print(f"[INFO] Found {len(missing_data['missing_files'])} missing files from previous run")
        print(f"[INFO] Previous retry attempts: {missing_data.get('total_attempts', 0)}")
    
    confirm = input("\nProceed? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled")
        sys.exit(0)
    
    try:
        start_time = time.time()
        
        gcs_uri, uploaded_files = prepare_and_upload_documents()
        corpus, final_count = create_rag_corpus_with_retry(gcs_uri, uploaded_files)
        save_deployment_info(corpus, final_count, len(uploaded_files))
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("DEPLOYMENT COMPLETE")
        print("=" * 70)
        print(f"\nCorpus: {corpus.name}")
        print(f"Files imported: {final_count}/{len(uploaded_files)}")
        print(f"Completion: {(final_count/len(uploaded_files)*100):.1f}%")
        print(f"Total time: {elapsed/60:.1f} minutes")
        
        if final_count == len(uploaded_files):
            print("\n[SUCCESS] All files imported successfully!")
            print("[INFO] Progress files have been cleaned up")
        else:
            print(f"\n[WARNING] Only {final_count}/{len(uploaded_files)} files imported")
            print("[ACTION] Run script again to continue importing missing files")
            print("[TIP] Sometimes Vertex AI needs time to process - wait 10 minutes and retry")
        
        print("\nNext steps:")
        print("  1. Wait 5-10 minutes for indexing")
        print("  2. Test: python query_rag_improved.py")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted - progress saved")
        print("Run the script again to resume")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nDeployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()