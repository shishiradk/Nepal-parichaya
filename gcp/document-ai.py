import os
from google.cloud import storage
from markitdown import MarkItDown
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
GCS_OUTPUT_URI = os.getenv("GCS_OUTPUT_URI", "gs://gcp_document_1/output/nepali_ocr_results/")
LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "markdown_output")
SERVICE_ACCOUNT_KEY = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./service-account-key.json")

# --- Setup ---
# Set credentials environment variable if not already set
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY

# Verify credentials file exists
if not os.path.exists(SERVICE_ACCOUNT_KEY):
    raise FileNotFoundError(
        f"Service account key file not found at: {SERVICE_ACCOUNT_KEY}\n"
        f"Please ensure your service account JSON key is in the correct location.\n"
        f"You can set the path in your .env file: GOOGLE_APPLICATION_CREDENTIALS=path/to/your/key.json"
    )

storage_client = storage.Client()
md_converter = MarkItDown()

def parse_gcs_uri(uri):
    """
    Extract bucket name and prefix from GCS URI.
    E.g., "gs://bucket_name/folder/" -> bucket_name, folder/
    """
    parts = uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix

def extract_text_from_docai_json(json_path):
    """
    Extract text from Document AI JSON output.
    Document AI JSON has a specific structure with 'text' field containing full text.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Document AI structure typically has document.text or pages[].text
        if isinstance(data, dict):
            # Check for standard Document AI response structure
            if 'text' in data:
                return data['text']
            elif 'document' in data and 'text' in data['document']:
                return data['document']['text']
            elif 'pages' in data:
                # Extract text from pages
                text_parts = []
                for page in data['pages']:
                    if 'text' in page:
                        text_parts.append(page['text'])
                return '\n\n'.join(text_parts)
        
        # Fallback: return JSON as formatted text
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    except Exception as e:
        print(f"   Warning: Could not parse Document AI JSON structure: {e}")
        return None

def convert_json_to_markdown():
    """
    Downloads Document AI JSON output from GCS and converts it to Markdown.
    """
    bucket_name, prefix = parse_gcs_uri(GCS_OUTPUT_URI)
    bucket = storage_client.bucket(bucket_name)
    
    # 1. Create local output directory
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    print(f"Searching for JSON files in gs://{bucket_name}/{prefix}")
    
    # 2. List all JSON files - use recursive listing to find nested files
    blobs = list(bucket.list_blobs(prefix=prefix))
    json_blobs = [blob for blob in blobs if blob.name.endswith(".json")]
    
    if not json_blobs:
        print(f"No JSON files found in gs://{bucket_name}/{prefix}")
        return
    
    print(f"Found {len(json_blobs)} JSON file(s) to process.\n")
    
    processed_count = 0
    error_count = 0
    
    # 3. Process each JSON file
    for blob in json_blobs:
        print(f"Processing: {blob.name}")
        
        try:
            # Download the JSON file locally (use temp directory that works on both Windows and Unix)
            import tempfile
            temp_dir = tempfile.gettempdir()
            local_json_path = os.path.join(temp_dir, f"temp_{processed_count}_{os.path.basename(blob.name)}")
            blob.download_to_filename(local_json_path)
            
            # Try to extract text from Document AI JSON first
            extracted_text = extract_text_from_docai_json(local_json_path)
            
            if extracted_text:
                # Use extracted text directly as markdown
                markdown_content = extracted_text
                print(f"   -> Extracted text from Document AI JSON")
            else:
                # Fallback: Try MarkItDown conversion
                markdown_result = md_converter.convert(local_json_path)
                markdown_content = markdown_result.text_content
                print(f"   -> Converted using MarkItDown")
            
            # 4. Save the Markdown content
            # Preserve directory structure from GCS in output
            relative_path = blob.name.replace(prefix, "").lstrip("/")
            md_filename = os.path.splitext(relative_path)[0] + ".md"
            local_md_path = os.path.join(LOCAL_OUTPUT_DIR, md_filename)
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(local_md_path), exist_ok=True)
            
            with open(local_md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"   -> Saved Markdown to {local_md_path}\n")
            processed_count += 1
            
        except Exception as e:
            print(f"   ERROR processing {blob.name}: {e}\n")
            error_count += 1
        
        finally:
            # Clean up temporary JSON file
            if os.path.exists(local_json_path):
                os.remove(local_json_path)
    
    # 5. Summary
    print(f"\n{'='*50}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*50}")
    print(f"Successfully processed: {processed_count} file(s)")
    print(f"Errors encountered: {error_count} file(s)")
    print(f"Output directory: {os.path.abspath(LOCAL_OUTPUT_DIR)}")
    print(f"{'='*50}\n")

# --- Execution ---
if __name__ == "__main__":
    try:
        convert_json_to_markdown()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()