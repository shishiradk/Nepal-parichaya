import sys
import time
import json
import logging
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from google.cloud import documentai, storage
from google.api_core.client_options import ClientOptions
from google.api_core import retry, exceptions
from google.cloud.exceptions import GoogleCloudError

from config import AppConfig, get_config

# tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("INFO: Install tqdm: pip install tqdm")
    
# Configuration
def setup_logging(config: AppConfig) -> logging.Logger:
    """Configure structured logging from config"""
    logger = logging.getLogger("batch_ocr_processor")
    logger.setLevel(config.output.log_level.upper())
    
    # clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if config.output.log_file:
        file_handler = logging.FileHandler(config.output.log_file)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
    return logger

class BatchOCRProcessor:
    """Production-ready batch OCR processor for Nepali documents"""
    
    def __init__(self, config: AppConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = None
        self.storage_client = None
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize Google Cloud clients with retry configuration"""
        try:
            # set credentials
            if self.config.security.service_account_key_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.security.service_account_key_path
            
            opts = ClientOptions(
                api_endpoint=f"{self.config.processor.location}-documentai.googleapis.com"
            )
            self.client = documentai.DocumentProcessorServiceClient(client_options=opts)
            self.storage_client = storage.Client(project=self.config.processor.project_id)
            self.logger.info("Successfully initialized Google Cloud clients")
        except Exception as e:
            self.logger.error(f"Failed to initialize clients: {e}")
            raise
        
    @retry.Retry(
        predicate=retry.if_exception_type(
            exceptions.ServiceUnavailable,
            exceptions.DeadlineExceeded,
            exceptions.InternalServerError
        ),
        initial=1.0,
        maximum=60.0,
        multiplier=2.0,
        deadline=300.0
    )
    def start_batch_process(self) -> Dict:
        """
        Start batch processing operation with retry logic

        Returns:
            Dict with operation details
        """
        try:
            self.logger.info("="*70)
            self.logger.info("STARTING BATCH OCR PROCESS")
            self.logger.info("="*70)
            self.logger.info(f"Environment: {self.config.environment.value}")
            self.logger.info(f"Input URI: {self.config.storage.input_gcs_uri}")
            self.logger.info(f"Output URI: {self.config.storage.output_gcs_uri}")
            self.logger.info(f"Language hints: {self.config.processor.language_hints}")
            self.logger.info(f"Processor version: {self.config.processor.processor_version}")
            
            # Build processor version path
            name = self.client.processor_version_path(
                self.config.processor.project_id,
                self.config.processor.location,
                self.config.processor.processor_id,
                self.config.processor.processor_version
            )
            
            # Configure batch input
            gcs_document = documentai.GcsDocument(
                gcs_uri=self.config.storage.input_gcs_uri,
                mime_type=self.config.storage.mime_type
            )
            gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
            input_config = documentai.BatchDocumentsInputConfig(
                gcs_documents=gcs_documents
            )
            
            # Configure output
            output_config = documentai.DocumentOutputConfig(
                gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                    gcs_uri=self.config.storage.output_gcs_uri
                )
            )
            
            # Configure OCR options from config
            ocr_hints = documentai.OcrConfig.Hints(
                language_hints=self.config.processor.language_hints
            )
            ocr_config = documentai.OcrConfig(
                hints=ocr_hints,
                enable_native_pdf_parsing=self.config.ocr.enable_native_pdf_parsing,
                enable_image_quality_scores=self.config.ocr.enable_image_quality_scores,
                enable_symbol=self.config.ocr.enable_symbol,
                advanced_ocr_options=self.config.ocr.advanced_ocr_options
            )
            process_options = documentai.ProcessOptions(
                ocr_config=ocr_config
            )
            
            # Create batch request
            request = documentai.BatchProcessRequest(
                name=name,
                input_documents=input_config,
                document_output_config=output_config,
                process_options=process_options,
            )
            
            # Start operation
            operation = self.client.batch_process_documents(request)
            operation_name = operation.operation.name
            
            self.logger.info(f"Operation started: {operation_name}")
            self.logger.info("Processing continues in background")
            self.logger.info("Monitor progress at: https://console.cloud.google.com/ai/document-ai")
            
            return {
                "success": True,
                "operation_name": operation_name,
                "operation": operation,
                "start_time": time.time()
            }
        
        except exceptions.GoogleAPICallError as e:
            self.logger.error(f"API call failed: {e}")
            return {"success": False, "error": str(e), "error_type": "api_error"}
        except Exception as e:
            self.logger.error(f"Unexpected error starting batch process: {e}")
            return {"success": False, "error": str(e), "error_type": "unexpected"}
            
    def get_operation_status(self, operation_name: str) -> Dict:
        """
        Get current status of a batch operation
       
        Args:
            operation_name: Full operation name
           
        Returns:
            Dict with status information
        """
        try:
            # get the operation
            operation = self.client._transport.operations_client.get_operation(
                name=operation_name
            )
            
            # check if done
            if operation.done:
                if operation.error:
                    return {
                        "status": "FAILED",
                        "done": True,
                        "error": operation.error.message,
                        "error_code": operation.error.code
                    }
                else:
                    return {
                        "status": "COMPLETED",
                        "done": True,
                        "error": None
                    }
            else:
                return {
                    "status": "RUNNING",
                    "done": False,
                    "error": None
                }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "done": True,
                "error": str(e)
            }
    
    def wait_for_completion_with_progress(self, operation, start_time: float) -> Dict:
        """
        Wait for batch operation to complete with live progress display
        
        Args:
            operation: Long-running operation object
            start_time: Operation start timestamp
        
        Returns:
            Dict with completion status
        """
        poll_interval = self.config.processing.poll_interval
        max_wait_time = self.config.processing.max_wait_time
        
        self.logger.info("Waiting for operation to complete...")
        self.logger.info(f"Poll interval: {poll_interval}s")
        
        if max_wait_time:
            self.logger.info(f"Maximum wait time: {max_wait_time}s")
            
        try:
            operation_name = operation.operation.name
            last_update_time = time.time()
            status_counts = {"check_count": 0, "running_count": 0}
            
            # initialize progress bar if tqdm is available
            if TQDM_AVAILABLE and max_wait_time:
                pbar = tqdm(
                    total=max_wait_time,
                    desc="OCR Processing",
                    unit="s",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]"
                )
            else:
                pbar = None
                
            while not operation.done():
                elapsed = time.time() - start_time
                status_counts["check_count"] += 1
                
                # Check timeout
                if max_wait_time and elapsed > max_wait_time:
                    self.logger.warning(f"Maximum wait time ({max_wait_time}s) exceeded")
                    if pbar:
                        pbar.close()
                    return {
                        "success": False,
                        "error": "timeout",
                        "elapsed_time": elapsed
                    }
                
                # Update progress bar
                if pbar:
                    pbar.n = min(int(elapsed), max_wait_time)
                    pbar.refresh()
                    
                # Show status update every 30 seconds or poll_interval
                if time.time() - last_update_time >= 30:
                    status = self.get_operation_status(operation_name)
                    if status["status"] == "RUNNING":
                        status_counts["running_count"] += 1
                        self.logger.info(
                            f"Status: STILL PROCESSING - Check {status_counts['check_count']}, "
                            f"Elapsed: {elapsed/60:.1f} minutes"
                        )
                    last_update_time = time.time()
                    
                # Sleep before next check
                sleep_time = min(poll_interval, 10)
                time.sleep(sleep_time)
            
            # Operation completed
            elapsed_time = time.time() - start_time
            
            # Close progress bar
            if pbar:
                pbar.n = min(int(elapsed_time), max_wait_time)
                pbar.refresh()
                pbar.close()
                
            # Check for errors
            if operation.exception():
                error_msg = str(operation.exception())
                self.logger.error(f"Operation failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "elapsed_time": elapsed_time
                }
                
            self.logger.info("="*70)
            self.logger.info("BATCH PROCESSING COMPLETE")
            self.logger.info("="*70)
            self.logger.info(f"Processing time: {elapsed_time/60:.1f} minutes")
            self.logger.info(f"Output location: {self.config.storage.output_gcs_uri}")
            
            return {
                "success": True,
                "operation_name": operation.operation.name,
                "output_uri": self.config.storage.output_gcs_uri,
                "elapsed_time": elapsed_time
            }
            
        except Exception as e:
            if pbar:
                pbar.close()
            self.logger.error(f"Error waiting for completion: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }
        
    def wait_for_completion(self, operation, start_time: float) -> Dict:
        """
        Wait for batch operation to complete with progress monitoring
        
        Args:
            operation: Long-running operation object
            start_time: Operation start timestamp
        
        Returns:
            Dict with completion status
        """
        return self.wait_for_completion_with_progress(operation, start_time)
    
    def monitor_operation(self, operation_name: str, interval: int = 30):
        """
        Monitor an operation with live updates
        
        Args:
            operation_name: Full operation name
            interval: Check interval in seconds
        """
        print(f"\nMonitoring operation: {operation_name}")
        print("-" * 70)
        
        start_time = time.time()
        check_count = 0
        
        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=100,
                desc="Checking Status",
                bar_format="{l_bar}{bar}| {desc}"
            )
        else:
            pbar = None
        
        try:
            while True:
                check_count += 1
                status = self.get_operation_status(operation_name)
                elapsed = time.time() - start_time
                
                if status["status"] == "RUNNING":
                    if pbar:
                        pbar.set_description(f"RUNNING - Check {check_count}, Elapsed: {elapsed/60:.1f}m")
                        pbar.refresh()
                    else:
                        print(f"\rStatus: RUNNING - Check {check_count}, Elapsed: {elapsed/60:.1f} minutes", end="")
                
                elif status["status"] == "COMPLETED":
                    if pbar:
                        pbar.set_description(f"COMPLETED - Total time: {elapsed/60:.1f}m")
                        pbar.close()
                    print(f"\n\nProcessing COMPLETED successfully!")
                    print(f"Total time: {elapsed/60:.1f} minutes")
                    break
                
                elif status["status"] == "FAILED":
                    if pbar:
                        pbar.set_description(f"FAILED - Error: {status['error']}")
                        pbar.close()
                    print(f"\n\nProcessing FAILED!")
                    print(f"Error: {status['error']}")
                    break
                
                else:
                    if pbar:
                        pbar.close()
                    print(f"\n\nStatus check ERROR: {status['error']}")
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            if pbar:
                pbar.close()
            print("\n\nMonitoring stopped by user")
    
    def extract_text_with_progress(self) -> Optional[Dict]:
        """
        Extract text from batch processing results with progress bar
        
        Returns:
            Dict with extraction statistics
        """
        try:
            self.logger.info("="*70)
            self.logger.info("EXTRACTING TEXT FROM RESULTS")
            self.logger.info("="*70)
            
            # Create output directory
            output_path = Path(self.config.output.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if self.config.processing.save_pages:
                pages_path = output_path / "pages"
                pages_path.mkdir(exist_ok=True)
            
            # Parse GCS URI
            bucket_name, prefix = self._parse_gcs_uri(self.config.storage.output_gcs_uri)
            bucket = self.storage_client.bucket(bucket_name)
            
            # List all JSON result files
            blobs = list(bucket.list_blobs(prefix=prefix))
            json_blobs = [b for b in blobs if b.name.endswith('.json')]
            
            if not json_blobs:
                self.logger.warning(f"No JSON files found at {self.config.storage.output_gcs_uri}")
                return None
            
            self.logger.info(f"Found {len(json_blobs)} result files")
            
            all_texts = []
            total_pages = 0
            total_chars = 0
            devanagari_chars = 0
            failed_files = []
            
            # Create progress bar for file processing
            if TQDM_AVAILABLE:
                file_pbar = tqdm(
                    total=len(json_blobs),
                    desc="Extracting text",
                    unit="file",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )
            else:
                file_pbar = None
            
            for idx, blob in enumerate(json_blobs, 1):
                try:
                    # Update progress
                    if file_pbar:
                        file_pbar.set_description(f"Processing: {Path(blob.name).name[:30]}...")
                        file_pbar.update(1)
                    else:
                        self.logger.info(f"Processing file {idx}/{len(json_blobs)}: {blob.name}")
                    
                    # Download and parse
                    content = blob.download_as_string()
                    doc_data = json.loads(content)
                    
                    if 'text' not in doc_data:
                        self.logger.warning(f"No text found in {blob.name}")
                        continue
                    
                    text = doc_data['text']
                    all_texts.append(text)
                    
                    # Save individual file
                    filename = Path(blob.name).stem + '.txt'
                    output_file = output_path / filename
                    output_file.write_text(text, encoding='utf-8')
                    
                    # Count pages
                    if 'pages' in doc_data:
                        page_count = len(doc_data['pages'])
                        total_pages += page_count
                        
                        # Save individual pages
                        if self.config.processing.save_pages:
                            self._save_pages(
                                doc_data['pages'], 
                                text, 
                                pages_path, 
                                filename
                            )
                    
                    # Update statistics
                    total_chars += len(text)
                    devanagari_chars += sum(
                        1 for char in text if '\u0900' <= char <= '\u097F'
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {blob.name}: {e}")
                    failed_files.append(blob.name)
                    continue
            
            if file_pbar:
                file_pbar.close()
            
            # Save combined text
            if all_texts:
                combined_text = "\n\n".join(all_texts)
                combined_file = output_path / "all_text_combined.txt"
                combined_file.write_text(combined_text, encoding='utf-8')
                
                # Calculate statistics
                nepali_percentage = (devanagari_chars / total_chars * 100) if total_chars > 0 else 0
                
                stats = {
                    "documents_processed": len(all_texts),
                    "total_pages": total_pages,
                    "total_characters": total_chars,
                    "devanagari_characters": devanagari_chars,
                    "nepali_percentage": nepali_percentage,
                    "failed_files": len(failed_files),
                    "output_directory": str(output_path),
                    "combined_file": str(combined_file),
                    "environment": self.config.environment.value
                }
                
                # Log summary
                self.logger.info("="*70)
                self.logger.info("EXTRACTION SUMMARY")
                self.logger.info("="*70)
                self.logger.info(f"Documents processed: {stats['documents_processed']}")
                self.logger.info(f"Total pages: {stats['total_pages']}")
                self.logger.info(f"Total characters: {stats['total_characters']:,}")
                self.logger.info(f"Nepali characters: {stats['devanagari_characters']:,} ({nepali_percentage:.1f}%)")
                self.logger.info(f"Combined text: {combined_file}")
                
                if failed_files:
                    self.logger.warning(f"Failed files: {len(failed_files)}")
                    for failed in failed_files:
                        self.logger.warning(f"  - {failed}")
                
                # Save stats to JSON if configured
                if self.config.output.save_statistics:
                    stats_file = output_path / "extraction_stats.json"
                    stats_file.write_text(json.dumps(stats, indent=2), encoding='utf-8')
                
                return stats
            else:
                self.logger.error("No text extracted from any files")
                return None
                
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return None
    
    def extract_text(self) -> Optional[Dict]:
        """
        Extract text from batch processing results using config settings
        
        Returns:
            Dict with extraction statistics
        """
        if TQDM_AVAILABLE:
            return self.extract_text_with_progress()
        else:
            try:
                self.logger.info("="*70)
                self.logger.info("EXTRACTING TEXT FROM RESULTS")
                self.logger.info("="*70)
                
                # Create output directory
                output_path = Path(self.config.output.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                if self.config.processing.save_pages:
                    pages_path = output_path / "pages"
                    pages_path.mkdir(exist_ok=True)
                
                # Parse GCS URI
                bucket_name, prefix = self._parse_gcs_uri(self.config.storage.output_gcs_uri)
                bucket = self.storage_client.bucket(bucket_name)
                
                # List all JSON result files
                blobs = list(bucket.list_blobs(prefix=prefix))
                json_blobs = [b for b in blobs if b.name.endswith('.json')]
                
                if not json_blobs:
                    self.logger.warning(f"No JSON files found at {self.config.storage.output_gcs_uri}")
                    return None
                
                self.logger.info(f"Found {len(json_blobs)} result files")
                
                all_texts = []
                total_pages = 0
                total_chars = 0
                devanagari_chars = 0
                failed_files = []
                
                for idx, blob in enumerate(json_blobs, 1):
                    try:
                        self.logger.info(f"Processing file {idx}/{len(json_blobs)}: {blob.name}")
                        
                        # Download and parse
                        content = blob.download_as_string()
                        doc_data = json.loads(content)
                        
                        if 'text' not in doc_data:
                            self.logger.warning(f"No text found in {blob.name}")
                            continue
                        
                        text = doc_data['text']
                        all_texts.append(text)
                        
                        # Save individual file
                        filename = Path(blob.name).stem + '.txt'
                        output_file = output_path / filename
                        output_file.write_text(text, encoding='utf-8')
                        
                        # Count pages
                        if 'pages' in doc_data:
                            page_count = len(doc_data['pages'])
                            total_pages += page_count
                            
                            # Save individual pages
                            if self.config.processing.save_pages:
                                self._save_pages(
                                    doc_data['pages'], 
                                    text, 
                                    pages_path, 
                                    filename
                                )
                        
                        # Update statistics
                        total_chars += len(text)
                        devanagari_chars += sum(
                            1 for char in text if '\u0900' <= char <= '\u097F'
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process {blob.name}: {e}")
                        failed_files.append(blob.name)
                        continue
                
                # Save combined text
                if all_texts:
                    combined_text = "\n\n".join(all_texts)
                    combined_file = output_path / "all_text_combined.txt"
                    combined_file.write_text(combined_text, encoding='utf-8')
                    
                    # Calculate statistics
                    nepali_percentage = (devanagari_chars / total_chars * 100) if total_chars > 0 else 0
                    
                    stats = {
                        "documents_processed": len(all_texts),
                        "total_pages": total_pages,
                        "total_characters": total_chars,
                        "devanagari_characters": devanagari_chars,
                        "nepali_percentage": nepali_percentage,
                        "failed_files": len(failed_files),
                        "output_directory": str(output_path),
                        "combined_file": str(combined_file),
                        "environment": self.config.environment.value
                    }
                    
                    # Log summary
                    self.logger.info("="*70)
                    self.logger.info("EXTRACTION SUMMARY")
                    self.logger.info("="*70)
                    self.logger.info(f"Documents processed: {stats['documents_processed']}")
                    self.logger.info(f"Total pages: {stats['total_pages']}")
                    self.logger.info(f"Total characters: {stats['total_characters']:,}")
                    self.logger.info(f"Nepali characters: {stats['devanagari_characters']:,} ({nepali_percentage:.1f}%)")
                    self.logger.info(f"Combined text: {combined_file}")
                    
                    if failed_files:
                        self.logger.warning(f"Failed files: {len(failed_files)}")
                        for failed in failed_files:
                            self.logger.warning(f"  - {failed}")
                    
                    # Save stats to JSON if configured
                    if self.config.output.save_statistics:
                        stats_file = output_path / "extraction_stats.json"
                        stats_file.write_text(json.dumps(stats, indent=2), encoding='utf-8')
                    
                    return stats
                else:
                    self.logger.error("No text extracted from any files")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Text extraction failed: {e}")
                return None
    
    def _parse_gcs_uri(self, uri: str) -> Tuple[str, str]:
        """Parse GCS URI into bucket and prefix"""
        parts = uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket_name, prefix
    
    def _save_pages(
        self, 
        pages: List, 
        full_text: str, 
        pages_dir: Path, 
        base_filename: str
    ):
        """Save individual page text files"""
        for page_num, page in enumerate(pages, 1):
            try:
                page_text = ""
                if 'layout' in page and 'textAnchor' in page['layout']:
                    segments = page['layout']['textAnchor'].get('textSegments', [])
                    for segment in segments:
                        start = int(segment.get('startIndex', 0))
                        end = int(segment.get('endIndex', 0))
                        page_text += full_text[start:end]
                
                if page_text.strip():
                    page_filename = f"{Path(base_filename).stem}_page_{page_num:04d}.txt"
                    page_file = pages_dir / page_filename
                    page_file.write_text(page_text, encoding='utf-8')
                    
            except Exception as e:
                self.logger.warning(f"Failed to save page {page_num}: {e}")


def main():
    """Main execution function with config management"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Batch OCR Processor")
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to config file (JSON/YAML). Uses environment variables if not specified.'
    )
    parser.add_argument(
        '--env',
        type=str,
        choices=['development', 'staging', 'production'],
        help='Environment to use (alternative to --config)'
    )
    parser.add_argument(
        '--wait',
        action='store_true',
        help='Wait for completion and show live progress'
    )
    parser.add_argument(
        '--monitor',
        type=str,
        help='Monitor an existing operation by operation name'
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.env:
            from config import ConfigManager, Environment
            manager = ConfigManager()
            config = manager.load_config(Environment(args.env))
            print(f"Loaded configuration for {args.env} environment")
        elif args.config:
            config = get_config(args.config)
            print(f"Loaded configuration from {args.config}")
        else:
            config = get_config()
            print("Loaded configuration from environment variables")
        
        # Setup logging
        logger = setup_logging(config)
        logger.info(f"Configuration loaded for {config.environment.value} environment")
        
        # Log configuration summary
        logger.info(f"Project: {config.processor.project_id}")
        logger.info(f"Processor: {config.processor.processor_id}")
        logger.info(f"Input: {config.storage.input_gcs_uri}")
        logger.info(f"Output: {config.storage.output_gcs_uri}")
        
        # Initialize processor
        processor = BatchOCRProcessor(config, logger)
        
        # Handle monitor mode
        if args.monitor:
            print(f"\nStarting monitoring for operation: {args.monitor}")
            processor.monitor_operation(args.monitor, interval=30)
            sys.exit(0)
        
        # Start batch process
        result = processor.start_batch_process()
        
        if not result.get("success"):
            logger.error(f"Failed to start batch process: {result.get('error')}")
            sys.exit(1)
        
        # Show operation info
        operation_name = result["operation_name"]
        print(f"\nOperation started: {operation_name}")
        print(f"To monitor: python {sys.argv[0]} --monitor '{operation_name}'")
        
        # Wait for completion if configured or --wait flag
        should_wait = args.wait or config.processing.wait_for_completion
        if should_wait:
            completion_result = processor.wait_for_completion(
                result["operation"],
                result["start_time"]
            )
            
            if not completion_result.get("success"):
                logger.error(f"Processing failed: {completion_result.get('error')}")
                sys.exit(1)
            
            # Extract text if configured
            if config.processing.extract_text:
                extraction_stats = processor.extract_text()
                
                if extraction_stats:
                    logger.info("Processing pipeline completed successfully")
                else:
                    logger.warning("Text extraction completed with issues")
        else:
            logger.info("Batch process started in async mode")
            logger.info(f"Operation name: {operation_name}")
            logger.info(f"Run with --wait to wait for completion")
            logger.info(f"Or monitor with: python {sys.argv[0]} --monitor '{operation_name}'")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except GoogleCloudError as e:
        print(f"Google Cloud error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
