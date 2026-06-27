import os
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigSource(Enum):
    """Configuration sources"""
    ENV_FILE = "env_file"
    ENVIRONMENT = "environment"
    JSON_FILE = "json_file"
    YAML_FILE = "yaml_file"

@dataclass
class ProcessorConfig:
    """Document AI Processor Configuration"""
    project_id: str
    processor_id: str
    location: str
    processor_version: str
    language_hints: List[str]
    
    def validate(self) -> List[str]:
        """Validate processor configuration"""
        errors = []
        if not self.project_id:
            errors.append("PROJECT_ID is required in .env file")
        if not self.processor_id:
            errors.append("PROCESSOR_ID is required in .env file")
        if not self.location:
            errors.append("PROCESSOR_LOCATION is required in .env file")
        if not self.processor_version:
            errors.append("PROCESSOR_VERSION is required in .env file")
        return errors

@dataclass
class StorageConfig:
    """GCS Storage Configuration"""
    input_gcs_uri: str
    output_gcs_uri: str
    mime_type: str
    
    def validate(self) -> List[str]:
        """Validate storage configuration"""
        errors = []
        if not self.input_gcs_uri or not self.input_gcs_uri.startswith("gs://"):
            errors.append("INPUT_GCS_URI must start with gs:// in .env file")
        if not self.output_gcs_uri or not self.output_gcs_uri.startswith("gs://"):
            errors.append("OUTPUT_GCS_URI must start with gs:// in .env file")
        if len(self.input_gcs_uri.split("/")) < 4:
            errors.append("INPUT_GCS_URI must include bucket and path in .env file")
        return errors

@dataclass
class ProcessingConfig:
    """Processing behavior configuration"""
    wait_for_completion: bool
    extract_text: bool
    save_pages: bool
    poll_interval: int
    max_wait_time: Optional[int]
    
    def validate(self) -> List[str]:
        """Validate processing configuration"""
        errors = []
        if self.poll_interval < 1:
            errors.append("POLL_INTERVAL must be at least 1 second in .env file")
        if self.max_wait_time and self.max_wait_time < 60:
            errors.append("MAX_WAIT_TIME must be at least 60 seconds if set in .env file")
        return errors

@dataclass
class OutputConfig:
    """Output configuration"""
    output_dir: str
    log_file: str
    log_level: str
    save_statistics: bool
    
    def validate(self) -> List[str]:
        """Validate output configuration"""
        errors = []
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            errors.append(f"LOG_LEVEL must be one of {valid_levels} in .env file")
        return errors

@dataclass
class OCRConfig:
    """OCR-specific configuration"""
    enable_native_pdf_parsing: bool
    enable_image_quality_scores: bool
    enable_symbol: bool
    advanced_ocr_options: List[str]
    
    def validate(self) -> List[str]:
        """Validate OCR configuration"""
        return []

@dataclass
class SecurityConfig:
    """Security and authentication configuration"""
    service_account_key_path: Optional[str]
    use_default_credentials: bool
    
    def validate(self) -> List[str]:
        """Validate security configuration"""
        errors = []
        if self.service_account_key_path and not Path(self.service_account_key_path).exists():
            errors.append(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {self.service_account_key_path}")
        return errors

@dataclass
class ChunkingConfig:
    """Text chunking configuration"""
    # Directory configuration
    input_dir: str = "./markdown_output"
    output_dir: str = "./chunked_output"
    
    # Chunking parameters
    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_size: int = 100
    max_chunk_size: int = 1200
    
    # Language-specific settings
    nepali_separators: Tuple[str, ...] = ("\n\n", "\n", "।", "॥", "?", "!", ". ", " ", "")
    
    # Quality control
    validate_chunks: bool = True
    min_nepali_chars_ratio: float = 0.2
    remove_empty_chunks: bool = True
    deduplicate_chunks: bool = True
    
    # Metadata settings
    include_page_numbers: bool = True
    extract_document_metadata: bool = True
    generate_chunk_ids: bool = True
    
    # Performance
    max_files_to_process: Optional[int] = None
    
    # Header splitting configuration
    headers_to_split_on: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("#", "Chapter"),
        ("##", "Section"),
        ("###", "SubSection"),
        ("####", "SubSubSection")
    ])
    
    def validate(self) -> List[str]:
        """Validate chunking configuration"""
        errors = []
        
        if self.chunk_size < 100:
            errors.append("CHUNK_SIZE must be at least 100 characters")
        
        if self.chunk_overlap >= self.chunk_size:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        if self.min_chunk_size > self.chunk_size:
            errors.append("MIN_CHUNK_SIZE must not exceed CHUNK_SIZE")
        
        if self.max_chunk_size < self.chunk_size:
            errors.append("MAX_CHUNK_SIZE must not be less than CHUNK_SIZE")
        
        if not 0 <= self.min_nepali_chars_ratio <= 1:
            errors.append("MIN_NEPALI_CHARS_RATIO must be between 0 and 1")
        
        return errors

@dataclass
class MarkdownConversionConfig:
    """Markdown conversion configuration"""
    input_gcs_uri: str
    output_dir: str = "./markdown_output"
    
    def validate(self) -> List[str]:
        """Validate markdown conversion configuration"""
        errors = []
        if not self.input_gcs_uri or not self.input_gcs_uri.startswith("gs://"):
            errors.append("MARKDOWN_INPUT_GCS_URI must start with gs://")
        return errors

@dataclass
class RAGConfig:
    """RAG and Embedding configuration"""
    # Embedding settings
    embedding_model: str = "text-multilingual-embedding-002"
    
    # Vertex AI settings
    vertex_ai_location: str = "us-central1"
    llm_model: str = "text-bison@002"
    
    # Directory settings
    embeddings_output_dir: str = "./embeddings"
    vector_store_dir: str = "./vector_store"
    
    # Processing settings
    batch_size: int = 5
    max_retries: int = 3
    
    # Vector store settings
    vector_store_type: str = "vertex_ai_matching_engine"
    index_machine_type: str = "e2-standard-16"
    min_replica_count: int = 1
    max_replica_count: int = 2
    
    # Search settings
    similarity_threshold: float = 0.7
    max_retrieved_chunks: int = 5
    
    def validate(self) -> List[str]:
        """Validate RAG configuration"""
        errors = []
        
        # Validate batch size
        if self.batch_size < 1 or self.batch_size > 10:
            errors.append("EMBEDDING_BATCH_SIZE must be between 1 and 10")
        
        # Validate location
        valid_locations = [
            "us",
            "us-central1", 
            "us-east1", 
            "us-west1",
            "europe-west1", 
            "europe-west4",
            "asia-east1",
            "asia-northeast1",
            "asia-southeast1"
        ]
        if self.vertex_ai_location not in valid_locations:
            errors.append(f"VERTEX_AI_LOCATION must be one of: {', '.join(valid_locations)}")
        
        # Validate embedding model
        valid_models = [
            "text-embedding-005",
            "text-multilingual-embedding-002",
            "gemini-embedding-001",
            
        ]
        if self.embedding_model not in valid_models:
            errors.append(f"EMBEDDING_MODEL '{self.embedding_model}' is not recognized. "
                         f"Recommended: text-multilingual-embedding-002 for Nepali")
        
        # Validate similarity threshold
        if not 0 <= self.similarity_threshold <= 1:
            errors.append("SIMILARITY_THRESHOLD must be between 0 and 1")
        
        # Validate max retrieved chunks
        if self.max_retrieved_chunks < 1 or self.max_retrieved_chunks > 20:
            errors.append("MAX_RETRIEVED_CHUNKS must be between 1 and 20")
        
        return errors

@dataclass
class AppConfig:
    """Main application configuration"""
    environment: Environment
    processor: ProcessorConfig
    storage: StorageConfig
    processing: ProcessingConfig
    output: OutputConfig
    ocr: OCRConfig
    security: SecurityConfig
    chunking: ChunkingConfig
    markdown_conversion: MarkdownConversionConfig
    rag: RAGConfig
    
    # Metadata
    config_version: str = "1.0"
    config_source: ConfigSource = ConfigSource.ENV_FILE
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate all configuration sections"""
        all_errors = {}
        
        sections = {
            "processor": self.processor,
            "storage": self.storage,
            "processing": self.processing,
            "output": self.output,
            "ocr": self.ocr,
            "security": self.security,
            "chunking": self.chunking,
            "markdown_conversion": self.markdown_conversion,
            "rag": self.rag
        }
        
        for section_name, section in sections.items():
            errors = section.validate()
            if errors:
                all_errors[section_name] = errors
        
        return all_errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """Load configuration from dictionary"""
        return cls(
            environment=Environment(data.get("environment", "development")),
            processor=ProcessorConfig(**data["processor"]),
            storage=StorageConfig(**data["storage"]),
            processing=ProcessingConfig(**data.get("processing", {})),
            output=OutputConfig(**data.get("output", {})),
            ocr=OCRConfig(**data.get("ocr", {})),
            security=SecurityConfig(**data.get("security", {})),
            chunking=ChunkingConfig(**data.get("chunking", {})),
            markdown_conversion=MarkdownConversionConfig(**data.get("markdown_conversion", {})),
            rag=RAGConfig(**data.get("rag", {})),
            config_version=data.get("config_version", "1.0"),
            config_source=ConfigSource(data.get("config_source", "env_file"))
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'AppConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        config = cls.from_dict(data)
        config.config_source = ConfigSource.JSON_FILE
        return config
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'AppConfig':
        """Load configuration from YAML file"""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config. Install: pip install pyyaml")
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        config = cls.from_dict(data)
        config.config_source = ConfigSource.YAML_FILE
        return config
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables"""
        from dotenv import load_dotenv
        load_dotenv()
        
        # Determine environment
        env_name = os.getenv("ENVIRONMENT", "development")
        environment = Environment(env_name)
        
        # Build configuration from environment variables
        config = cls(
            environment=environment,
            processor=ProcessorConfig(
                project_id=os.getenv("PROJECT_ID", ""),
                processor_id=os.getenv("PROCESSOR_ID", ""),
                location=os.getenv("PROCESSOR_LOCATION", ""),
                processor_version=os.getenv("PROCESSOR_VERSION", ""),
                language_hints=os.getenv("LANGUAGE_HINTS", "").split(",") if os.getenv("LANGUAGE_HINTS") else []
            ),
            storage=StorageConfig(
                input_gcs_uri=os.getenv("INPUT_GCS_URI", ""),
                output_gcs_uri=os.getenv("OUTPUT_GCS_URI", ""),
                mime_type=os.getenv("MIME_TYPE", "")
            ),
            processing=ProcessingConfig(
                wait_for_completion=os.getenv("WAIT_FOR_COMPLETION", "true").lower() == "true",
                extract_text=os.getenv("EXTRACT_TEXT", "true").lower() == "true",
                save_pages=os.getenv("SAVE_PAGES", "true").lower() == "true",
                poll_interval=int(os.getenv("POLL_INTERVAL", "10")),
                max_wait_time=int(os.getenv("MAX_WAIT_TIME")) if os.getenv("MAX_WAIT_TIME") else None
            ),
            output=OutputConfig(
                output_dir=os.getenv("OUTPUT_DIR", "./output"),
                log_file=os.getenv("LOG_FILE", "processing.log"),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                save_statistics=os.getenv("SAVE_STATISTICS", "true").lower() == "true"
            ),
            ocr=OCRConfig(
                enable_native_pdf_parsing=os.getenv("ENABLE_NATIVE_PDF_PARSING", "true").lower() == "true",
                enable_image_quality_scores=os.getenv("ENABLE_IMAGE_QUALITY_SCORES", "false").lower() == "true",
                enable_symbol=os.getenv("ENABLE_SYMBOL", "false").lower() == "true",
                advanced_ocr_options=os.getenv("ADVANCED_OCR_OPTIONS", "").split(",") if os.getenv("ADVANCED_OCR_OPTIONS") else []
            ),
            security=SecurityConfig(
                service_account_key_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                use_default_credentials=os.getenv("USE_DEFAULT_CREDENTIALS", "false").lower() == "true"
            ),
            chunking=ChunkingConfig(
                input_dir=os.getenv("CHUNKING_INPUT_DIR", "./markdown_output"),
                output_dir=os.getenv("CHUNKING_OUTPUT_DIR", "./chunked_output"),
                chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
                min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100")),
                max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "1200")),
                validate_chunks=os.getenv("VALIDATE_CHUNKS", "true").lower() == "true",
                min_nepali_chars_ratio=float(os.getenv("MIN_NEPALI_CHARS_RATIO", "0.2")),
                deduplicate_chunks=os.getenv("DEDUPLICATE_CHUNKS", "true").lower() == "true",
                max_files_to_process=int(os.getenv("MAX_FILES_TO_PROCESS")) if os.getenv("MAX_FILES_TO_PROCESS") else None
            ),
            markdown_conversion=MarkdownConversionConfig(
                input_gcs_uri=os.getenv("MARKDOWN_INPUT_GCS_URI", os.getenv("OUTPUT_GCS_URI", "")),
                output_dir=os.getenv("MARKDOWN_OUTPUT_DIR", "./markdown_output")
            ),
            rag=RAGConfig(
                embedding_model=os.getenv("EMBEDDING_MODEL", "textembedding-gecko@003"),
                embeddings_output_dir=os.getenv("EMBEDDINGS_OUTPUT_DIR", "./embeddings"),
                vector_store_type=os.getenv("VECTOR_STORE_TYPE", "vertex_ai_matching_engine"),
                batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "5")),
                max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
            ),
            config_source=ConfigSource.ENV_FILE
        )
        
        return config
    
    @classmethod
    def load(cls, source: Optional[str] = None) -> 'AppConfig':
        """
        Load configuration from the appropriate source
        
        Args:
            source: Path to config file or None for environment variables
        
        Returns:
            AppConfig instance
        """
        if source is None:
            return cls.from_env()
        
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {source}")
        
        # Determine file type and load
        if source_path.suffix == ".json":
            return cls.from_json(str(source_path))
        elif source_path.suffix in [".yaml", ".yml"]:
            return cls.from_yaml(str(source_path))
        else:
            raise ValueError(f"Unsupported config file format: {source_path.suffix}")

class ConfigManager:
    """Configuration manager with environment-specific configs"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def get_config_path(self, environment: Environment) -> Path:
        """Get configuration file path for environment"""
        return self.config_dir / f"config.{environment.value}.json"
    
    def save_config(self, config: AppConfig, environment: Optional[Environment] = None):
        """Save configuration for specific environment"""
        env = environment or config.environment
        config_path = self.get_config_path(env)
        config.to_json(str(config_path))
        return config_path
    
    def load_config(self, environment: Environment) -> AppConfig:
        """Load configuration for specific environment"""
        config_path = self.get_config_path(environment)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {config_path}")
        return AppConfig.from_json(str(config_path))
    
    def create_default_configs(self):
        """Create configuration files for all environments from .env"""
        from dotenv import load_dotenv
        load_dotenv()
        
        # Create config from .env values
        config = AppConfig.from_env()
        
        # Save config for the environment specified in .env
        path = self.save_config(config, config.environment)
        print(f"✓ Created config from .env: {path}")
        
        # Also create configs for other environments with same values
        other_environments = [env for env in Environment if env != config.environment]
        
        for other_env in other_environments:
            # Create a copy with different environment
            other_config = AppConfig(
                environment=other_env,
                processor=config.processor,
                storage=config.storage,
                processing=config.processing,
                output=config.output,
                ocr=config.ocr,
                security=config.security,
                chunking=config.chunking,
                markdown_conversion=config.markdown_conversion,
                rag=config.rag
            )
            
            # Validate before saving
            errors = other_config.validate()
            if not errors:
                other_path = self.save_config(other_config, other_env)
                print(f"✓ Created config for {other_env.value}: {other_path}")
            else:
                print(f"⚠ Skipped {other_env.value} config: Missing required values in .env")

def get_config(source: Optional[str] = None) -> AppConfig:
    """
    Convenience function to load configuration
    
    Args:
        source: Config file path or None for environment variables
    
    Returns:
        Validated AppConfig instance
    
    Raises:
        ValueError: If configuration is invalid
    """
    config = AppConfig.load(source)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        error_msg = "Configuration validation failed. Check your .env file:\n"
        for section, section_errors in errors.items():
            error_msg += f"\n{section.upper()} section:\n"
            for error in section_errors:
                error_msg += f"  - {error}\n"
        raise ValueError(error_msg)
    
    return config

# Example usage
if __name__ == "__main__":
    import sys
    
    # Create configs from .env
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        try:
            manager = ConfigManager()
            manager.create_default_configs()
            print("\n✓ Configuration files created successfully!")
            print("  Check 'configs/' directory")
        except Exception as e:
            print(f"✗ Failed to create configs: {e}")
            print("\nMake sure your .env file has all required variables:")
            print("  - PROJECT_ID")
            print("  - PROCESSOR_ID")
            print("  - PROCESSOR_LOCATION")
            print("  - INPUT_GCS_URI")
            print("  - OUTPUT_GCS_URI")
            print("  - GOOGLE_APPLICATION_CREDENTIALS")
        sys.exit(0)
    
    # Load and validate config
    try:
        config = get_config()
        print("="*70)
        print("CONFIGURATION LOADED SUCCESSFULLY")
        print("="*70)
        print(f"Environment:       {config.environment.value}")
        print(f"Project ID:        {config.processor.project_id}")
        print(f"Processor ID:      {config.processor.processor_id}")
        print(f"Input GCS:         {config.storage.input_gcs_uri}")
        print(f"Output GCS:        {config.storage.output_gcs_uri}")
        print(f"Markdown Output:   {config.markdown_conversion.output_dir}")
        print(f"Chunking Input:    {config.chunking.input_dir}")
        print(f"Chunking Output:   {config.chunking.output_dir}")
        print(f"Chunk Size:        {config.chunking.chunk_size} chars")
        print(f"Chunk Overlap:     {config.chunking.chunk_overlap} chars")
        print(f"Embedding Model:   {config.rag.embedding_model}")
        print(f"Embeddings Output: {config.rag.embeddings_output_dir}")
        print("="*70)
    except Exception as e:
        print("="*70)
        print("CONFIGURATION ERROR")
        print("="*70)
        print(f"{e}")
        print("="*70)
        sys.exit(1)