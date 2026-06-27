import json
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

def generate_id_from_data(record: Dict[str, Any], filename: str = None) -> str:
    """
    Generate a unique ID from record data or filename.
    """
    # Try to use existing fields as ID
    if 'uri' in record:
        # Use URI as base for ID
        uri = record['uri']
        # Clean URI to make valid ID
        return uri.replace('gs://', '').replace('/', '_').replace('.', '_')
    elif 'id' in record:
        return str(record['id'])
    elif filename:
        # Use filename without extension
        return Path(filename).stem
    else:
        # Generate hash from record content
        content = json.dumps(record, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()


def generate_dummy_embedding(text: str, dimension: int = 768) -> List[float]:
    """
    Generate a dummy embedding vector from text.
    In production, you should use a real embedding model like sentence-transformers.
    This is just a placeholder that creates deterministic values based on text hash.
    """
    # Create a hash from text
    text_hash = hashlib.md5(text.encode()).digest()
    
    # Convert hash bytes to floats between -1 and 1
    embedding = []
    for i in range(dimension):
        byte_val = text_hash[i % len(text_hash)]
        # Normalize to range [-1, 1]
        embedding.append((byte_val / 255.0) * 2 - 1)
    
    return embedding


def validate_record(record: Dict[str, Any], index: int) -> List[str]:
    """
    Validate a single record against the requirements.
    Returns a list of validation errors (empty if valid).
    """
    errors = []
    
    # Check required 'id' field
    if 'id' not in record:
        errors.append(f"Record {index}: Missing required 'id' field")
    
    # Check at least one embedding field exists
    has_embedding = 'embedding' in record
    has_sparse_embedding = 'sparse_embedding' in record
    
    if not has_embedding and not has_sparse_embedding:
        errors.append(f"Record {index} (id: {record.get('id', 'unknown')}): Must contain at least one of 'embedding' or 'sparse_embedding'")
    
    # Validate dense embedding
    if has_embedding:
        embedding = record['embedding']
        if not isinstance(embedding, list):
            errors.append(f"Record {index} (id: {record.get('id')}): 'embedding' must be a list")
        elif not all(isinstance(x, (int, float)) for x in embedding):
            errors.append(f"Record {index} (id: {record.get('id')}): 'embedding' must contain only numbers")
    
    # Validate sparse embedding
    if has_sparse_embedding:
        sparse = record['sparse_embedding']
        if not isinstance(sparse, dict):
            errors.append(f"Record {index} (id: {record.get('id')}): 'sparse_embedding' must be an object")
        else:
            if 'values' not in sparse or 'dimensions' not in sparse:
                errors.append(f"Record {index} (id: {record.get('id')}): 'sparse_embedding' must have 'values' and 'dimensions' fields")
            else:
                values = sparse['values']
                dimensions = sparse['dimensions']
                
                if not isinstance(values, list) or not isinstance(dimensions, list):
                    errors.append(f"Record {index} (id: {record.get('id')}): 'sparse_embedding.values' and 'dimensions' must be lists")
                elif len(values) != len(dimensions):
                    errors.append(f"Record {index} (id: {record.get('id')}): 'sparse_embedding.values' and 'dimensions' must have same length")
                elif not all(isinstance(x, (int, float)) for x in values):
                    errors.append(f"Record {index} (id: {record.get('id')}): 'sparse_embedding.values' must contain only numbers")
                elif not all(isinstance(x, int) for x in dimensions):
                    errors.append(f"Record {index} (id: {record.get('id')}): 'sparse_embedding.dimensions' must contain only integers")
    
    # Validate restricts
    if 'restricts' in record:
        restricts = record['restricts']
        if not isinstance(restricts, list):
            errors.append(f"Record {index} (id: {record.get('id')}): 'restricts' must be a list")
        else:
            for i, restrict in enumerate(restricts):
                if 'namespace' not in restrict:
                    errors.append(f"Record {index} (id: {record.get('id')}): restricts[{i}] missing 'namespace'")
                if 'allow' in restrict and not isinstance(restrict['allow'], list):
                    errors.append(f"Record {index} (id: {record.get('id')}): restricts[{i}].allow must be a list")
                if 'deny' in restrict and not isinstance(restrict['deny'], list):
                    errors.append(f"Record {index} (id: {record.get('id')}): restricts[{i}].deny must be a list")
    
    # Validate numeric_restricts
    if 'numeric_restricts' in record:
        numeric_restricts = record['numeric_restricts']
        if not isinstance(numeric_restricts, list):
            errors.append(f"Record {index} (id: {record.get('id')}): 'numeric_restricts' must be a list")
        else:
            for i, restrict in enumerate(numeric_restricts):
                if 'namespace' not in restrict:
                    errors.append(f"Record {index} (id: {record.get('id')}): numeric_restricts[{i}] missing 'namespace'")
                
                has_value = any(k in restrict for k in ['value_int', 'value_float', 'value_double'])
                if not has_value:
                    errors.append(f"Record {index} (id: {record.get('id')}): numeric_restricts[{i}] must have one of value_int, value_float, or value_double")
                
                if 'op' in restrict:
                    errors.append(f"Record {index} (id: {record.get('id')}): numeric_restricts[{i}] must not have 'op' field")
    
    # Validate crowding_tag
    if 'crowding_tag' in record:
        if not isinstance(record['crowding_tag'], str):
            errors.append(f"Record {index} (id: {record.get('id')}): 'crowding_tag' must be a string")
    
    return errors


def convert_to_vector_format(input_data, filename=None, id_field='id', 
                             embedding_field=None, text_field='text',
                             sparse_embedding_field=None, restricts_field=None,
                             numeric_restricts_field=None, crowding_tag_field=None,
                             generate_embeddings=False, embedding_dimension=768):
    """
    Convert input JSON data to vector database format.
    
    Parameters:
    - input_data: dict or list of dicts to convert
    - filename: filename for ID generation if needed
    - id_field: field name to use as 'id' (default: 'id')
    - embedding_field: field name containing dense embedding array
    - text_field: field name containing text to generate embeddings from
    - sparse_embedding_field: field name containing sparse embedding data
    - restricts_field: field name containing token restricts
    - numeric_restricts_field: field name containing numeric restricts
    - crowding_tag_field: field name containing crowding tag
    - generate_embeddings: if True, generate embeddings from text
    - embedding_dimension: dimension of generated embeddings
    """
    
    def process_record(record):
        """Process a single record"""
        output = {}
        
        # Handle ID field - generate if not present
        if id_field in record:
            output['id'] = str(record[id_field])
        else:
            output['id'] = generate_id_from_data(record, filename)
        
        # Dense embedding
        if embedding_field and embedding_field in record:
            embedding = record[embedding_field]
            if isinstance(embedding, list):
                output['embedding'] = [float(x) for x in embedding]
            else:
                raise ValueError(f"Field '{embedding_field}' must be a list")
        elif generate_embeddings and text_field in record:
            # Generate embedding from text
            text = record[text_field]
            if isinstance(text, str):
                output['embedding'] = generate_dummy_embedding(text, embedding_dimension)
            else:
                raise ValueError(f"Field '{text_field}' must be a string to generate embeddings")
        
        # Sparse embedding
        if sparse_embedding_field and sparse_embedding_field in record:
            sparse_data = record[sparse_embedding_field]
            if isinstance(sparse_data, dict) and 'values' in sparse_data and 'dimensions' in sparse_data:
                output['sparse_embedding'] = {
                    'values': [float(x) for x in sparse_data['values']],
                    'dimensions': [int(x) for x in sparse_data['dimensions']]
                }
            else:
                raise ValueError(f"Field '{sparse_embedding_field}' must have 'values' and 'dimensions'")
        
        # Check if at least one embedding exists
        if 'embedding' not in output and 'sparse_embedding' not in output:
            raise ValueError("Each record must contain at least one of 'embedding' or 'sparse_embedding'. Set generate_embeddings=True to create them from text.")
        
        # Token restricts
        if restricts_field and restricts_field in record:
            restricts = record[restricts_field]
            if isinstance(restricts, list):
                output['restricts'] = []
                for restrict in restricts:
                    processed_restrict = {'namespace': restrict['namespace']}
                    if 'allow' in restrict:
                        processed_restrict['allow'] = restrict['allow']
                    if 'deny' in restrict:
                        processed_restrict['deny'] = restrict['deny']
                    output['restricts'].append(processed_restrict)
        
        # Numeric restricts
        if numeric_restricts_field and numeric_restricts_field in record:
            numeric_restricts = record[numeric_restricts_field]
            if isinstance(numeric_restricts, list):
                output['numeric_restricts'] = []
                for restrict in numeric_restricts:
                    processed_restrict = {'namespace': restrict['namespace']}
                    if 'value_int' in restrict:
                        processed_restrict['value_int'] = int(restrict['value_int'])
                    elif 'value_float' in restrict:
                        processed_restrict['value_float'] = float(restrict['value_float'])
                    elif 'value_double' in restrict:
                        processed_restrict['value_double'] = float(restrict['value_double'])
                    output['numeric_restricts'].append(processed_restrict)
        
        # Crowding tag
        if crowding_tag_field and crowding_tag_field in record:
            output['crowding_tag'] = str(record[crowding_tag_field])
        
        return output
    
    # Handle both single dict and list of dicts
    if isinstance(input_data, dict):
        return [process_record(input_data)]
    elif isinstance(input_data, list):
        return [process_record(record) for record in input_data]
    else:
        raise ValueError("Input must be a dict or list of dicts")


def convert_json_file(input_file, output_file, validate=True, **kwargs):
    """
    Convert a JSON file to vector database format.
    
    Parameters:
    - input_file: path to input JSON file
    - output_file: path to output JSON file
    - validate: whether to validate output (default: True)
    - **kwargs: field mapping parameters (id_field, embedding_field, etc.)
    """
    
    print(f"\n📄 Processing: {input_file}")
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            # Try to load as regular JSON
            data = json.load(f)
        except json.JSONDecodeError:
            # If that fails, try JSONL format (one JSON object per line)
            f.seek(0)
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    
    # Convert data
    try:
        converted_data = convert_to_vector_format(data, filename=input_file, **kwargs)
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        return False
    
    # Validate converted data
    if validate:
        all_errors = []
        for i, record in enumerate(converted_data):
            errors = validate_record(record, i)
            all_errors.extend(errors)
        
        if all_errors:
            print(f"❌ Validation failed with {len(all_errors)} error(s):")
            for error in all_errors[:10]:
                print(f"  - {error}")
            if len(all_errors) > 10:
                print(f"  ... and {len(all_errors) - 10} more errors")
            return False
        else:
            print(f"✅ Validation passed")
    
    # Write output file (JSONL format - one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in converted_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully converted {len(converted_data)} records")
    print(f"   Output written to: {output_file}")
    return True


def process_folder(input_folder, output_folder, validate=True, **kwargs):
    """
    Process all JSON files in a folder.
    
    Parameters:
    - input_folder: path to folder containing JSON files
    - output_folder: path to output folder
    - validate: whether to validate output (default: True)
    - **kwargs: field mapping parameters
    """
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_path.rglob('*.json')) + list(input_path.rglob('*.jsonl'))
    
    if not json_files:
        print(f"❌ No JSON files found in {input_folder}")
        return 0, []
    
    print(f"Found {len(json_files)} JSON file(s) to process")
    print("="*70)
    
    success_count = 0
    failed_files = []
    
    for json_file in json_files:
        relative_path = json_file.relative_to(input_path)
        output_file = output_path / f"converted_{relative_path.name}"
        
        try:
            success = convert_json_file(
                str(json_file),
                str(output_file),
                validate=validate,
                **kwargs
            )
            if success:
                success_count += 1
            else:
                failed_files.append(str(json_file))
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {str(e)}")
            failed_files.append(str(json_file))
    
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Total files: {len(json_files)}")
    print(f"   ✅ Successfully converted: {success_count}")
    print(f"   ❌ Failed: {len(failed_files)}")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for file in failed_files[:10]:
            print(f"   - {file}")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more")
    
    return success_count, failed_files


def analyze_data_structure(input_folder):
    """
    Analyze the structure of data to determine field mappings.
    """
    print("\n🔍 Analyzing data structure...")
    
    input_path = Path(input_folder)
    json_files = list(input_path.rglob('*.json'))[:5]  # Sample first 5 files
    
    if not json_files:
        print("No JSON files found to analyze")
        return {}
    
    field_counts = {}
    sample_record = None
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                records = [data] if isinstance(data, dict) else data
                
                for record in records:
                    if sample_record is None:
                        sample_record = record
                    
                    for field in record.keys():
                        field_counts[field] = field_counts.get(field, 0) + 1
                        
                        if isinstance(record[field], dict):
                            for subfield in record[field].keys():
                                field_counts[f"{field}.{subfield}"] = field_counts.get(f"{field}.{subfield}", 0) + 1
        except Exception as e:
            continue
    
    print("📊 Field analysis results:")
    for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {field}: {count} occurrences")
    
    if sample_record:
        print("\n📝 Sample record structure:")
        print(json.dumps(sample_record, indent=2, ensure_ascii=False)[:500] + "...")
    
    # Suggest mappings
    suggestions = {
        'id_field': None,
        'text_field': None,
        'embedding_field': None
    }
    
    # Look for ID fields
    id_patterns = ['id', '_id', 'doc_id', 'record_id', 'uri']
    for pattern in id_patterns:
        for field in field_counts.keys():
            if pattern in field.lower():
                suggestions['id_field'] = field
                break
        if suggestions['id_field']:
            break
    
    # Look for text fields
    text_patterns = ['text', 'content', 'body', 'description']
    for pattern in text_patterns:
        for field in field_counts.keys():
            if pattern in field.lower():
                suggestions['text_field'] = field
                break
        if suggestions['text_field']:
            break
    
    # Look for embedding fields
    embedding_patterns = ['embedding', 'vector', 'embed', 'dense']
    for pattern in embedding_patterns:
        for field in field_counts.keys():
            if pattern in field.lower():
                suggestions['embedding_field'] = field
                break
        if suggestions['embedding_field']:
            break
    
    print("\n💡 Recommended settings:")
    print(f"  id_field: '{suggestions['id_field']}' (will generate from filename/uri if not found)")
    print(f"  text_field: '{suggestions['text_field']}' (for embedding generation)")
    print(f"  embedding_field: '{suggestions['embedding_field']}' (will generate from text if not found)")
    print(f"  generate_embeddings: True (recommended since no embedding field found)")
    
    return suggestions


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("📁 Nepali OCR Data to Vector Format Converter")
    print("="*70)
    
    INPUT_FOLDER = "nepali_ocr_data"
    OUTPUT_FOLDER = "corpus_file"
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        sys.exit(1)
    
    # Analyze structure
    suggestions = analyze_data_structure(INPUT_FOLDER)
    
    print("\n" + "="*70)
    print("🚀 Starting conversion with embedding generation...")
    print("="*70)
    
    # Process with settings
    success_count, failed_files = process_folder(
        INPUT_FOLDER,
        OUTPUT_FOLDER,
        validate=True,
        id_field=suggestions.get('id_field') or 'uri',  # Fall back to uri
        text_field=suggestions.get('text_field') or 'text',  # Default to 'text'
        generate_embeddings=True,  # Generate embeddings from text
        embedding_dimension=768  # Standard BERT dimension
    )
    
    # Create consolidated file
    if success_count > 0:
        print("\n" + "="*70)
        print("📦 Creating consolidated output file...")
        
        output_path = Path(OUTPUT_FOLDER)
        output_files = list(output_path.glob('converted_*.json'))
        
        consolidated_data = []
        for output_file in output_files:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        consolidated_data.append(json.loads(line))
        
        consolidated_file = output_path / "consolidated_vectors.jsonl"
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            for record in consolidated_data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"✅ Created consolidated file with {len(consolidated_data)} records")
        print(f"   Location: {consolidated_file}")
    
    print("\n" + "="*70)
    print("🎉 Processing complete!")
    print("="*70)
    print("\n⚠️  IMPORTANT NOTE:")
    print("The embeddings generated are DUMMY/PLACEHOLDER embeddings.")
    print("For production use, replace with real embeddings using:")
    print("  - sentence-transformers (Hugging Face)")
    print("  - OpenAI embeddings")
    print("  - Google Vertex AI embeddings")
    print("  - Other embedding models")
    print("="*70)