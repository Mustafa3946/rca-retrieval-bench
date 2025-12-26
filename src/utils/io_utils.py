"""
I/O utilities for reading JSON and JSONL files.
"""

import json
from pathlib import Path
from typing import List, Dict, Generator, Union


def read_json_or_jsonl(path: str, stream: bool = False) -> Union[List[Dict], Generator[Dict, None, None]]:
    """
    Reads either JSON array or JSONL format.
    
    Args:
        path: Path to file (.json or .jsonl)
        stream: If True and file is JSONL, return generator (memory efficient)
        
    Returns:
        List of dicts or generator of dicts
        
    Raises:
        ValueError: If file format is invalid
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Detect format by extension
    is_jsonl = path_obj.suffix.lower() == '.jsonl'
    
    if is_jsonl:
        if stream:
            return _read_jsonl_stream(path)
        else:
            return list(_read_jsonl_stream(path))
    else:
        # Standard JSON array
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array in {path}, got {type(data)}")
            return data


def _read_jsonl_stream(path: str) -> Generator[Dict, None, None]:
    """Stream JSONL file line by line."""
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    print(f"Warning: Line {line_num} is not a dict, skipping")
                    continue
                yield obj
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue


def load_embedded_ids(output_path: str) -> set:
    """
    Load already embedded template IDs from existing embeddings file.
    
    Args:
        output_path: Path to template_embeddings.jsonl
        
    Returns:
        Set of template_id strings
    """
    path = Path(output_path)
    if not path.exists():
        return set()
    
    embedded_ids = set()
    try:
        for record in read_json_or_jsonl(str(path), stream=True):
            if 'template_id' in record:
                embedded_ids.add(record['template_id'])
    except Exception as e:
        print(f"Warning: Could not load existing embeddings: {e}")
        return set()
    
    return embedded_ids
