#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Content Finder
Search through local JSON files to find what actually exists
"""

import json
from pathlib import Path
import sys
import re


def search_in_files(search_terms, input_folder="nepali_ocr_data"):
    """
    Search for terms in local JSON files
    """
    if not Path(input_folder).exists():
        print(f"Error: Folder '{input_folder}' not found")
        return
    
    json_files = list(Path(input_folder).rglob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return
    
    print("=" * 70)
    print(f"Searching in {len(json_files)} files")
    print(f"Search terms: {search_terms}")
    print("=" * 70)
    
    results = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = [data] if isinstance(data, dict) else data
            
            for idx, record in enumerate(records):
                text = record.get('text', '')
                
                # Check if any search term is in the text
                found_terms = []
                for term in search_terms:
                    if term.lower() in text.lower():
                        found_terms.append(term)
                
                if found_terms:
                    # Find context around the term
                    contexts = []
                    for term in found_terms:
                        pattern = re.compile(f'.{{0,100}}{re.escape(term)}.{{0,100}}', re.IGNORECASE)
                        matches = pattern.findall(text)
                        contexts.extend(matches[:2])  # First 2 matches
                    
                    results.append({
                        'file': json_file.name,
                        'record_idx': idx,
                        'found_terms': found_terms,
                        'contexts': contexts,
                        'full_text': text
                    })
        
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
    
    print(f"\nFound {len(results)} matching records\n")
    
    if results:
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] File: {result['file']}")
            print(f"    Record: {result['record_idx']}")
            print(f"    Found terms: {', '.join(result['found_terms'])}")
            print(f"    Context snippets:")
            for ctx in result['contexts'][:2]:
                print(f"      ... {ctx} ...")
            
            if i <= 5:
                show_full = input(f"\n    Show full text? (y/n): ").strip().lower()
                if show_full == 'y':
                    print(f"\n    Full text:\n    {result['full_text'][:1000]}...\n")
    else:
        print("No matches found!")
        print("\nTips:")
        print("  - Try different search terms")
        print("  - Check spelling")
        print("  - The content might not be in your documents")
    
    return results


def search_common_topics():
    """Search for common topics people ask about"""
    topics = {
        'राष्ट्रिय गान': ['राष्ट्रिय गान', 'राष्ट्रगान', 'national anthem', 'sayaun thunga'],
        'प्रान्त': ['प्रान्त', 'province', 'कोशी', 'मधेश', 'बाग्मती', 'गण्डकी', 'लुम्बिनी', 'कर्णाली', 'सुदूरपश्चिम'],
        'राजपुत र मुगल': ['राजपुत', 'मुगल', 'rajput', 'mughal', 'शैली'],
        'जनसंख्या': ['जनसंख्या', 'जनगणना', 'population', 'census'],
        'संविधान': ['संविधान', 'constitution', '२०७२', '2072'],
    }
    
    print("=" * 70)
    print("Searching for Common Topics")
    print("=" * 70)
    
    for topic, terms in topics.items():
        print(f"\n{topic}:")
        print("-" * 70)
        results = search_in_files(terms)
        
        if results:
            print(f"  Found in {len(results)} locations")
            print(f"  Files: {', '.join(set(r['file'] for r in results[:5]))}")
        else:
            print(f"  NOT FOUND in local files")


def find_all_unique_topics(input_folder="nepali_ocr_data", sample_size=100):
    """
    Extract unique topics/keywords from documents
    """
    print("=" * 70)
    print("Finding Unique Topics in Documents")
    print("=" * 70)
    
    json_files = list(Path(input_folder).rglob('*.json'))[:sample_size]
    
    all_text = ""
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = [data] if isinstance(data, dict) else data
            
            for record in records:
                text = record.get('text', '')
                all_text += " " + text
        
        except Exception as e:
            continue
    
    # Extract common Nepali words (simple approach)
    words = re.findall(r'[\u0900-\u097F]+', all_text)
    
    # Count frequency
    from collections import Counter
    word_freq = Counter(words)
    
    print(f"\nAnalyzed {len(json_files)} files")
    print(f"Total Nepali words: {len(words)}")
    print(f"Unique words: {len(word_freq)}")
    
    print("\nTop 30 most common words:")
    for word, count in word_freq.most_common(30):
        if len(word) > 2:  # Skip very short words
            print(f"  {word}: {count}")


def main():
    """Main menu"""
    print("\n" + "=" * 70)
    print("Document Content Finder")
    print("=" * 70)
    print("\nOptions:")
    print("  1. Search for specific terms")
    print("  2. Search common topics")
    print("  3. Find all unique topics")
    print("  4. Exit")
    
    choice = input("\nChoice: ").strip()
    
    if choice == "1":
        terms_input = input("\nEnter search terms (comma-separated): ").strip()
        terms = [t.strip() for t in terms_input.split(',')]
        search_in_files(terms)
    
    elif choice == "2":
        search_common_topics()
    
    elif choice == "3":
        find_all_unique_topics()
    
    elif choice == "4":
        return
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        terms = sys.argv[1:]
        search_in_files(terms)
    else:
        main()