#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Nepali RAG Query Script
"""

import vertexai
from vertexai.preview import rag
import json
import sys

# Configuration
PROJECT_ID = "gen-lang-client-0000379298"
REGION = "us-east1"

# Initialize
vertexai.init(project=PROJECT_ID, location=REGION)

# Load corpus info
with open("rag_deployment_info.json", "r") as f:
    info = json.load(f)

CORPUS_NAME = info["corpus_name"]

def query_nepali_rag(question):
    """Query the Nepali RAG system"""
    
    SYSTEM_PROMPT = """You are an expert on Nepal Parichaya. Follow these rules:
1. Answer in SAME LANGUAGE as question (Nepali→Nepali, English→English)
2. Fix OCR errors automatically in your response
3. For Nepal Parichaya: It's the official government reference book, 12th edition (2081 Asar)
4. For national anthem: "सयौं थुँगा फूलका हामी", adopted 2007, lyrics by Byakul Maila
5. Always provide specific information and corrections
6. Structure: Direct answer → Details → Context"""
    
    # Retrieve context
    results = rag.retrieval_query(
        corpus_name=CORPUS_NAME,
        query=question,
        max_results=5
    )
    
    # Format context
    context_text = ""
    for i, result in enumerate(results):
        context_text += f"Source {i+1}: {result.text}\n\n"
    
    # Generate response
    from vertexai.language_models import TextGenerationModel
    model = TextGenerationModel.from_pretrained("text-bison@002")
    
    prompt = f"""{SYSTEM_PROMPT}

CONTEXT FROM DOCUMENTS:
{context_text}

QUESTION: {question}

ANSWER (in same language as question, fix OCR errors):"""
    
    response = model.predict(
        prompt,
        temperature=0.2,
        max_output_tokens=1024
    )
    
    return response.text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")
    
    print(f"\nQuestion: {question}")
    print("\n" + "="*60)
    
    answer = query_nepali_rag(question)
    print("Answer:\n")
    print(answer)
    
    # Show sources
    print("\n" + "="*60)
    print("Retrieved from Nepal Parichaya documents")