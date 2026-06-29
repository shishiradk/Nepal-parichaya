#!/usr/bin/env python3
"""
List available generative AI models in your GCP project
"""

import vertexai
from vertexai.generative_models import GenerativeModel
import json

PROJECT_ID = "gen-lang-client-0000379298"
REGION = "us-east1"

# Initialize
vertexai.init(project=PROJECT_ID, location=REGION)

print("=" * 70)
print("AVAILABLE GENERATIVE AI MODELS")
print("=" * 70)
print(f"Project: {PROJECT_ID}")
print(f"Region: {REGION}\n")

# List of models to test
models_to_test = [
    # Gemini models
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash",
    "gemini-1.5-flash-002",
    "gemini-1.0-pro",
    
    # PaLM models (older)
    "text-bison",
    "text-bison-32k",
    "text-unicorn",
    
    # CodeStorm/other
    "claude-opus",
    "claude-sonnet",
]

available_models = []
unavailable_models = []

print("Testing models...\n")

for model_name in models_to_test:
    try:
        model = GenerativeModel(model_name)
        # Try a simple test
        response = model.generate_content("hello")
        available_models.append(model_name)
        print(f"✅ {model_name}")
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            unavailable_models.append(model_name)
            print(f"❌ {model_name} - Not found in this project")
        elif "permission" in error_msg.lower() or "not have access" in error_msg.lower():
            unavailable_models.append(model_name)
            print(f"⛔ {model_name} - No access/permission")
        else:
            print(f"⚠️  {model_name} - {error_msg[:50]}...")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if available_models:
    print(f"\n✅ {len(available_models)} AVAILABLE MODEL(S):")
    for model in available_models:
        print(f"   - {model}")
else:
    print("\n❌ No models are available in your project")
    print("\nPossible solutions:")
    print("  1. Enable the Vertex AI API:")
    print("     gcloud services enable aiplatform.googleapis.com")
    print("  2. Check your GCP quotas and permissions")
    print("  3. Contact your GCP administrator")

if unavailable_models:
    print(f"\n❌ {len(unavailable_models)} unavailable models (tested but not accessible)")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
if available_models:
    print(f"\nUpdate rag_query.py to use: {available_models[0]}")
    print(f"\nRun this command to update the file:")
    print(f"  sed -i 's/gemini-.*/{available_models[0]}/g' rag_query.py")
else:
    print("\nYou need to:")
    print("  1. Enable Vertex AI APIs in your GCP project")
    print("  2. Request access to specific models if needed")
    print("  3. Check your billing and quota settings")
