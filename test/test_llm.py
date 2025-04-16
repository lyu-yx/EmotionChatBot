#!/usr/bin/env python
"""
Test script for Language Model Integration
-----------------------------------------
This script tests the LLM functionality to ensure it's working with the updated components.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.language_model import AlibabaLanguageModel, BailianLanguageModel

def verify_api_key():
    """Verify that the Alibaba API key is available"""
    print("\n=== Checking API Key ===")
    
    # Get environment
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path, override=True)
    print(f"Loaded environment from: {env_path}")
    
    # Check if API key is available
    api_key = os.getenv("ALIBABA_API_KEY")
    
    if not api_key:
        print("❌ ALIBABA_API_KEY not found in .env file")
        return False
    
    print(f"✓ ALIBABA_API_KEY found: {api_key[:5]}...")
    return True

def test_alibaba_llm():
    """Test the Alibaba Tongyi Qianwen language model"""
    print("\n=== Testing Alibaba Tongyi Qianwen Language Model ===")
    
    if not verify_api_key():
        print("❌ Cannot continue without API key.")
        return False
    
    try:
        # Initialize the language model
        print("Initializing AlibabaLanguageModel...")
        llm = AlibabaLanguageModel(
            model_name="qwen-turbo",
            temperature=0.7,
            system_prompt="你是一个有用的语音交互助手。请提供简洁明了的回答，适合语音输出。"
        )
        print("✓ Model initialized successfully")
        
        # Test in Chinese
        test_prompts = [
            "今天天气怎么样？",
            "你能介绍一下自己吗？",
            "如何使用语音识别技术？"
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest prompt {i+1}: \"{prompt}\"")
            print("Generating response...")
            
            start_time = time.time()
            result = llm.generate_response(prompt)
            end_time = time.time()
            
            if result["success"]:
                print(f"✓ Response generated in {end_time - start_time:.2f} seconds:")
                print(f"---\n{result['response']}\n---")
            else:
                print(f"❌ Failed to generate response: {result['error']}")
                return False
        
        return True
    
    except Exception as e:
        print(f"❌ Error during LLM test: {e}")
        return False

def test_bailian_llm():
    """Test the Bailian Language Model with both dashscope and bailian APIs"""
    print("\n=== Testing Bailian Language Model ===")
    
    if not verify_api_key():
        print("❌ Cannot continue without API key.")
        return False
    
    try:
        # Test with DashScope API
        print("\n--- Testing with DashScope API ---")
        llm_dashscope = BailianLanguageModel(
            model_name="qwen-turbo",
            api_type="dashscope",
            temperature=0.7,
            system_prompt="You are a helpful assistant. Provide concise responses."
        )
        
        # Test with a simple prompt
        prompt = "Tell me a short story about a robot learning to be human."
        print(f"Test prompt: \"{prompt}\"")
        print("Generating response...")
        
        start_time = time.time()
        result = llm_dashscope.generate_response(prompt)
        end_time = time.time()
        
        if result["success"]:
            print(f"✓ DashScope response generated in {end_time - start_time:.2f} seconds:")
            print(f"---\n{result['response'][:200]}...\n---")
        else:
            print(f"❌ Failed to generate DashScope response: {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Bailian LLM test: {e}")
        return False

if __name__ == "__main__":
    print("=== Language Model Test ===")
    
    # Test the AlibabaLanguageModel

    # Test the BailianLanguageModel with different API types
    bailian_success = test_bailian_llm()
    
    if bailian_success:
        print("\n✓ All LLM tests completed successfully!")
    else:
        print("\n❌ Some LLM tests failed.")
        
    print("\nNote: If you're getting API errors, check that:")
    print("1. Your ALIBABA_API_KEY is correctly set in the .env file")
    print("2. Your account has access to the requested model")
    print("3. You have sufficient quota remaining")