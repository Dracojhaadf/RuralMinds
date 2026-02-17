"""
Simple test script for streaming functionality
"""
import json

def test_streaming_logic():
    """Test the streaming logic without full backend dependencies"""
    
    # Simulate streaming response from Ollama
    mock_ollama_response = [
        '{"response": "The"}',
        '{"response": " answer"}',
        '{"response": " is"}',
        '{"response": " 42"}',
        '{"response": "."}',
    ]
    
    print("Testing streaming token parsing...")
    full_response = ""
    
    for line in mock_ollama_response:
        try:
            chunk = json.loads(line)
            if "response" in chunk:
                token = chunk["response"]
                full_response += token
                print(f"Token: '{token}' | Accumulated: '{full_response}'")
        except json.JSONDecodeError:
            continue
    
    print(f"\n[OK] Final response: '{full_response}'")
    assert full_response == "The answer is 42.", f"Expected 'The answer is 42.', got '{full_response}'"
    print("[OK] Streaming logic test passed!")
    
    # Test generator pattern
    print("\nTesting generator pattern...")
    
    def mock_streaming_generator():
        """Mock streaming generator"""
        for line in mock_ollama_response:
            try:
                chunk = json.loads(line)
                if "response" in chunk:
                    yield {'type': 'token', 'content': chunk["response"]}
            except json.JSONDecodeError:
                continue
        yield {'type': 'chunks', 'content': []}
    
    accumulated = ""
    for chunk in mock_streaming_generator():
        if chunk['type'] == 'token':
            accumulated += chunk['content']
            print(f"Yielded token: '{chunk['content']}'")
    
    print(f"\n[OK] Generator accumulated: '{accumulated}'")
    assert accumulated == "The answer is 42.", f"Expected 'The answer is 42.', got '{accumulated}'"
    print("[OK] Generator pattern test passed!")
    
    print("\n" + "="*50)
    print("[SUCCESS] All streaming tests passed!")
    print("="*50)

if __name__ == "__main__":
    test_streaming_logic()
