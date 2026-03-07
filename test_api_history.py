import asyncio
import os
import httpx
from dotenv import load_dotenv

# Load env to get API key
load_dotenv()
API_KEY = os.getenv("MOBILE_API_KEY", "")
BASE_URL = "http://127.0.0.1:8765"

async def test_history_endpoint():
    print(f"Testing API endpoint: {BASE_URL}/api/ai/chat/history")
    print(f"Using API Key: {API_KEY[:5]}...")
    
    headers = {"X-API-Key": API_KEY}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/api/ai/chat/history", headers=headers, timeout=10.0)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("Response JSON:")
                import json
                print(json.dumps(data, indent=2))
                
                history = data.get("history", [])
                print(f"History items count: {len(history)}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_history_endpoint())
