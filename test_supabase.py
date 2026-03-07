import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print(f"URL: {SUPABASE_URL}")
# Print first/last chars of key to verify it's loaded but not leak it entirely
if SUPABASE_KEY:
    print(f"KEY: {SUPABASE_KEY[:5]}...{SUPABASE_KEY[-5:]}")
else:
    print("KEY: None")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Supabase credentials not found in .env")
    exit(1)

async def test_supabase():
    print("Initializing Supabase client...")
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Client initialized.")
        
        # Test Insert
        print("Attempting to insert test message...")
        data = {"role": "user", "content": "Test message from script"}
        
        # Note: supabase-py is synchronous by default, but we can run it in executor or just call it directly for this test script
        response = supabase.table("chat_messages").insert(data).execute()
        print(f"Insert response: {response}")
        
        # Test Select
        print("Attempting to fetch messages...")
        response = supabase.table("chat_messages").select("*").limit(5).execute()
        print(f"Select response data: {response.data}")
        
    except Exception as e:
        print(f"Error during Supabase operations: {e}")

if __name__ == "__main__":
    # Just run sync for the test script to keep it simple, unless supabase lib forces async
    # The standard supabase-py client is synchronous.
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Client initialized.")
        
        print("Attempting to insert test message...")
        data = {"role": "user", "content": "Test message from script"}
        response = supabase.table("chat_messages").insert(data).execute()
        print(f"Insert success! Response: {response}")
        
        print("Attempting to fetch messages...")
        response = supabase.table("chat_messages").select("*").limit(5).execute()
        print(f"Fetch success! Data: {response.data}")
        
    except Exception as e:
        print(f"FAILED: {e}")
