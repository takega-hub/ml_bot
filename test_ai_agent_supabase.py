import os
import asyncio
import logging
from bot.ai_agent_service import AIAgentService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agent_supabase():
    print("Testing AIAgentService Supabase integration...")
    
    # Instantiate agent
    agent = AIAgentService()
    
    print(f"Supabase Client in Agent: {agent.supabase}")
    if agent.supabase:
        print("Success: Supabase client is set.")
    else:
        print("Failure: Supabase client is None.")
        # Print env vars to debug
        print(f"SUPABASE_URL env: {os.getenv('SUPABASE_URL')}")
        print(f"SUPABASE_KEY env: {os.getenv('SUPABASE_KEY')}")
        return

    # Try to save a message
    print("Attempting to save message via agent...")
    await agent._save_chat_message("user", "Test message from agent test script")
    print("Save method called.")
    
    # Try to fetch history
    print("Attempting to fetch history...")
    history = await agent._get_chat_history(limit=5)
    print(f"History fetched: {len(history)} items")
    print(history)

if __name__ == "__main__":
    asyncio.run(test_agent_supabase())
