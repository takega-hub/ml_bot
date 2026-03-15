import os

import pytest
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


@pytest.mark.skipif(
    not SUPABASE_URL or not SUPABASE_KEY,
    reason="Supabase credentials not found in environment",
)
def test_supabase():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    data = {"role": "user", "content": "Test message from script"}
    response = supabase.table("chat_messages").insert(data).execute()
    assert response is not None

    response = supabase.table("chat_messages").select("*").limit(5).execute()
    assert response is not None
    assert isinstance(response.data, list)


if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("Supabase credentials not found in environment")
    test_supabase()
