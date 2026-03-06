import requests
import json

url = "http://localhost:8765/api/pairs"
headers = {"X-API-Key": "d44f6467fdb55f9dd22eaadc12c611bc"}

try:
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
