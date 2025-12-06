# src/test.py
import json
import ollama

def test_ollama():
    client = ollama.Client(host="http://localhost:11434")
    resp = client.chat(model="llama3.2", messages=[{"role":"user","content":"hi"}])
    # print readable
    print(json.dumps(resp, indent=2, default=str))

if __name__ == "__main__":
    test_ollama()
