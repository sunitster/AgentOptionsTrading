# ai/client.py

from openai import OpenAI

class LocalAIClient:
    def __init__(self, model: str = "llama3.2"):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.model = model

    def ask(self, prompt: str) -> str:
        """Send a prompt to local Llama3.2 model via Ollama server."""
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return res.choices[0].message.content
