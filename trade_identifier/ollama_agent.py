from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def ask_ollama(plan, rag_data):
    prompt = f"""
    You are an expert options strategist.

    Market conditions:
    IV Percentile: {plan['ivp']}
    ATR Regime: {plan['atr_regime']}
    NIFTY Trend: {plan['trend']}

    Proposed trade:
    {plan['legs']}

    Historical similar outcomes (from RAG):
    {rag_data}

    Should we:
    - Take Iron Condor
    - Switch to Iron Fly
    - Tighten wings
    - Widen wings
    - Skip?

    Output JSON:
    {{
      "action": "IC | IFLY | TIGHTEN | WIDEN | SKIP",
      "confidence": 0-1,
      "adjust": {{}}
    }}
    """

    resp = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message["content"]
