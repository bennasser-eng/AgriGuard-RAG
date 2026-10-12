import os
from groq import Groq

client = Groq(api_key="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")
models = client.models.list()
for m in models.data:
    print(m.id)
