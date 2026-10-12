import os
from groq import Groq


models = client.models.list()
for m in models.data:
    print(m.id)
