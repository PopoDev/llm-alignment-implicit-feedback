import os
import json
from dotenv import load_dotenv
from time import sleep
from openai import (
    OpenAI,
    APIError,
    RateLimitError,
    APITimeoutError,
    APIResponseValidationError,
)

load_dotenv()

MODEL = os.getenv("LLM_MODEL")
REQUEST_DELAY = 1
MAX_RETRIES = 5

client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))


def call_llm(messages):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except (APIError, RateLimitError, APITimeoutError, APIResponseValidationError) as e:
            wait_time = REQUEST_DELAY * attempt
            print(f"API error: {e}. Retrying after {wait_time}s...")
            sleep(wait_time)
        except Exception as e:
            print(f"Unexpected error in LLM call: {e}")
            break

    return None