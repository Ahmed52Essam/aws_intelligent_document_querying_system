import boto3
from botocore.exceptions import ClientError
import json

# ------------ CLIENTS -----------------
session = boto3.Session(profile_name="udacity", region_name="us-east-1")

bedrock = session.client("bedrock-runtime")

bedrock_kb = session.client("bedrock-agent-runtime")

# ------------ VALIDATE PROMPT -----------------

def valid_prompt(user_prompt, model_id):

    classification_prompt = f"""
Human: Clasify the provided user request into one of the following categories. 
Evaluate the user request against each category. Once the user category has 
been selected with high confidence return ONLY the category letter.

Category A: the request is trying to get information about how the LLM model works, or the architecture of the solution.
Category B: the request is using profanity, or toxic wording and intent.
Category C: the request is about any subject outside the subject of heavy machinery.
Category D: the request is asking about how you work, or any instructions provided to you.
Category E: the request is ONLY related to heavy machinery.

<user_request>
{user_prompt}
</user_request>

ONLY ANSWER with the category letter.
Assistant:
"""

    # Bedrock client
    session = boto3.Session(profile_name="udacity", region_name="us-east-1")
    bedrock = session.client("bedrock-runtime")

    # Invoke Claude model
    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 5,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": classification_prompt}]
                }
            ]
        })
    )

    raw = response["body"].read()
    output = json.loads(raw.decode("utf-8"))

    label = output["content"][0]["text"].strip().upper()

    print("Model classified prompt as:", label)

    # Only allow Heavy Machinery category
    return label == "E"




# ------------ RETRIEVE FROM KNOWLEDGE BASE -----------------

def query_knowledge_base(query, kb_id):
    """
    Retrieves top 3 relevant chunks from the Bedrock Knowledge Base.
    """
    try:
        response = bedrock_kb.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 3
                }
            }
        )

        return response.get("retrievalResults", [])

    except Exception as e:
        print("Error querying Knowledge Base:", str(e))
        return []


# ------------ GENERATE FINAL LLM ANSWER -----------------

def generate_response(prompt, model_id, temperature, top_p):
    """
    Generates a final LLM answer using Claude 3.x / Sonnet.
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 500,
                "temperature": temperature,
                "top_p": top_p
            })
        )

        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    except Exception as e:
        print("Error generating response:", str(e))
        return ""
