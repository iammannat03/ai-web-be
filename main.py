from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch


# Load environment variables
load_dotenv()

# LangFlow code
import argparse
import json
from argparse import RawTextHelpFormatter
import requests
from typing import Optional
import warnings
try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None

app = FastAPI()

# Load configuration from environment variables
BASE_API_URL = os.getenv("BASE_API_URL")
LANGFLOW_ID = os.getenv("LANGFLOW_ID")
FLOW_ID = os.getenv("FLOW_ID")
APPLICATION_TOKEN = os.getenv("APPLICATION_TOKEN")
ENDPOINT = os.getenv("ENDPOINT", "")  # Default to empty string if not set

# Validate required environment variables
if not all([BASE_API_URL, LANGFLOW_ID, FLOW_ID, APPLICATION_TOKEN]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# You can tweak the flow by adding a tweaks dictionary
# e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
TWEAKS = {
  "ChatInput-8qj7E": {},
  "ParseData-z37RQ": {},
  "Prompt-BcKDH": {},
  "SplitText-C9acM": {},
  "ChatOutput-awtAS": {},
  "File-ZLs80": {},
  "AstraDB-BC6oU": {},
  "AstraDB-4fi43": {},
  "NVIDIAEmbeddingsComponent-mqPbW": {},
  "GoogleGenerativeAIModel-fQ2v6": {}
}

def run_flow(message: str,
  endpoint: str,
  output_type: str = "chat",
  input_type: str = "chat",
  tweaks: Optional[dict] = None,
  application_token: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if application_token:
        headers = {"Authorization": "Bearer " + application_token, "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

def run_rag_chatbot():
    parser = argparse.ArgumentParser(description="""Run a flow with a given message and optional tweaks.
Run it like: python <your file>.py "your message here" --endpoint "your_endpoint" --tweaks '{"key": "value"}'""",
        formatter_class=RawTextHelpFormatter)
    parser.add_argument("message", type=str, help="The message to send to the flow")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT or FLOW_ID, help="The ID or the endpoint name of the flow")
    parser.add_argument("--tweaks", type=str, help="JSON string representing the tweaks to customize the flow", default=json.dumps(TWEAKS))
    parser.add_argument("--application_token", type=str, default=APPLICATION_TOKEN, help="Application Token for authentication")
    parser.add_argument("--output_type", type=str, default="chat", help="The output type")
    parser.add_argument("--input_type", type=str, default="chat", help="The input type")
    parser.add_argument("--upload_file", type=str, help="Path to the file to upload", default=None)
    parser.add_argument("--components", type=str, help="Components to upload the file to", default=None)

    args = parser.parse_args()
    try:
      tweaks = json.loads(args.tweaks)
    except json.JSONDecodeError:
      raise ValueError("Invalid tweaks JSON string")

    if args.upload_file:
        if not upload_file:
            raise ImportError("Langflow is not installed. Please install it to use the upload_file function.")
        elif not args.components:
            raise ValueError("You need to provide the components to upload the file to.")
        tweaks = upload_file(file_path=args.upload_file, host=BASE_API_URL, flow_id=ENDPOINT, components=args.components, tweaks=tweaks)

    response = run_flow(
        message=args.message,
        endpoint=args.endpoint,
        output_type=args.output_type,
        input_type=args.input_type,
        tweaks=tweaks,
        application_token=args.application_token
    )

    print(json.dumps(response, indent=2))

@app.get("/rag-chatbot")
def read_root():
    return {"message": "Hello this is me mannat"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None =None):
    return {"item_id": item_id, "q": q}

# Create a model for the chat request
class ChatRequest(BaseModel):
    message: str
    output_type: str = "chat"
    input_type: str = "chat"
    tweaks: dict | None = None

@app.post("/rag-chatbot/chat")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        print(f"Using FLOW_ID: {FLOW_ID}")
        print(f"Using ENDPOINT: {ENDPOINT}")
        
        # Use FLOW_ID directly since ENDPOINT is empty
        endpoint = FLOW_ID
        print(f"Using endpoint: {endpoint}")
        
        response = run_flow(
            message=chat_request.message,
            endpoint=endpoint,  # Use FLOW_ID directly
            output_type=chat_request.output_type,
            input_type=chat_request.input_type,
            tweaks=chat_request.tweaks or TWEAKS,
            application_token=APPLICATION_TOKEN
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add an endpoint for file upload if needed
class FileUploadRequest(BaseModel):
    message: str
    file_path: str
    components: str
    tweaks: dict | None = None

@app.post("/rag-chatbot/upload-and-chat")
async def upload_and_chat_endpoint(request: FileUploadRequest):
    try:
        if not upload_file:
            raise HTTPException(
                status_code=400, 
                detail="Langflow is not installed. Please install it to use file upload."
            )
        
        # Handle file upload with tweaks
        updated_tweaks = upload_file(
            file_path=request.file_path,
            host=BASE_API_URL,
            flow_id=ENDPOINT,
            components=request.components,
            tweaks=request.tweaks or TWEAKS
        )

        # Run the flow with uploaded file
        response = run_flow(
            message=request.message,
            endpoint=ENDPOINT or FLOW_ID,
            tweaks=updated_tweaks,
            application_token=APPLICATION_TOKEN
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Move this to the top of your file, after imports
print("Loading sentiment analysis model... This may take a few minutes...")
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
print("Model loaded successfully!")

# Function to perform sentiment analysis using Roberta
def analyze_sentiment(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_text).logits
        
    scores = output[0].detach().cpu().numpy()  # Move to CPU for numpy
    scores = softmax(scores)
    
    sentiment_labels = ['negative', 'neutral', 'positive']
    sentiment = sentiment_labels[scores.argmax()]
    sentiment_score = float(scores.max())  # Convert numpy.float32 to Python float
    
    return {
        "sentiment": sentiment,
        "score": sentiment_score  # This will now be a regular Python float
    }

# Create a Pydantic model for the request
class SentimentRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    try:
        sentiment_result = analyze_sentiment(request.text)
        return sentiment_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {e}")