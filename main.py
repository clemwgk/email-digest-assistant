import os
import base64
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import openai

# Kill switch
if os.getenv("DISABLE_DIGEST", "").strip() == "1":
    print("Digest disabled via DISABLE_DIGEST=1")
    raise SystemExit(0)

# Load env vars if running locally
load_dotenv()

# Debug logging toggle
DEBUG = os.getenv("DEBUG", "").strip() == "1"
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def authenticate():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def fetch_unread_messages(service, user_id="me", max_results=10):
    debug_print(f"Fetching up to {max_results} unread messages...")
    results = service.users().messages().list(userId=user_id, q="is:unread", maxResults=max_results).execute()
    return results.get("messages", [])

def get_message_details(service, msg_id, user_id="me"):
    msg = service.users().messages().get(userId=user_id, id=msg_id).execute()
    headers = msg["payload"].get("headers", [])
    subject, sender = "", ""
    for header in headers:
        if header["name"] == "Subject":
            subject = header["value"]
        elif header["name"] == "From":
            sender = header["value"]
    snippet = msg.get("snippet", "")
    return sender, subject, snippet

def summarize_email(content):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Summarize this email"},
            {"role": "user", "content": content}
        ],
        max_tokens=100
    )
    return resp.choices[0].message["content"].strip()

def main():
    service = authenticate()
    messages = fetch_unread_messages(service)
    if not messages:
        print("No new emails.")
        return
    digest = []
    for m in messages:
        sender, subject, snippet = get_message_details(service, m["id"])
        content = f"From: {sender}\nSubject: {subject}\n\nSnippet: {snippet}"
        summary = summarize_email(content)
        digest.append(f"From: {sender}\nSubject: {subject}\nSummary: {summary}\n---")
    print("\n".join(digest))

if __name__ == "__main__":
    main()
