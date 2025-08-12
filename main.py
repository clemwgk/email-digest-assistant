import os
import base64
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError


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

# --- OpenAI client (create once) ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-4.1-mini"
MODEL_CANDIDATES = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-3.5-turbo"]
MAX_RETRIES = 2
BASE_BACKOFF = 2.0  # seconds

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
    last_err = None
    for model in MODEL_CANDIDATES:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Summarize this email"},
                        {"role": "user", "content": content}
                    ],
                    max_tokens=100,
                    temperature=0.2
                )
                if DEBUG:
                    print(f"[summarize] model={model} attempt={attempt} OK")
                return resp.choices[0].message.content.strip()
            except (RateLimitError, APIError, APIConnectionError, APITimeoutError) as e:
                last_err = e
                if DEBUG:
                    print(f"[summarize] model={model} attempt={attempt} error: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(BASE_BACKOFF * attempt)  # 2s, then 4s
                else:
                    break  # try next model
            except Exception as e:
                last_err = e
                if DEBUG:
                    print(f"[summarize] model={model} non-retryable error: {e}")
                break
    raise RuntimeError(f"All model attempts failed. Last error: {last_err}")

def send_digest_email(subject, body_text):
    sender = os.getenv("EMAIL_USERNAME")
    password = os.getenv("EMAIL_APP_PASSWORD")
    if not sender or not password:
        raise RuntimeError("Missing EMAIL_USERNAME or EMAIL_APP_PASSWORD")

    recipient = sender  # send to self
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Plain text part
    msg.attach(MIMEText(body_text, "plain"))

    # Optional HTML version for nicer formatting
    html_body = f"<html><body><pre style='font-family:monospace;white-space:pre-wrap'>{body_text}</pre></body></html>"
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())

def main():
    service = authenticate()
    messages = fetch_unread_messages(service)
    digest = []

    if not messages:
        output = "No new emails today."
    else:
        for m in messages:
            sender, subject, snippet = get_message_details(service, m["id"])
            content = f"From: {sender}\nSubject: {subject}\n\nSnippet: {snippet}"
            summary = summarize_email(content)
            digest.append(f"From: {sender}\nSubject: {subject}\nSummary: {summary}\n---")
        output = "\n".join(digest)

    print(output)  # keep console output

    # Email it to yourself
    from datetime import datetime as _dt
    subject = f"AI Email Digest — {_dt.now().strftime('%Y-%m-%d')}"
    send_digest_email(subject, output)

if __name__ == "__main__":
    main()
