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
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Summarize this email"},
            {"role": "user", "content": content}
        ],
        max_tokens=100,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

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
