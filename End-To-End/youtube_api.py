import os
import googleapiclient.discovery
import google_auth_oauthlib.flow
import googleapiclient.errors

def init_youtube(oauth=False):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    if oauth:
        scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]
        client_secrets_file = "client_secret.json"
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
            client_secrets_file, scopes)
        credentials = flow.run_console()
        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, credentials=credentials)
    else:
        DEVELOPER_KEY = "AIzaSyBEjf7WknjMp6wmzhnpeJIsWDhGk3Uq-MM"
        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey = DEVELOPER_KEY)

    return youtube