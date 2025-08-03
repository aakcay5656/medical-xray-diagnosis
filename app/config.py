import os
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT")

if ENVIRONMENT == "PRODUCTION":
    BASE_URL = " "
else:
    BASE_URL = "http://127.0.0.1:8000"