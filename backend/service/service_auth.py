# service/service_auth.py
import os
from datetime import datetime, timedelta, timezone
import jwt
import bcrypt
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret")
ALGORITHM = "HS256"


# ---- Hashing password ----
def hash_password(password: str) -> str:
    # Tronca a 72 byte e usa bcrypt direttamente
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode('utf-8')[:72]
    return bcrypt.checkpw(password_bytes, hashed_password.encode('utf-8'))



# ---- JWT ----
def genera_token(email: str, ruolo: str) -> str:
    expiration = datetime.now(timezone.utc) + timedelta(days=1)
    payload = {
        "sub": email,
        "exp": expiration,
        "role": ruolo
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

def verifica_token(token: str) -> dict:
    """Verifica e decodifica un JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise Exception("Token scaduto")
    except jwt.JWTError:
        raise Exception("Token non valido")
