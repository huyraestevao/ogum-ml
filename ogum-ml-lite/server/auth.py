"""Authentication helpers powered by JSON Web Tokens."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from . import storage
from .models import Role, Token, User, UserPublic

SECRET_KEY = os.environ.get("OGUM_SERVER_SECRET", "super-secret-ogum-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class AuthError(Exception):
    """Raised when authentication fails."""


def hash_password(password: str) -> str:
    """Return a deterministic hash for the password.

    A lightweight SHA256 hash keeps dependencies small while still
    providing a hashed representation compatible with JSON storage.
    """

    digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return digest


def verify_password(plain_password: str, password_hash: str) -> bool:
    return hash_password(plain_password) == password_hash


def _load_users() -> dict[str, User]:
    users: dict[str, User] = {}
    for entry in storage.iter_users():
        try:
            user = User(**entry)
        except TypeError:
            continue
        users[user.username] = user
    return users


def authenticate_user(username: str, password: str) -> Optional[User]:
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def create_access_token(
    *, username: str, role: Role, expires_delta: timedelta | None = None
) -> str:
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_jwt(user: User) -> Token:
    token = create_access_token(username=user.username, role=user.role)
    return Token(access_token=token)


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> UserPublic:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:  # pragma: no cover - safety branch
        raise credentials_exception from exc
    username: str | None = payload.get("sub")
    if username is None:
        raise credentials_exception
    role = payload.get("role", "user")
    return UserPublic(username=username, role=role)


def require_admin(user: UserPublic) -> None:
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )


def upsert_user(username: str, password: str, role: Role = "user") -> User:
    users = _load_users()
    user = User(username=username, password_hash=hash_password(password), role=role)
    users[username] = user
    storage.save_users([u.model_dump() for u in users.values()])
    return user
