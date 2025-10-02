from __future__ import annotations

from datetime import timedelta

from jose import jwt
from server import auth


def test_password_hash_roundtrip() -> None:
    password = "secret123"
    hashed = auth.hash_password(password)
    assert hashed == auth.hash_password(password)
    assert auth.verify_password(password, hashed)
    assert not auth.verify_password("wrong", hashed)


def test_create_access_token(monkeypatch) -> None:
    monkeypatch.setenv("OGUM_SERVER_SECRET", "unit-test-secret")
    auth.SECRET_KEY = "unit-test-secret"
    token = auth.create_access_token(
        username="alice", role="admin", expires_delta=timedelta(minutes=5)
    )
    payload = jwt.decode(token, "unit-test-secret", algorithms=[auth.ALGORITHM])
    assert payload["sub"] == "alice"
    assert payload["role"] == "admin"


def test_upsert_and_authenticate(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OGUM_STORAGE_PATH", str(tmp_path))
    user = auth.upsert_user("bob", "password", role="user")
    assert user.username == "bob"
    assert auth.authenticate_user("bob", "password")
    assert auth.authenticate_user("bob", "wrong") is None
    assert auth.authenticate_user("unknown", "password") is None
