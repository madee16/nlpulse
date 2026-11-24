"""API key service handling issuance, cryptographic verification, and revocation."""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import AuthenticationError, ResourceNotFoundError
from src.core.security import generate_api_key, hash_api_key
from src.models.api_key import APIKey


class APIKeyService:
    """Service encapsulating API key issuance, authentication, and tenant lifecycle."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_key(
        self,
        developer_id: uuid.UUID,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[APIKey, str]:
        """Generate and persist a new API key for a developer.

        Returns:
            Tuple[APIKey, str]: (persisted_api_key, raw_secret_key)
                - raw_secret_key: Returned once for display; NEVER stored in DB.
        """
        raw_key, key_prefix, key_hash = generate_api_key(prefix="nlp_live")

        expires_at = None
        if expires_in_days is not None:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        api_key = APIKey(
            developer_id=developer_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name.strip(),
            scopes=scopes or ["analyze:read"],
            expires_at=expires_at,
        )
        self.session.add(api_key)
        await self.session.commit()
        await self.session.refresh(api_key)
        return api_key, raw_key

    async def list_keys(self, developer_id: uuid.UUID) -> List[APIKey]:
        """List all API keys belonging to a developer."""
        stmt = (
            select(APIKey)
            .where(APIKey.developer_id == developer_id)
            .order_by(APIKey.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_key_for_developer(self, developer_id: uuid.UUID, key_id: uuid.UUID) -> Optional[APIKey]:
        """Fetch an API key ensuring strict tenant ownership."""
        stmt = (
            select(APIKey)
            .where(APIKey.id == key_id, APIKey.developer_id == developer_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def revoke_key(self, developer_id: uuid.UUID, key_id: uuid.UUID) -> APIKey:
        """Revoke an API key belonging to a developer.

        Raises:
            ResourceNotFoundError: If the key does not exist or belongs to another tenant.
        """
        key = await self.get_key_for_developer(developer_id, key_id)
        if not key:
            raise ResourceNotFoundError(f"API key '{key_id}' not found.")

        key.is_active = False
        key.revoked_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(key)
        return key

    async def authenticate_raw_key(self, raw_key: str) -> APIKey:
        """Verify raw key against stored hash and check expiration and revocation.

        Raises:
            AuthenticationError: On invalid, revoked, or expired keys.
        """
        if not raw_key or not isinstance(raw_key, str):
            raise AuthenticationError("API key header missing or empty.")

        key_hash = hash_api_key(raw_key)
        stmt = select(APIKey).where(APIKey.key_hash == key_hash)
        result = await self.session.execute(stmt)
        api_key = result.scalar_one_or_none()

        if not api_key:
            raise AuthenticationError("Invalid API key.")

        if not api_key.is_active or api_key.revoked_at is not None:
            raise AuthenticationError("API key has been revoked.")

        now = datetime.now(timezone.utc)
        if api_key.expires_at:
            expires_at = api_key.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at < now:
                raise AuthenticationError("API key has expired.")

        # Update last_used_at timestamp
        api_key.last_used_at = now
        await self.session.commit()
        return api_key
