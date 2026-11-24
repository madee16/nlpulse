"""Developer service handling tenant registration and quota management."""

from typing import Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import ResourceConflictError, ResourceNotFoundError
from src.models.developer import Developer

# Default quotas mapped per tier
TIER_QUOTAS = {
    "free": 1000,
    "pro": 10000,
    "enterprise": 100000,
}


class DeveloperService:
    """Service encapsulating Developer tenant operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_developer(
        self,
        email: str,
        name: str,
        tier: str = "free",
    ) -> Developer:
        """Register a new developer tenant.

        Raises:
            ResourceConflictError: If a developer with the same email already exists.
        """
        existing = await self.get_by_email(email)
        if existing:
            raise ResourceConflictError(f"Developer with email '{email}' is already registered.")

        quota = TIER_QUOTAS.get(tier.lower(), 1000)
        developer = Developer(
            email=email.strip().lower(),
            name=name.strip(),
            tier=tier.lower(),
            monthly_request_quota=quota,
        )
        self.session.add(developer)
        await self.session.commit()
        await self.session.refresh(developer)
        return developer

    async def get_by_id(self, developer_id: uuid.UUID) -> Optional[Developer]:
        """Fetch a developer by UUID primary key."""
        stmt = select(Developer).where(Developer.id == developer_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[Developer]:
        """Fetch a developer by email address."""
        stmt = select(Developer).where(Developer.email == email.strip().lower())
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_required_developer(self, developer_id: uuid.UUID) -> Developer:
        """Fetch developer or raise ResourceNotFoundError."""
        dev = await self.get_by_id(developer_id)
        if not dev:
            raise ResourceNotFoundError(f"Developer '{developer_id}' not found.")
        return dev
