from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import BigInteger, ForeignKey, Numeric, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base, CreateAtMixin, UpdateAtMixin


class Policy(Base, CreateAtMixin, UpdateAtMixin):
    """用户保单"""

    __tablename__ = "policies"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"))
    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id")
    )
    policy_number: Mapped[str] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(
        String(20), default="active", server_default="active"
    )
    effective_at: Mapped[datetime] = mapped_column(server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column()
    holder_name: Mapped[str | None] = mapped_column(String(80))
    insured_name: Mapped[str | None] = mapped_column(String(80))
    insured_phone: Mapped[str | None] = mapped_column(String(40))
    coverage_amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    premium_amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    policy_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    coverage_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )


class Claim(Base, UpdateAtMixin):
    """理赔案件（申请时间用submitted_at，表内无created_at列）"""

    __tablename__ = "claims"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"))
    policy_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("policies.id")
    )
    product_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("products.id")
    )
    claim_number: Mapped[str | None] = mapped_column(String(60))
    claim_type: Mapped[str] = mapped_column(String(50))
    amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(30), default="submitted", server_default="submitted"
    )
    submitted_at: Mapped[datetime] = mapped_column(server_default=func.now())
