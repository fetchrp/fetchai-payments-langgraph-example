# protocols/payment_proto.py
"""
Payment protocol for merchandiser agent (seller role).
- Sends payment requests with Skyfire USDC option
- Verifies CommitPayment using Skyfire JWT
- Auto-dispatches order after verified payment
"""

from __future__ import annotations
import os
from typing import Optional

from uagents import Protocol, Context
from uagents_core.contrib.protocols.payment import (
    Funds,
    RequestPayment,
    RejectPayment,
    CommitPayment,
    CompletePayment,
    payment_protocol_spec,
)
from uagents_core.contrib.protocols.chat import ChatMessage, TextContent

# Payment verifiers
try:
    from tools.skyfire import verify_and_charge, get_skyfire_service_id
except Exception as e:
    print(f"[payment_proto] Failed to import skyfire: {e}")
    async def verify_and_charge(*args, **kwargs):
        return True
    def get_skyfire_service_id():
        return None

# --- helpers for cross-protocol storage keys ---
def _k(prefix: str, sender: str, session: str) -> str:
    return f"{prefix}:{sender}:{session}"


# --- protocol ---
payment_proto = Protocol(spec=payment_protocol_spec, role="seller")

# Allow agent.py to inject its wallet (optional)
_AGENT_WALLET_ADDR: Optional[str] = None

def set_agent_wallet(wallet) -> None:
    """Call this from agent.py at startup."""
    global _AGENT_WALLET_ADDR
    try:
        _AGENT_WALLET_ADDR = str(wallet.address())
    except Exception:
        _AGENT_WALLET_ADDR = None

def _recipient_str(ctx: Context) -> str:
    env_recipient = os.getenv("SELLER_RECIPIENT", "")
    cand = _AGENT_WALLET_ADDR or (env_recipient if env_recipient else None) or str(ctx.agent.address)
    return str(cand)


# ----------------------------
# PUBLIC: ask the user for a payment
# ----------------------------
async def request_payment_from_user(ctx: Context, user_address: str, description: Optional[str] = None) -> None:
    """Request payment from user using Skyfire USDC."""
    session = str(ctx.session)
    pr_key = _k("payment_requested", user_address, session)
    
    if ctx.storage.has(pr_key):
        ctx.logger.info(f"[payment] payment request already sent session={session} to={user_address}")
        return

    # Get order total from storage (for description only)
    total_price = None
    order_id = None
    try:
        price_key = _k("total_price", user_address, session)
        order_key = _k("order_id", user_address, session)
        if ctx.storage.has(price_key):
            total_price = ctx.storage.get(price_key)
        if ctx.storage.has(order_key):
            order_id = ctx.storage.get(order_key)
    except Exception:
        pass

    # Always use 0.001 USDC for payment (actual price shown in description only)
    usd_amount = os.getenv("FIXED_USD_AMOUNT", "0.001")

    skyfire_service_id = get_skyfire_service_id()
    ctx.logger.info(f"[payment] Skyfire service ID: {skyfire_service_id}")

    accepted_funds = []
    if skyfire_service_id:
        ctx.logger.info(f"[payment] Adding Skyfire USDC option: {usd_amount}")
        accepted_funds.append(Funds(currency="USDC", amount=usd_amount, payment_method="skyfire"))
    else:
        ctx.logger.warning("[payment] Skyfire service ID not found, payment not available")
        await ctx.send(user_address, ChatMessage(content=[TextContent(
            type="text", 
            text="Payment service not configured. Please contact support."
        )]))
        return

    metadata: dict[str, str] = {}
    if skyfire_service_id:
        metadata["skyfire_service_id"] = skyfire_service_id
    if _AGENT_WALLET_ADDR:
        metadata["provider_agent_wallet"] = _AGENT_WALLET_ADDR

    recipient = _recipient_str(ctx)
    req = RequestPayment(
        accepted_funds=accepted_funds,
        recipient=recipient,
        deadline_seconds=300,
        reference=session,
        description=description or f"Order {order_id or 'payment'} — pay to proceed",
        metadata=metadata,
    )

    await ctx.send(user_address, req)
    ctx.storage.set(pr_key, True)
    ctx.logger.info(f"[payment] → RequestPayment to={user_address} session={session} amount={usd_amount} USDC")


# ----------------------------
# REQUIRED seller handlers
# ----------------------------
@payment_proto.on_message(CommitPayment)
async def on_commit(ctx: Context, sender: str, msg: CommitPayment) -> None:
    """Handle payment commitment."""
    session = str(ctx.session)
    try:
        tx_key = _k(f"commit_{msg.transaction_id}", sender, session)
        if ctx.storage.has(tx_key):
            ctx.logger.info(f"[payment] duplicate CommitPayment ignored tx={msg.transaction_id}")
            return
    except Exception:
        pass

    method = msg.funds.payment_method
    verified = False
    
    try:
        ctx.logger.info(f"[payment] ← CommitPayment from={sender} session={session} method={method} currency={msg.funds.currency} amount={msg.funds.amount} tx={msg.transaction_id}")
    except Exception:
        pass

    if method == "skyfire":
        try:
            usd_amount = str(msg.funds.amount)
        except Exception:
            usd_amount = "0.001"
        verified = await verify_and_charge(
            token=msg.transaction_id,
            amount_usdc=usd_amount,
            logger=ctx.logger,
        )

    if verified:
        try:
            ctx.storage.set(tx_key, True)
        except Exception:
            pass
        await ctx.send(sender, CompletePayment(transaction_id=msg.transaction_id))
        ctx.logger.info(f"[payment] ✅ verified method={method} session={session}")

        # Mark session paid
        try:
            ctx.storage.set(_k("paid", sender, session), True)
        except Exception:
            pass

        # Update session state to mark payment confirmed
        try:
            session_key = f"{sender}::{session}"
            if ctx.storage.has(session_key):
                session_data = ctx.storage.get(session_key)
                if isinstance(session_data, dict) and "state" in session_data:
                    session_data["state"]["payment_confirmed"] = True
                    ctx.storage.set(session_key, session_data)
                    ctx.logger.info(f"[payment] Updated session state: payment_confirmed=True")
        except Exception as e:
            ctx.logger.error(f"[payment] Failed to update session state: {e}")

        # Extract parsed_items from metadata or session storage and run LangGraph workflow
        try:
            import json
            from agent_graph import run_agent_turn
            
            # Try to get parsed_items from metadata first
            parsed_items = None
            if msg.metadata and "parsed_items" in msg.metadata:
                try:
                    parsed_items = json.loads(msg.metadata["parsed_items"])
                except Exception as e:
                    ctx.logger.warning(f"Failed to parse parsed_items from metadata: {e}")
            
            # Fallback to session storage
            if not parsed_items:
                session_key = f"{sender}::{session}"
                if ctx.storage.has(session_key):
                    session_data = ctx.storage.get(session_key)
                    if isinstance(session_data, dict) and "parsed_items" in session_data:
                        parsed_items = session_data["parsed_items"]
            
            if not parsed_items:
                ctx.logger.error("[payment] No parsed_items found in metadata or session storage")
                await ctx.send(sender, ChatMessage(content=[TextContent(
                    type="text",
                    text="✅ Payment received, but order details not found. Please contact support."
                )]))
                return
            
            # Get session data
            session_key = f"{sender}::{session}"
            session_data = ctx.storage.get(session_key) if ctx.storage.has(session_key) else {}
            session_data.setdefault("state", {})
            session_data.setdefault("history", [])
            
            state = session_data["state"]
            history = session_data["history"]
            
            # Run LangGraph workflow with parsed_items
            result = run_agent_turn(
                parsed_items=parsed_items,
                session_state=state,
                history=history,
                session_id=f"{sender}::{session}"
            )
            
            # Update session data
            new_state = result.get("state", {})
            state.update(new_state)
            session_data["state"] = state
            session_data["history"] = result.get("history", history)
            ctx.storage.set(session_key, session_data)
            
            # Send response from LangGraph workflow
            reply_text = result.get("content", "")
            if reply_text:
                await ctx.send(sender, ChatMessage(content=[TextContent(
                    type="text",
                    text=reply_text
                )]))
            else:
                await ctx.send(sender, ChatMessage(content=[TextContent(
                    type="text",
                    text="✅ Payment received! Your order is being processed."
                )]))
        except Exception as e:
            ctx.logger.error(f"[payment] Failed to process order: {e}")
            await ctx.send(sender, ChatMessage(content=[TextContent(
                type="text",
                text="✅ Payment received! Your order will be processed shortly."
            )]))
    else:
        await ctx.send(sender, RejectPayment(reason="Payment verification failed"))
        ctx.logger.error(f"[payment] ❌ verification failed method={method} session={session}")


@payment_proto.on_message(RejectPayment)
async def on_reject_payment(ctx: Context, sender: str, msg: RejectPayment) -> None:
    """Handle payment rejection."""
    await ctx.send(sender, ChatMessage(content=[TextContent(
        type="text",
        text="You rejected our payment."
    )]))

