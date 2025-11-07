# protocols/chat_proto.py
"""
Chat protocol adapter for the merchandiser agent.
Integrates LangGraph agent with uAgents chat protocol.
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional, List
from uuid import uuid4

from uagents import Protocol, Context
from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    chat_protocol_spec,
)
from uagents_core.contrib.protocols.payment import (
    Funds,
    RequestPayment,
    payment_protocol_spec,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Protocol initialization
chat_proto = Protocol(spec=chat_protocol_spec)

# Pydantic models for structured output
class ParsedItem(BaseModel):
    item_name: str = Field(description="Name of the item")
    quantity: int = Field(description="Quantity requested", ge=1)


class ParsedItems(BaseModel):
    items: List[ParsedItem] = Field(description="List of items and quantities")


# Initialize LLM for parsing
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    temperature=0,
)


def _get_session_key(sender: str, session_id: str) -> str:
    """Generate a unique storage key for sender + session."""
    return f"{sender}::{session_id}"


def _get_session_data(ctx: Context, sender: str, session_id: str) -> Dict[str, Any]:
    """Retrieve or initialize session data for this sender + session."""
    key = _get_session_key(sender, session_id)
    session_data = ctx.storage.get(key) or {}
    session_data.setdefault("state", {
        "order_id": None,
        "item": None,
        "quantity": None,
        "total_price": None,
        "payment_requested": False,
        "payment_confirmed": False,
        "order_dispatched": False,
    })
    session_data.setdefault("history", [])
    return session_data


def _save_session_data(ctx: Context, sender: str, session_id: str, session_data: Dict[str, Any]) -> None:
    """Save session data for this sender + session."""
    key = _get_session_key(sender, session_id)
    ctx.storage.set(key, session_data)


def _extract_text(msg: ChatMessage) -> str:
    """Extract text content from ChatMessage."""
    parts = []
    for item in msg.content or []:
        if isinstance(item, TextContent) and item.text:
            parts.append(item.text)
    return "\n".join(parts).strip()


async def _ack(ctx: Context, sender: str, msg: ChatMessage) -> None:
    """Send acknowledgement."""
    try:
        await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))
    except Exception:
        pass


@chat_proto.on_message(ChatMessage)
async def handle_chat(ctx: Context, sender: str, msg: ChatMessage) -> None:
    """Handle incoming chat messages."""
    ctx.logger.info(f"Chat message from {sender}")
    await _ack(ctx, sender, msg)

    text = _extract_text(msg)
    if not text:
        return

    # Parse user text using GPT-4o structured output
    try:
        structured_llm = llm.with_structured_output(ParsedItems)
        
        prompt = f"""Parse the following purchase request and extract all items and quantities.

User request: {text}

Extract all items mentioned and their quantities. If no quantity is specified, assume 1.
Return a structured list of items with their quantities."""

        result = structured_llm.invoke(prompt)
        
        # Convert Pydantic model to list of dicts
        parsed_items = [
            {"item_name": item.item_name, "quantity": item.quantity}
            for item in result.items
        ]
        
        ctx.logger.info(f"Parsed {len(parsed_items)} items from user request")
        for item in parsed_items:
            ctx.logger.info(f"   - {item['quantity']}x {item['item_name']}")
        
    except Exception as e:
        ctx.logger.error(f"Failed to parse user request: {e}")
        await ctx.send(sender, ChatMessage(content=[TextContent(
            type="text",
            text="Sorry, I couldn't understand your request. Please specify items and quantities."
        )]))
        return

    # Get session-specific data
    session_id = str(ctx.session)
    session_data = _get_session_data(ctx, sender, session_id)
    
    # Store parsed_items in session data
    session_data["parsed_items"] = parsed_items
    _save_session_data(ctx, sender, session_id, session_data)

    # Check stock availability BEFORE processing payment (Cashier Agent stock check)
    try:
        from tools.database import check_stock_multiple
        
        # Check stock for all items
        stock_results = check_stock_multiple(parsed_items)
        unavailable_items = []
        
        for item in parsed_items:
            item_name = item["item_name"]
            requested_qty = item["quantity"]
            available_qty = stock_results.get(item_name, 0)
            
            if available_qty < requested_qty:
                unavailable_items.append({
                    "item_name": item_name,
                    "requested": requested_qty,
                    "available": available_qty
                })
        
        # If any items are unavailable, send ChatMessage and return
        if unavailable_items:
            unavailable_parts = []
            for item in unavailable_items:
                unavailable_parts.append(
                    f"{item['item_name']} (requested: {item['requested']}, available: {item['available']})"
                )
            unavailable_msg = f"Sorry, we don't have enough stock for: {', '.join(unavailable_parts)}. Please adjust your order."
            
            ctx.logger.info(f"Stock unavailable for items: {unavailable_items}")
            await ctx.send(sender, ChatMessage(content=[TextContent(
                type="text",
                text=unavailable_msg
            )]))
            return
        
        ctx.logger.info(f"Stock check passed for all {len(parsed_items)} items")
        
    except Exception as e:
        ctx.logger.error(f"Failed to check stock: {e}")
        await ctx.send(sender, ChatMessage(content=[TextContent(
            type="text",
            text="Sorry, I encountered an error checking stock availability."
        )]))
        return

    # Send RequestPayment with parsed_items in metadata (only if stock is available)
    try:
        # Get Skyfire service ID
        try:
            from tools.skyfire import get_skyfire_service_id
            skyfire_service_id = get_skyfire_service_id()
        except Exception:
            skyfire_service_id = None
        
        if not skyfire_service_id:
            await ctx.send(sender, ChatMessage(content=[TextContent(
                type="text",
                text="Payment service not configured. Please contact support."
            )]))
            return

        # Prepare metadata with parsed_items
        metadata = {
            "parsed_items": json.dumps(parsed_items),  # Store as JSON string in metadata
            "skyfire_service_id": skyfire_service_id,
        }
        
        # Add agent wallet if available
        try:
            from protocols.payment_proto import _AGENT_WALLET_ADDR
            if _AGENT_WALLET_ADDR:
                metadata["provider_agent_wallet"] = _AGENT_WALLET_ADDR
        except Exception:
            pass

        # Calculate total price from database (Cashier Agent logic in uAgent)
        from tools.database import get_items_prices
        
        prices = get_items_prices(parsed_items)
        total_price = 0.0
        items_with_prices = []
        
        for item in parsed_items:
            item_name = item["item_name"]
            quantity = item["quantity"]
            price = prices.get(item_name, 0.0)
            item_total = price * quantity
            total_price += item_total
            items_with_prices.append({
                "item_name": item_name,
                "quantity": quantity,
                "unit_price": price,
                "total_price": item_total
            })
        
        # Store price information in metadata
        metadata["total_price"] = str(total_price)
        metadata["items_with_prices"] = json.dumps(items_with_prices)
        
        # Create payment request - use agent address as recipient
        recipient = str(ctx.agent.address)
        # Use fixed amount for actual payment (testing), but show real price in description
        usd_amount = os.getenv("FIXED_USD_AMOUNT", "0.001")
        
        # Create description from parsed items with prices
        items_desc_parts = []
        for item in parsed_items:
            item_name = item["item_name"]
            quantity = item["quantity"]
            price = prices.get(item_name, 0.0)
            items_desc_parts.append(f"{quantity}x {item_name} @ ${price:.2f} each")
        
        items_desc = ", ".join(items_desc_parts)
        description = f"Purchase: {items_desc} | Total: ${total_price:.2f} USDC"
        
        req = RequestPayment(
            accepted_funds=[Funds(currency="USDC", amount=usd_amount, payment_method="skyfire")],
            recipient=recipient,
            deadline_seconds=300,
            reference=str(uuid4()),
            description=description,
            metadata=metadata,
        )
        
        await ctx.send(sender, req)
        ctx.logger.info(f"Sent RequestPayment to {sender} with {len(parsed_items)} items | Total: ${total_price:.2f} USDC")
        
    except Exception as e:
        ctx.logger.error(f"Failed to send payment request: {e}")
        await ctx.send(sender, ChatMessage(content=[TextContent(
            type="text",
            text="Sorry, I encountered an error processing your request."
        )]))


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement) -> None:
    """Handle acknowledgements."""
    ctx.logger.info(f"Ack received from {sender} for {msg.acknowledged_msg_id}")

