# agent.py
"""
Main agent file for the merchandiser agent.
Wraps LangGraph agent inside uAgent with chat and payment protocols.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import dotenv
from uagents import Agent, Context

# Ensure project root on sys.path so local packages can be imported
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables
dotenv.load_dotenv()
try:
    dotenv.load_dotenv(ROOT / ".env")
    dotenv.load_dotenv(ROOT.parent / ".env")
except Exception:
    pass

# Protocols
from protocols.chat_proto import chat_proto
from protocols.payment_proto import payment_proto, set_agent_wallet

# Optional: Skyfire helper (for logging / sanity check)
try:
    from tools.skyfire import get_skyfire_service_id
except Exception:
    def get_skyfire_service_id():
        return None

# Config
AGENT_NAME = os.getenv("AGENT_NAME", "Cashier Agent")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8031"))
AGENT_SEED = os.getenv("AGENT_SEED", "cashier_agent_seed_live_Innovation_Lab")

# Create the agent instance
agent = Agent(
    name=AGENT_NAME,
    port=AGENT_PORT,
    mailbox=True,
    seed=AGENT_SEED,
)

# Supply wallet to payment protocol for verification
set_agent_wallet(agent.wallet)

@agent.on_event("startup")
async def on_startup(ctx: Context):
    ctx.logger.info(f"{AGENT_NAME} is up. Wallet address: {agent.wallet.address()}")
    
    # Initialize database if it doesn't exist
    try:
        from tools.database import init_database
        init_database()
        ctx.logger.info("Database initialized/verified")
    except Exception as e:
        ctx.logger.error(f"Failed to initialize database: {e}")
    
    ssi = get_skyfire_service_id()
    if ssi:
        ctx.logger.info(f"Detected Skyfire service ID: {ssi}")
    else:
        ctx.logger.info("No Skyfire service ID configured (SELLER_SERVICE_ID missing).")

# Include protocols and publish their manifests
agent.include(chat_proto, publish_manifest=True)
agent.include(payment_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()

