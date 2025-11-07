"""
Multi-Agent Cashier System using LangGraph.
Accepts structured input: [{item_name: str, quantity: int}, ...]
Implements Stock Management, Warehouse, Cashier, and Restocker agents as nodes.
"""

from __future__ import annotations
import os
import logging
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from typing_extensions import NotRequired

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

from tools.database import (
    init_database,
    check_stock,
    check_stock_multiple,
    subtract_inventory,
    restock_item,
)

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The conversation messages"]
    user_request: NotRequired[str]
    parsed_items: NotRequired[List[Dict[str, Any]]]  # [{item_name: str, quantity: int}, ...]
    stock_check_results: NotRequired[Dict[str, Dict[str, Any]]]  # item_name -> {available_qty, requested_qty, is_available}
    stock_availability: NotRequired[bool]
    inventory_subtracted: NotRequired[bool]
    updated_stock_quantities: NotRequired[Dict[str, int]]
    payment_requested: NotRequired[bool]
    payment_status: NotRequired[str]  # "none", "pending", "successful", "failed"
    status: NotRequired[str]  # "parsing", "checking_stock", "rejected", "subtracting_inventory", "requesting_payment", "payment_pending", "payment_successful", "payment_failed", "restocking", "completed"
    restocked_items: NotRequired[List[str]]
    error_message: NotRequired[Optional[str]]


# Note: Parser Agent removed - system now accepts structured input directly


def stock_management_node(state: AgentState) -> AgentState:
    """
    Stock Management Agent Node: Checks database for stock availability.
    Sets stock_availability to False and status to "rejected" if items unavailable.
    """
    logger.info("📦 Stock Management Agent: Checking stock availability...")
    
    parsed_items = state.get("parsed_items", [])
    
    if not parsed_items:
        new_state = dict(state)
        new_state["status"] = "rejected"
        new_state["error_message"] = "No items to check stock for"
        new_state["stock_availability"] = False
        return new_state
    
    # Check stock for all items
    stock_check_results = {}
    all_available = True
    
    for item in parsed_items:
        item_name = item.get("item_name", "")
        requested_qty = item.get("quantity", 0)
        
        available_qty = check_stock(item_name)
        is_available = available_qty >= requested_qty
        
        stock_check_results[item_name] = {
            "available_qty": available_qty,
            "requested_qty": requested_qty,
            "is_available": is_available,
        }
        
        logger.info(
            f"   {item_name}: {available_qty} available, {requested_qty} requested - {'✅ Available' if is_available else '❌ Insufficient'}"
        )
        
        if not is_available:
            all_available = False
    
    new_state = dict(state)
    new_state["stock_check_results"] = stock_check_results
    new_state["stock_availability"] = all_available
    
    if all_available:
        new_state["status"] = "checking_stock"
        logger.info("✅ Stock Management Agent: All items available")
    else:
        new_state["status"] = "rejected"
        logger.info("❌ Stock Management Agent: Stock unavailable - rejecting payment")
        
        # Add rejection message
        unavailable_items = [
            item_name
            for item_name, data in stock_check_results.items()
            if not data["is_available"]
        ]
        unavailable_str = ", ".join(unavailable_items)
        rejection_msg = AIMessage(
            content=f"Sorry, we don't have enough stock for: {unavailable_str}. Payment rejected."
        )
        new_state["messages"] = list(state.get("messages", [])) + [rejection_msg]
    
    return new_state


def warehouse_node(state: AgentState) -> AgentState:
    """
    Warehouse Agent Node: Subtracts inventory from database.
    Only executes if stock_availability is True.
    """
    logger.info("🏭 Warehouse Agent: Subtracting inventory...")
    
    parsed_items = state.get("parsed_items", [])
    updated_stock_quantities = {}
    
    if not parsed_items:
        new_state = dict(state)
        new_state["status"] = "subtracting_inventory"
        new_state["error_message"] = "No items to subtract"
        return new_state
    
    # Subtract inventory for each item
    for item in parsed_items:
        item_name = item.get("item_name", "")
        quantity = item.get("quantity", 0)
        
        success = subtract_inventory(item_name, quantity)
        
        if success:
            # Get updated quantity
            updated_qty = check_stock(item_name)
            updated_stock_quantities[item_name] = updated_qty
            logger.info(f"   ✅ Subtracted {quantity}x {item_name}. Remaining: {updated_qty}")
        else:
            logger.error(f"   ❌ Failed to subtract {quantity}x {item_name}")
            new_state = dict(state)
            new_state["status"] = "subtracting_inventory"
            new_state["error_message"] = f"Failed to subtract inventory for {item_name}"
            return new_state
    
    new_state = dict(state)
    new_state["inventory_subtracted"] = True
    new_state["updated_stock_quantities"] = updated_stock_quantities
    new_state["status"] = "subtracting_inventory"
    
    # Add confirmation message
    items_str = ", ".join([f"{item['quantity']}x {item['item_name']}" for item in parsed_items])
    warehouse_msg = AIMessage(
        content=f"Inventory updated. Items reserved: {items_str}"
    )
    new_state["messages"] = list(state.get("messages", [])) + [warehouse_msg]
    
    logger.info("✅ Warehouse Agent: Inventory subtracted successfully")
    return new_state


def cashier_node(state: AgentState) -> AgentState:
    """
    Cashier Agent Node: Confirms order completion after payment.
    Payment has already been received (CommitPayment was sent before this workflow runs).
    Only executes if inventory_subtracted is True.
    """
    logger.info("💳 Cashier Agent: Processing order completion...")
    
    parsed_items = state.get("parsed_items", [])
    
    new_state = dict(state)
    new_state["payment_status"] = "successful"
    new_state["status"] = "completed"
    
    # Create order completion message (payment already received)
    items_summary = ", ".join([
        f"{item['quantity']}x {item['item_name']}"
        for item in parsed_items
    ])
    
    completion_msg = AIMessage(
        content=f"✅ Order completed successfully! Your purchase of {items_summary} has been processed. Thank you for your order!"
    )
    new_state["messages"] = list(state.get("messages", [])) + [completion_msg]
    
    logger.info("✅ Cashier Agent: Order completed")
    return new_state


def restocker_node(state: AgentState) -> AgentState:
    """
    Restocker Agent Node: Restocks items when payment fails.
    Only executes if payment_status is "failed".
    """
    logger.info("📥 Restocker Agent: Restocking items...")
    
    parsed_items = state.get("parsed_items", [])
    restocked_items = []
    
    if not parsed_items:
        new_state = dict(state)
        new_state["status"] = "restocking"
        new_state["error_message"] = "No items to restock"
        return new_state
    
    # Restock each item
    for item in parsed_items:
        item_name = item.get("item_name", "")
        quantity = item.get("quantity", 0)
        
        success = restock_item(item_name, quantity)
        
        if success:
            restocked_items.append(item_name)
            logger.info(f"   ✅ Restocked {quantity}x {item_name}")
        else:
            logger.error(f"   ❌ Failed to restock {quantity}x {item_name}")
    
    new_state = dict(state)
    new_state["restocked_items"] = restocked_items
    new_state["status"] = "completed"
    
    # Add restocking confirmation message
    if restocked_items:
        restock_msg = AIMessage(
            content=f"Items restocked: {', '.join(restocked_items)}. Inventory restored."
        )
        new_state["messages"] = list(state.get("messages", [])) + [restock_msg]
    
    logger.info(f"✅ Restocker Agent: Restocked {len(restocked_items)} items")
    return new_state
    

# Router functions for conditional edges
def route_after_stock_check(state: AgentState) -> str:
    """Route after stock check: warehouse if available, END if rejected."""
    stock_availability = state.get("stock_availability", False)
    
    if stock_availability:
        return "warehouse"
    else:
        # Status is already set to "rejected" in stock_management_node
        return END


def route_after_payment(state: AgentState) -> str:
    """Route after payment: restocker if failed, END if successful."""
    payment_status = state.get("payment_status", "none")
    
    if payment_status == "failed":
        return "restocker"
    elif payment_status == "successful":
        return END
    else:
        # Still pending, wait (in real scenario, this would timeout)
        # For testing, we'll simulate payment failure after some time
        return END


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("stock_management", stock_management_node)
workflow.add_node("warehouse", warehouse_node)
workflow.add_node("cashier", cashier_node)
workflow.add_node("restocker", restocker_node)

# Set entry point
workflow.set_entry_point("stock_management")

# Conditional edge after stock check
workflow.add_conditional_edges(
    "stock_management",
    route_after_stock_check,
    {
        "warehouse": "warehouse",
        END: END,
    }
)

# Edge from warehouse to cashier
workflow.add_edge("warehouse", "cashier")

# Conditional edge after payment (for testing, we'll simulate payment failure)
# In real scenario, payment status would be updated externally
workflow.add_conditional_edges(
    "cashier",
    route_after_payment,
    {
        "restocker": "restocker",
        END: END,
    }
)

# Edge from restocker to end
workflow.add_edge("restocker", END)

# Compile the graph with checkpointing
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def run_agent_turn(
    parsed_items: List[Dict[str, Any]],
    session_state: dict,
    history: list,
    session_id: str,
) -> dict:
    """
    Run a single turn of the LangGraph agent.
    
    Args:
        parsed_items: Structured input list of items with item_name and quantity
                      Format: [{"item_name": "...", "quantity": <number>}, ...]
        session_state: Current session state
        history: Conversation history
        session_id: Unique session identifier
    
    Returns:
        Dictionary with content, state, and history
    """
    # Convert history to LangChain messages
    messages = []
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content:
                messages.append(AIMessage(content=content))
    
    # Create config for checkpointing
    config = {"configurable": {"thread_id": session_id}}
    
    # Initialize state with parsed_items
    initial_state = {
        "messages": messages,
        "parsed_items": parsed_items,
        "status": "checking_stock",
    }
    
    # Copy existing state fields if present
    for key in [
        "user_request", "stock_check_results",
        "stock_availability", "inventory_subtracted", "updated_stock_quantities",
        "payment_requested", "payment_status",
        "restocked_items", "error_message"
    ]:
        if key in session_state:
            initial_state[key] = session_state[key]
    
    # Run the graph
    final_state = None
    for event in app.stream(initial_state, config):
        final_state = event
    
    # Extract the last message and state
    if final_state:
        # Get the state from the last event
        last_node = list(final_state.keys())[-1]
        state = final_state[last_node]
        messages = state.get("messages", [])
        
        # Get the last assistant message
        assistant_content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                assistant_content = msg.content
                break
        
        # Build new state dictionary
        new_state = {}
        for key in [
            "user_request", "parsed_items", "stock_check_results",
            "stock_availability", "inventory_subtracted", "updated_stock_quantities",
            "payment_requested", "payment_status", "status",
            "restocked_items", "error_message"
        ]:
            if key in state:
                new_state[key] = state[key]
        
        # Create user message for history
        items_str = ", ".join([f"{item['quantity']}x {item['item_name']}" for item in parsed_items])
        user_content = f"Purchase request: {items_str}"
        
        # Update history
        updated_history = history + [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        return {
            "content": assistant_content,
            "state": new_state,
            "history": updated_history,
        }
    
    return {
        "content": "Processing your order...",
        "state": session_state,
        "history": history,
    }


if __name__ == "__main__":
    """
    Standalone testing block for the multi-agent cashier system.
    """
    import dotenv
    
    # Load environment variables
    dotenv.load_dotenv()
    
    # Initialize database
    print("🔧 Initializing database...")
    init_database()
    
    # Test scenario 1: Successful purchase flow
    test_items = [{"item_name": "tshirt", "quantity": 2}]
    print(f"\n🧪 Test 1: Successful Purchase Flow")
    print(f"Request: {test_items}\n")
    
    # Initialize session
    session_id = "test_session_001"
    session_state = {}
    history = []
    
    # Run the agent
    result = run_agent_turn(
        parsed_items=test_items,
        session_state=session_state,
        history=history,
        session_id=session_id,
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS - Test 1")
    print("="*60)
    print(f"\n📝 Response: {result['content']}")
    print(f"\n📊 Final Status: {result['state'].get('status', 'N/A')}")
    
    if result['state'].get('parsed_items'):
        print(f"\n🛒 Parsed Items:")
        for item in result['state']['parsed_items']:
            print(f"   - {item['quantity']}x {item['item_name']}")
    
    if result['state'].get('stock_availability') is not None:
        print(f"\n📦 Stock Available: {result['state']['stock_availability']}")
    
    if result['state'].get('stock_check_results'):
        print(f"\n📊 Stock Check Results:")
        for item_name, data in result['state']['stock_check_results'].items():
            print(f"   - {item_name}: {data['available_qty']} available, {data['requested_qty']} requested")
    
    if result['state'].get('inventory_subtracted'):
        print(f"\n✅ Inventory Subtracted: Yes")
        if result['state'].get('updated_stock_quantities'):
            print(f"   Updated Quantities:")
            for item_name, qty in result['state']['updated_stock_quantities'].items():
                print(f"     - {item_name}: {qty}")
    
    if result['state'].get('payment_requested'):
        print(f"\n💳 Payment Requested: Yes")
        print(f"   Payment Status: {result['state'].get('payment_status', 'N/A')}")
    
    if result['state'].get('error_message'):
        print(f"\n❌ Error: {result['state']['error_message']}")
    
    # Test scenario 2: Payment failure -> Restocking
    # To test restocker, we need to simulate the state after cashier with payment failure
    print("\n\n" + "="*60)
    print("🧪 Test 2: Payment Failure -> Restocking")
    print("="*60)
    
    # Create state that simulates payment failure after cashier
    # This would normally come from the payment protocol
    test_messages = []
    if result['history']:
        for msg in result['history']:
            if msg.get("role") == "user":
                test_messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                test_messages.append(AIMessage(content=msg.get("content", "")))
    
    test_state_for_restocker = {
        "messages": test_messages,
        "parsed_items": result['state'].get('parsed_items', []),
        "inventory_subtracted": True,
        "payment_requested": True,
        "payment_status": "failed",  # Payment failed
        "status": "payment_failed",
    }
    
    # Manually test restocker node
    restocker_state = restocker_node(test_state_for_restocker)
    
    print(f"\n📝 Restocker Result")
    print(f"\n📊 Final Status: {restocker_state.get('status', 'N/A')}")
    
    if restocker_state.get('restocked_items'):
        print(f"\n📥 Restocked Items: {', '.join(restocker_state['restocked_items'])}")
    
    if restocker_state.get('messages'):
        for msg in restocker_state['messages']:
            if isinstance(msg, AIMessage):
                print(f"\n💬 Message: {msg.content}")
    
    # Test scenario 3: Stock unavailable -> Rejection
    print("\n\n" + "="*60)
    print("🧪 Test 3: Stock Unavailable -> Rejection")
    print("="*60)
    
    test_items_3 = [{"item_name": "tshirt", "quantity": 100}]  # More than available stock
    print(f"Request: {test_items_3}\n")
    
    result_3 = run_agent_turn(
        parsed_items=test_items_3,
        session_state={},
        history=[],
        session_id="test_session_003",
    )
    
    print(f"\n📝 Response: {result_3['content']}")
    print(f"\n📊 Final Status: {result_3['state'].get('status', 'N/A')}")
    
    if result_3['state'].get('stock_availability') is not None:
        print(f"\n📦 Stock Available: {result_3['state']['stock_availability']}")
    
    if result_3['state'].get('stock_check_results'):
        print(f"\n📊 Stock Check Results:")
        for item_name, data in result_3['state']['stock_check_results'].items():
            print(f"   - {item_name}: {data['available_qty']} available, {data['requested_qty']} requested")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
