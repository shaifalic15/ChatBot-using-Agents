from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import json
import os
from openai import AsyncOpenAI
from database import search_parts_db, get_part_db, search_repairs_db
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

try:
    repairs_vectorstore = FAISS.load_local(
        "data/repairs_vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )
    repairs_retriever = repairs_vectorstore.as_retriever(search_kwargs={"k": 5})
    logger.info("✅ Loaded repairs vector store")
except Exception as e:
    repairs_retriever = None
    logger.warning(f"⚠️ No vector store found: {e}")

class AgentState(TypedDict):
    query: str
    history: List[Dict]
    query_type: Optional[str]
    in_scope: bool
    entities: Dict
    tool_results: List[Dict]
    products: List[Dict]
    response: str

async def llm_call(prompt: str, model="gpt-5") -> str:
    """Call OpenAI LLM (gpt-5 via Responses API, others via Chat Completions)."""
    try:
        if str(model).startswith("gpt-5"):
            resp = await client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
                reasoning={"effort": "low"},       
                max_output_tokens=2000,           
            )

            text = (getattr(resp, "output_text", "") or "").strip()

            if not text and getattr(resp, "output", None):
                chunks = []
                for item in resp.output:
                    if getattr(item, "type", None) == "message":
                        for c in getattr(item, "content", []) or []:
                            if getattr(c, "type", None) in ("output_text", "text"):
                                chunks.append(getattr(c, "text", "") or "")
                text = "".join(chunks).strip()

            if not text:
                logger.error("Empty content returned from LLM (gpt-5)")
                return ""

            return text

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )

        content = (response.choices[0].message.content or "").strip()
        if not content:
            logger.error("Empty content returned from LLM")
            return ""
        return content

    except Exception as e:
        logger.error(f"LLM call error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ""

async def decide_tools(query: str, entities: Dict, query_type: str, previous_results: List[Dict] = None) -> Dict:
    """Use LLM to decide which database queries to run"""
    
    results_context = ""
    if previous_results:
        results_context = "\n\nPrevious Results Summary:\n"
        for result in previous_results[:3]:
            tool_name = result.get('tool', 'unknown')
            count = result.get('count', 0)
            results_context += f"- {tool_name}: Found {count} items\n"
    
    prompt = f"""You are a database query planner for PartSelect (refrigerator & dishwasher parts).

Query: {query}
Query Type: {query_type}
Detected Entities: {json.dumps(entities)}
{results_context}

Available Actions:
1. {{"type": "get_part", "part_number": "PS123"}} - Get specific part by PS number
2. {{"type": "search_parts", "query": "ice maker", "appliance_type": "refrigerator"}} - Full-text search for parts
3. {{"type": "search_repairs", "query": "not making ice", "appliance_type": "refrigerator"}} - Search repair guides
4. {{"type": "vector_search", "query": "ice maker troubleshooting"}} - Semantic search for symptoms

Decision Rules:
- If part_numbers detected in entities → ALWAYS use get_part for each part number
- If query_type is "troubleshooting" → ALWAYS run BOTH:
  1) vector_search (symptom-focused) AND
  2) search_parts (use the symptom/keyword from entities.symptoms[0] if available, else use the query)
- If query_type is "installation" → Use search_repairs and get_part if part number mentioned
- If query_type is "product_search" → Use search_parts
- If query_type is "compatibility" → Use get_part for the part number
- Use appliance_type from entities when available (refrigerator/dishwasher)
- If you already have enough information from previous results, set should_continue to false
- Maximum 2 actions per iteration

Return ONLY valid JSON (no markdown):
{{
  "actions": [
    {{"type": "get_part", "part_number": "PS11752778"}},
    {{"type": "search_repairs", "query": "ice maker installation", "appliance_type": "refrigerator"}}
  ],
  "should_continue": false
}}"""

    try:
        response = await llm_call(prompt, model="gpt-4o-mini")
        
        cleaned = re.sub(r'```json\n?|\n?```', '', response).strip()
        result = json.loads(cleaned)
        
        logger.info(f"Tool decision: {len(result.get('actions', []))} actions, continue={result.get('should_continue', False)}")
        return result
        
    except Exception as e:
        logger.error(f"Tool decision error: {e}")
        logger.error(f"Response was: {response[:200] if response else 'empty'}")
        
        fallback_actions = []
        
        if entities.get('part_numbers'):
            for part_num in entities['part_numbers'][:2]:
                fallback_actions.append({"type": "get_part", "part_number": part_num})
        elif query_type == 'troubleshooting':
            fallback_actions.append({"type": "vector_search", "query": query})
        elif query_type == 'installation':
            fallback_actions.append({"type": "search_repairs", "query": query, "appliance_type": entities.get('appliance_type')})
        else:
            fallback_actions.append({"type": "search_parts", "query": query, "appliance_type": entities.get('appliance_type')})
        
        return {
            "actions": fallback_actions,
            "should_continue": False
        }


async def analyze_node(state: AgentState) -> AgentState:
    """Node 1: Analyze query for scope and intent"""

    history_context = ""
    recent_part_numbers = []
    recent_model_numbers = []

    PART_RE = r"\bPS\d+\b"
    MODEL_RE = r"\b(?!PS\d+\b)[A-Z]{2,}[0-9]{3,}[A-Z0-9]*\b"

    if state.get("history"):
        history_context = "\n\nRecent Conversation:\n"
        for msg in state["history"][-5:]:  
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            history_context += f"{role}: {content[:150]}...\n" if len(content) > 150 else f"{role}: {content}\n"

            # Extract part numbers from history
            found_parts = re.findall(PART_RE, content.upper())
            recent_part_numbers.extend(found_parts)

            # Extract model numbers from history (exclude PS...)
            found_models = re.findall(MODEL_RE, content.upper())
            recent_model_numbers.extend(found_models)

    # Extract from current query
    query_text = state["query"]
    query_upper = query_text.upper()
    query_lower = query_text.lower()

    current_part_numbers = re.findall(PART_RE, query_upper)
    current_model_numbers = re.findall(MODEL_RE, query_upper)

    # If user says "this part" without specifying, use most recent part from history
    has_part_reference = any(phrase in query_lower for phrase in ["this part", "that part", "the part", "it"])

    if has_part_reference and not current_part_numbers and recent_part_numbers:
        logger.info(f"User referenced part contextually - using recent: {recent_part_numbers[-1]}")
        current_part_numbers = [recent_part_numbers[-1]]

    prompt = f"""Analyze this PartSelect customer query (refrigerator/dishwasher parts only):

Query: "{state['query']}"
{history_context}

Context: The user previously mentioned these parts: {list(set(recent_part_numbers)) if recent_part_numbers else "None"}

CRITICAL SCOPE RULES:
- IN SCOPE: ONLY queries about refrigerator parts OR dishwasher parts (appliances for food/dishes)
- OUT OF SCOPE: heaters, HVAC, furnaces, water heaters, space heaters, ovens, stoves, washers, dryers, microwaves, ANY other appliances
- If query is vague (just "heater"), assume OUT OF SCOPE unless context clearly indicates refrigerator/dishwasher

Examples:
- "How to fix my heater?" → OUT OF SCOPE (not refrigerator/dishwasher)
- "Ice maker not working" → IN SCOPE (refrigerator part)
- "Dishwasher won't drain" → IN SCOPE (dishwasher issue)
- "Oven temperature problem" → OUT OF SCOPE (not refrigerator/dishwasher)

Extract:
1. Is this in scope? (ONLY true if clearly about refrigerator OR dishwasher parts/issues)
2. Query type: installation|compatibility|troubleshooting|product_search|out_of_scope
3. Entities:
   - part_numbers: list of PS numbers
   - model_numbers: list of appliance model numbers (NOTE: PS numbers are NOT model numbers)
   - symptoms: list of problems/issues described
   - appliance_type: refrigerator|dishwasher|unknown

CRITICAL: If the query mentions "heater" without "refrigerator" or "dishwasher" context, mark as OUT OF SCOPE.

Return ONLY valid JSON (no markdown):
{{
  "in_scope": false,
  "query_type": "out_of_scope",
  "entities": {{
    "part_numbers": [],
    "model_numbers": [],
    "symptoms": [],
    "appliance_type": "unknown"
  }}
}}"""

    response = await llm_call(prompt)

    try:
        cleaned = re.sub(r"```json\n?|\n?```", "", response).strip()
        analysis = json.loads(cleaned)

        # Normalize structure
        if "entities" not in analysis or not isinstance(analysis["entities"], dict):
            analysis["entities"] = {"part_numbers": [], "model_numbers": [], "symptoms": [], "appliance_type": "unknown"}

        # Fallback: If LLM didn't extract part from history, do it manually
        if not analysis["entities"].get("part_numbers") or analysis["entities"]["part_numbers"] == [""]:
            if has_part_reference and recent_part_numbers:
                analysis["entities"]["part_numbers"] = [recent_part_numbers[-1]]
                logger.info(f"Manually added part from history: {recent_part_numbers[-1]}")
            elif current_part_numbers:
                analysis["entities"]["part_numbers"] = current_part_numbers

        # Add model numbers from query/history if missing
        if not analysis["entities"].get("model_numbers") or analysis["entities"]["model_numbers"] == [""]:
            if current_model_numbers:
                analysis["entities"]["model_numbers"] = current_model_numbers
            elif recent_model_numbers:
                analysis["entities"]["model_numbers"] = [recent_model_numbers[-1]]

    except Exception as e:
        logger.warning(f"Failed to parse analysis: {e}")
        logger.error(f"Response was: {response[:200]}")

        # Manual extraction fallback
        all_parts = list(set(current_part_numbers + recent_part_numbers))
        all_models = list(set(current_model_numbers + recent_model_numbers))

        analysis = {
            "in_scope": True,
            "query_type": "compatibility" if all_models else "product_search",
            "entities": {
                "part_numbers": all_parts[:1] if all_parts else [],
                "model_numbers": all_models[:1] if all_models else [],
                "symptoms": [],
                "appliance_type": "unknown",
            },
        }

    # --- SIMPLE OVERRIDE RULES (make it robust across models) ---
    entities = analysis.get("entities", {}) if isinstance(analysis.get("entities", {}), dict) else {}
    parts = entities.get("part_numbers", []) or []
    models = entities.get("model_numbers", []) or []

    # Never allow PS parts to be treated as models
    models = [m for m in models if not str(m).upper().startswith("PS")]
    entities["model_numbers"] = models
    analysis["entities"] = entities

    # Install + valid part number => always in-scope installation
    if ("install" in query_lower or "installation" in query_lower) and parts:
        analysis["in_scope"] = True
        analysis["query_type"] = "installation"

    logger.info(
        f"Analysis: type={analysis.get('query_type')}, parts={analysis.get('entities', {}).get('part_numbers')}, models={analysis.get('entities', {}).get('model_numbers')}"
    )

    return {
        **state,
        "in_scope": analysis.get("in_scope", True),
        "query_type": analysis.get("query_type", "product_search"),
        "entities": analysis.get("entities", {}),
    }

async def retrieve_node(state: AgentState) -> AgentState:
    """Node 2: Agentically retrieve information using LLM-guided tool calling"""
    
    if not state['in_scope']:
        return state
    
    query = state['query']
    entities = state['entities']
    query_type = state['query_type']
    tool_results = []
    products = []
    
    logger.info(f"Starting agentic retrieval for query type: {query_type}")
    
    # Agentic tool calling loop
    max_iterations = 3
    iteration = 0
    
    while iteration < max_iterations:
        logger.info(f"\n--- Retrieval Iteration {iteration + 1}/{max_iterations} ---")
        
        # LLM decides next actions
        decision = await decide_tools(query, entities, query_type, tool_results)
        
        if not decision.get('actions'):
            logger.info("No more actions planned")
            break
        
        # Execute each action
        for action in decision['actions']:
            action_type = action.get('type')
            
            try:
                if action_type == 'get_part':
                    part_num = action.get('part_number')
                    logger.info(f"  → Getting part {part_num}")
                    part = get_part_db(part_num)
                    if part:
                        part_dict = dict(part)
                        products.append(part_dict)
                        tool_results.append({
                            'tool': 'get_part',
                            'result': part_dict,
                            'count': 1
                        })
                        logger.info(f"  ✓ Found part: {part_dict.get('part_type', 'N/A')}")
                    else:
                        logger.info(f"  ✗ Part {part_num} not found")
                
                elif action_type == 'search_parts':
                    search_query = action.get('query', query)
                    appliance_type = action.get('appliance_type')
                    logger.info(f"  → Searching parts: '{search_query}' ({appliance_type or 'all'})")
                    parts = search_parts_db(search_query, appliance_type, limit=5)
                    if parts:
                        parts_list = [dict(p) for p in parts]
                        products.extend(parts_list)
                        tool_results.append({
                            'tool': 'search_parts_db',
                            'result': parts_list,
                            'count': len(parts_list)
                        })
                        logger.info(f"  ✓ Found {len(parts_list)} parts")
                    else:
                        logger.info(f"  ✗ No parts found")
                
                elif action_type == 'search_repairs':
                    search_query = action.get('query', query)
                    appliance_type = action.get('appliance_type')
                    logger.info(f"  → Searching repairs: '{search_query}' ({appliance_type or 'all'})")
                    repairs = search_repairs_db(search_query, appliance_type)
                    if repairs:
                        repairs_list = [dict(r) for r in repairs]
                        tool_results.append({
                            'tool': 'search_repairs_db',
                            'result': repairs_list,
                            'count': len(repairs_list)
                        })
                        logger.info(f"  ✓ Found {len(repairs_list)} repair guides")
                    else:
                        logger.info(f"  ✗ No repairs found")
                
                elif action_type == 'vector_search':
                    if repairs_retriever:
                        search_query = action.get('query', query)
                        logger.info(f"  → Vector search: '{search_query}'")
                        docs = repairs_retriever.invoke(search_query)
                        
                        rag_results = []
                        for doc in docs[:5]:
                            rag_results.append({
                                'symptom': doc.metadata.get('symptom', ''),
                                'description': doc.page_content[:300],
                                'appliance_type': doc.metadata.get('appliance_type', '')
                            })
                        
                        if rag_results:
                            tool_results.append({
                                'tool': 'rag_search',
                                'result': rag_results,
                                'count': len(rag_results)
                            })
                            logger.info(f"  ✓ RAG found {len(rag_results)} results")
                    else:
                        logger.warning("  ✗ Vector store not available")
            
            except Exception as e:
                logger.error(f"  ✗ Error executing {action_type}: {e}")
        
        iteration += 1
        
        # Check if LLM wants to continue
        if not decision.get('should_continue', False):
            logger.info("LLM decided retrieval is complete")
            break
    
    logger.info(f"Retrieval complete: {len(products)} products, {len(tool_results)} tool results")
    
    return {
        **state,
        "tool_results": tool_results,
        "products": products
    }

async def generate_node(state: AgentState) -> AgentState:
    """Node 3: Generate final response"""
    
    if not state['in_scope']:
        return {
            **state,
            "response": "I appreciate you reaching out! However, I specialize specifically in refrigerator and dishwasher parts. For help with appliances other than these two, you'd want to connect with a specialist in that area. But if you have any questions about refrigerator or dishwasher parts, I'm here to help!"
        }
    
    # Helper function to convert Decimal to float
    def serialize_data(obj):
        """Convert Decimal and other non-serializable types"""
        from decimal import Decimal
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: serialize_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_data(item) for item in obj]
        return obj
    
    # Build context from tool results
    context_parts = []
    
    for result in state['tool_results']:
        tool_name = result['tool']
        data = serialize_data(result['result'])
        
        if tool_name == 'rag_search':
            context_parts.append("=== Troubleshooting Information ===")
            for item in data[:3]:
                context_parts.append(
                    f"Symptom: {item.get('symptom', 'N/A')}\n"
                    f"Description: {item.get('description', 'N/A')}"
                )
        
        elif tool_name == 'search_repairs_db':
            context_parts.append("=== Repair Guides ===")
            for repair in data[:3]:
                parts_list = repair.get('parts', [])
                parts_str = ', '.join(parts_list[:3]) if parts_list else 'N/A'
                
                context_parts.append(
                    f"Symptom: {repair.get('symptom', 'N/A')}\n"
                    f"Description: {repair.get('symptom_description', 'N/A')[:200]}\n"
                    f"Parts needed: {parts_str}\n"
                    f"Repair Video: {repair.get('video_tutorial_url', 'N/A')}\n"
                    f"Guide URL: {repair.get('detailed_guide_url', 'N/A')}"
                )
        
        elif tool_name == 'get_part':
            part = data
            context_parts.append("=== Part Information ===")
            part_info = f"""Part Number: {part.get('part_number', 'N/A')}
Type: {part.get('part_type', 'N/A')}
Manufacturer Part Number: {part.get('man_part_number', 'N/A')}
Price: ${part.get('price', 0.0)}
Brand: {part.get('brand', 'N/A')}
Stock Status: {part.get('stock_status', 'N/A')}"""
            
            if part.get('installation_video_url'):
                part_info += f"\nInstallation Video: {part.get('installation_video_url')}"
            
            if part.get('item_url'):
                part_info += f"\nProduct Page: {part.get('item_url')}"
            
            if part.get('symptoms'):
                symptoms = part.get('symptoms', [])
                if symptoms:
                    part_info += f"\nCommon Symptoms: {', '.join(symptoms[:5])}"
            
            context_parts.append(part_info)
        
        elif tool_name == 'search_parts_db':
            context_parts.append(f"=== Found {len(data)} Parts ===")
            for i, part in enumerate(data[:3], 1):
                context_parts.append(
                    f"Part {i}:\n"
                    f"  - Part Number: {part.get('part_number', 'N/A')}\n"
                    f"  - Type: {part.get('part_type', 'N/A')}\n"
                    f"  - Price: ${part.get('price', 0.0)}\n"
                    f"  - Stock: {part.get('stock_status', 'N/A')}"
                )
    
    context = "\n\n".join(context_parts) if context_parts else "No specific data retrieved."
    
    # Build product summary
    product_info = ""
    if state['products']:
        serialized_products = serialize_data(state['products'])
        product_info = f"\n\n=== Available Products ({len(serialized_products)}) ===\n"
        
        for i, prod in enumerate(serialized_products[:5], 1):
            product_info += f"\nProduct {i}:\n"
            product_info += f"  Part Number: {prod.get('part_number', 'N/A')}\n"
            product_info += f"  Type: {prod.get('part_type', 'N/A')}\n"
            product_info += f"  Price: ${prod.get('price', 0.0)}\n"
            product_info += f"  Brand: {prod.get('brand', 'N/A')}\n"
            product_info += f"  Stock: {prod.get('stock_status', 'N/A')}\n"
            
            if prod.get('installation_video_url'):
                product_info += f"  Installation Video: {prod.get('installation_video_url')}\n"
            
            if prod.get('item_url'):
                product_info += f"  Product Page: {prod.get('item_url')}\n"
    
    prompt = f"""You are a knowledgeable and friendly PartSelect customer service representative helping customers with refrigerator and dishwasher parts.

Customer Query: {state['query']}
Query Type: {state['query_type']}

Retrieved Information:
{context}
{product_info}

TONE & PERSONALITY:
- Warm, friendly, and genuinely helpful (like a neighbor helping out)
- Conversational and natural - talk like a real person
- Reassuring and encouraging - make them feel confident
- Use phrases like: "Great question!", "Happy to help!", "I've got you covered!", "Good news!"
- Show enthusiasm when you find what they need
- Be empathetic if they're having problems

RESPONSE GUIDELINES BY QUERY TYPE:

Installation Questions:
- Start warm: "Happy to help with that!", "Great question!", "I've got you covered!"
- Reassure them: "The good news is..." or "This is actually straightforward..."
- If an Installation Video URL is available in the retrieved data:
  - DO NOT write step-by-step installation instructions.
  - Give only a 1–2 sentence high-level overview, then direct them to the video.
- If NO Installation Video URL is available:
  - Give a short 3–5 step numbered list (each step 1 sentence).
- Mention part details naturally: part name, part number, MPN, price, stock status.
- End warmly: "Let me know if you have any questions along the way - I'm here to help!"
- Length: 90-120 words

Compatibility Questions:
- Start friendly: "Absolutely, I can help you check that!", "Great question about compatibility!"
- Give context warmly: "The part you're asking about is the [part name] - it's [price] and in stock"
- Make it easy: "Here's the easiest way to verify this will work with your [model]..."
- Guide them with an inline link: "Open the [PSxxxx product page](URL) and use the compatibility checker right below the price."
- Reassure: "That tool will confirm if it's the right fit for your specific model"
- Close warmly: "Let me know if you need help with anything else!"
- Length: 100-120 words

Troubleshooting Questions:
- Start empathetic: "I'm sorry you're dealing with that issue!", "That's frustrating - let me help!"
- Show understanding: "I know how annoying it is when..."
- Be solution-focused: "Here's what commonly fixes this problem..."
- You MUST recommend at least 1 replacement part from "=== Available Products ===" (pick Product 1 if multiple).
- Include it inline with a Markdown link using the product URL:
  Example: The most common fix is the [<part_type> (PSxxxx)](<Product Page URL>).
- Also include price + stock status for that same part.
- If no products were retrieved, say you couldn’t find a specific part in the catalog yet and ask for the appliance model number.
- Offer resources, with an inline link: "If you want to see it done, here’s the [installation video](URL)."
- End supportively: "Hopefully that gets things working smoothly again! Let me know if you need anything else"
- Length: 120-150 words

Product Search Questions:
- Start enthusiastic: "I found what you're looking for!", "Great - here's what I have for you!"
- Describe warmly: "This is the [part name] - it's a [brand] part..."
- Share helpful details: "It's priced at [price] and in stock, so you can get it right away"
- Mention benefits: "This part addresses [symptoms] and comes with installation support"
- Close helpfully: "I can help you check compatibility or answer any other questions!"
- Length: 100-120 words

WRITING STYLE:
- Write in warm, flowing paragraphs (not bullet points)
- Use contractions (you'll, it's, that's) to sound natural
- Add transitional phrases: "Here's what I found...", "The good news is...", "To make this easy..."
- Be encouraging: "This should be straightforward", "You're all set", "This'll work great"
- Show care: "Let me know if...", "I'm here to help...", "Feel free to ask..."
- Keep it conversational - imagine you're talking face-to-face

CRITICAL RULES:
- Always stay on topic - answer what they asked with relevant context
- Use actual URLs from data (never placeholders)
- Don't mention installation difficulty/time (not in database)
- Be accurate - only use retrieved information
- Length varies by query type (see above)

LINK RULES (STRICT):
- Write in Markdown.
- Whenever you mention “product page”, “compatibility checker”, “installation video”, or “guide”, you MUST include the link inline in that same sentence as a Markdown link.
  Example: Use the compatibility checker on the [PS11752778 product page](URL).
- Link text must be self-explanatory (include part number or part name). No bare “Product page” labels.
- Use ONLY URLs that appear in Retrieved Information / Available Products.
- You may still include a final “Helpful links” section (only if query_type is NOT troubleshooting), but links must ALSO appear inline the first time you reference them.
- If query_type is troubleshooting: do NOT include a “Helpful links” section.

FORMAT RULES (STRICT):
- Reply in Markdown.
- Use short paragraphs with a blank line ONLY between paragraphs.
- Do NOT paste raw URLs. Always embed them like [Product page](URL) and [Installation video](URL).
- If query_type is NOT troubleshooting and you have URLs, end with:

  Helpful links
  - [Product page](URL)
  - [Installation video](URL)

- If query_type is troubleshooting, do NOT include Helpful links.

Response:"""

    logger.info(f"Generating response with {len(context)} chars of context")
    
    response = await llm_call(prompt)
    
    if not response:
        logger.error("LLM returned empty response")
        response = "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    return {
        **state,
        "response": response
    }

# Build LangGraph
def create_agent():
    """Create the LangGraph agent workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    # Define flow
    workflow.set_entry_point("analyze")
    
    # Conditional routing after analyze
    workflow.add_conditional_edges(
        "analyze",
        lambda s: "retrieve" if s['in_scope'] else "generate"
    )
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# Create global agent instance
agent = create_agent()
logger.info("✅ Agent initialized with agentic tool calling")

async def process_query(query: str, history: List[Dict] = []) -> Dict:
    """Main entry point - process user query through agent"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing query: {query}")
    logger.info(f"{'='*60}")
    
    initial_state = {
        'query': query,
        'history': history,
        'query_type': None,
        'in_scope': True,
        'entities': {},
        'tool_results': [],
        'products': [],
        'response': ''
    }
    
    try:
        final_state = await agent.ainvoke(initial_state)
        
        # Serialize products for JSON response
        from decimal import Decimal
        def serialize_for_json(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: serialize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_for_json(item) for item in obj]
            return obj
        
        serialized_products = serialize_for_json(final_state['products'])
        
        return {
            'response': final_state['response'],
            'products': serialized_products,
            'query_type': final_state['query_type']
        }
    except Exception as e:
        logger.error(f"Agent error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'response': "I apologize, but I encountered an error processing your request. Please try again.",
            'products': [],
            'query_type': 'error'
        }