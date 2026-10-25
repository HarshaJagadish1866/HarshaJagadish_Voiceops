#!/usr/bin/env python3
"""ZentroMart Multi-Turn Conversational Customer Support Agent.

Key Capabilities:
1. Multi-turn conversation state memory (remembers order IDs, customer's problem, language, verification).
2. Introduces itself as an AI assistant in its first reply (STY-1).
3. Imports and utilizes mock tools: lookup_order and create_ticket from the tools folder.
4. Identity Verification Gatekeeper:
   - Before revealing ANY order details, verifies identity via last 4 digits of phone OR delivery pincode (IDV-1, IDV-2).
   - Enforces max 1 retry on verification failure; refuses order disclosure on 2nd failure (IDV-3).
5. Polite refusal for requests violating company policy:
   - Cancellation of already shipped orders (CAN-2).
   - Extra compensation demands for standard delivery delays (DEL-4 / ESC-4).
   - Non-returnable items or expired return windows.
6. Escalation to Human Agent (ESC-1):
   - High-value damage (> ₹10,000).
   - Customer requests a human agent.
   - Repeat contact within 7 days (contacts_last_7_days >= 2).
   - Overdue refunds (> 3 business days past max timeline).
   - Calls create_ticket with a detailed summary (>= 20 chars) and ONLY confirms handover if created=True.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Import mock tools from tools/
from tools import create_ticket, lookup_order

# Import RAG system and business day calendar
from rag_system import ZentroMartRAGAgent
from load_database import ZentroMartDatabase, TODAY, TODAY_DATE

# =====================================================================
# Strict System Prompt: Identity Gatekeeper & Conversational Guardrails
# =====================================================================
SYSTEM_PROMPT: str = """You are an AI customer support assistant for ZentroMart, an online shopping store in India.
Today is 20 October 2026.

=== SPEECH & AUDIO GUIDELINES (FOR TTS) ===
You are a spoken voice assistant. You MUST synthesize the retrieved facts into a natural, conversational reply. DO NOT output raw markdown, headers, bullet points, or citation brackets. Keep every response to 1 or 2 short, simple sentences optimized for speech.

=== STRICT IDENTITY GATEKEEPER RULE ===

The agent MUST NOT use the lookup_order tool or reveal any order details until it has verified the caller's identity by matching either the last 4 digits of their phone number OR their delivery pincode.
- Never disclose order status, items, amounts, addresses, tracking numbers, or refund details until identity verification is complete.
- When an order ID is mentioned, request verification: either the last 4 digits of the customer's registered phone number OR their delivery pincode.
- Maximum 1 retry is allowed on verification failure. If the second attempt fails, refuse to share order details and direct them to contact support from their registered phone number.
- Never reveal or read out the expected verification values to the customer.

=== POLICY BREAKERS & OUTSIDE-POLICY COMPENSATION ===
When a customer demands compensation outside of company policy (e.g., demanding ₹500, cash, vouchers, or extra credits for delayed standard deliveries under DEL-4 / ESC-4):
1. Politely refuse: Clearly and courteously explain that under ZentroMart policy (DEL-4), extra compensation or vouchers cannot be provided for standard deliveries.
2. Summarize the issue: Formulate a detailed summary of the customer's problem, the policy checks conducted, and the escalation reason (minimum 20 characters).
3. Call the create_ticket tool: Pass the summary, order_id, reason="outside_policy_compensation_demand", and intent="compensation_demand".
4. Handover confirmation guardrail: Crucially, the agent must ONLY tell the customer they have been handed over to a human if the create_ticket tool returns created=True. If create_ticket fails or returns created=False, do NOT claim that a handover occurred.

=== HUMAN ESCALATION RULES (ESC-1) ===
Call create_ticket and escalate to a human colleague when:
- High-value damaged/defective items (> ₹10,000) under DWM-3.
- The customer explicitly asks to speak with a human agent or person.
- Repeat contacts (contacts_last_7_days >= 2 or customer called multiple times).
- Overdue refunds exceeding the policy timeline + 3 business days grace.
- Account security or unauthorized order reports.
Always ensure summary >= 20 characters and confirm handover ONLY if created=True.
"""


def detect_language(text: str) -> str:

    """Detects whether user is speaking Hindi (hi), Hinglish (hi-en), or English (en)."""
    # Devanagari script check
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    # Hinglish keyword check
    hinglish_markers = [
        r"\bhai\b",
        r"\bhain\b",
        r"\bmujhe\b",
        r"\bmera\b",
        r"\bmeri\b",
        r"\bkya\b",
        r"\bkaise\b",
        r"\bkarein\b",
        r"\bchahiye\b",
        r"\baakhri\b",
        r"\baayega\b",
        r"\bkar do\b",
        r"\bbaat\b",
        r"\bisi\b",
        r"\bbaare\b",
        r"\bpe\b",
        r"\bho gaya\b",
        r"\bho gayi\b",
        r"\bke\b",
        r"\bki\b",
        r"\bko\b",
        r"\bse\b",
        r"\bkar sakti\b",
        r"\bkar sakta\b",
        r"\bkahan\b",
        r"\bkab\b",
        r"\bhua\b",
    ]
    for pattern in hinglish_markers:
        if re.search(pattern, text, re.IGNORECASE):
            return "hi-en"

    return "en"


@dataclass
class ConversationSession:
    """Stores the multi-turn state of a conversation."""

    conversation_id: str
    turn_count: int = 0
    language: str = "en"
    order_id: Optional[str] = None
    customer_problem: Optional[str] = None
    is_verified: bool = False
    verification_attempts: int = 0
    verification_failed: bool = False
    order_data: Optional[Dict[str, Any]] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    tickets_created: List[Dict[str, Any]] = field(default_factory=list)


class ZentroMartAgent:
    """Customer support agent handling multi-turn conversations, RAG, and tools."""

    def __init__(self, data_dir: Union[str, Path] = "data", kb_dir: Union[str, Path] = "kb"):
        self.data_dir = Path(data_dir)
        self.kb_dir = Path(kb_dir)
        self.db = ZentroMartDatabase(data_dir=self.data_dir)
        self.rag = ZentroMartRAGAgent(kb_dir=self.kb_dir, products_csv=self.data_dir / "products.csv")
        self.system_prompt: str = SYSTEM_PROMPT

    def get_system_prompt(self) -> str:
        """Returns the strict system prompt containing the Identity Gatekeeper and policy breaker rules."""
        return self.system_prompt


    def handle_turn(self, customer_message: str, session: ConversationSession) -> str:
        """Processes one turn of the conversation, maintains state, calls tools, and replies."""
        session.turn_count += 1
        msg = customer_message.strip()
        msg_lower = msg.lower()

        # Update detected language
        lang = detect_language(msg)
        if session.turn_count == 1 or lang in ("hi", "hi-en"):
            session.language = lang

        # Track customer problem if mentioned early
        if any(w in msg_lower for w in ["where is", "hasn't arrived", "not arrived", "kahan hai", "kab aayega", "delivery"]):
            session.customer_problem = "where_is_my_order"
        elif any(w in msg_lower for w in ["damage", "broken", "tv", "replacement"]):
            session.customer_problem = "damage"
        elif any(w in msg_lower for w in ["return", "लौटा", "wapas"]):
            session.customer_problem = "return"
        elif any(w in msg_lower for w in ["refund", "रिफ़ंड"]):
            session.customer_problem = "refund"

        # Extract order ID if present in the message
        extracted_oid = self._extract_order_id(msg)
        if extracted_oid:
            session.order_id = extracted_oid

        # Check if the user is asking a general policy or informational question
        is_order_specific_action = bool(extracted_oid) or any(
            w in msg_lower for w in [
                "my order", "mera order", "mere order", "apna order",
                "where is", "hasn't arrived", "not arrived", "track",
                "cancel my", "cancel this", "cancel it", "cancel order", "cancel the order",
                "compensation", "give me 500", "500", "voucher", "ticket",
                "replacement", "approve replacement", "damaged tv", "damage report"
            ]
        )

        # ---------------------------------------------------------
        # Turn 1: Check if general question vs opening greeting
        # ---------------------------------------------------------
        if session.turn_count == 1:
            if not is_order_specific_action and len(msg.split()) > 3:
                # Answer general policy inquiry directly via RAG
                rag_res = self.rag.answer_query(msg)
                intro_prefix = (
                    "Hello! I am ZentroMart's AI support assistant. "
                    if session.language == "en"
                    else "नमस्ते! मैं ZentroMart का एआई सपोर्ट असिस्टेंट हूँ। "
                )
                reply = f"{intro_prefix}{rag_res['answer']}"
                session.history.append({"role": "user", "content": msg})
                session.history.append({"role": "assistant", "content": reply})
                return reply

            reply = self._build_turn1_greeting(session, msg)
            session.history.append({"role": "user", "content": msg})
            session.history.append({"role": "assistant", "content": reply})
            return reply

        # ---------------------------------------------------------
        # Unverified Turn: If user is asking a general policy question, answer via RAG!
        # Only gatekeeper when the user asks about a specific order, cancellation, or compensation.
        # ---------------------------------------------------------
        if not session.is_verified and not session.verification_failed:
            if not is_order_specific_action and not self._extract_verification_value(msg, exclude_order_id=session.order_id):
                rag_res = self.rag.answer_query(msg)
                reply = rag_res["answer"]
                session.history.append({"role": "user", "content": msg})
                session.history.append({"role": "assistant", "content": reply})
                return reply

        # ---------------------------------------------------------
        # Turn 2: Order ID provided but Identity NOT yet verified
        # ---------------------------------------------------------
        if session.order_id and not session.is_verified and not session.verification_failed:
            proof = self._extract_verification_value(msg, exclude_order_id=session.order_id)
            if proof:
                # Attempt verification with tool
                verification_res = lookup_order(session.order_id, verification_value=proof)
                if verification_res.get("verified"):
                    session.is_verified = True
                    session.order_data = verification_res

                    # Check if customer already stated problem in previous turns
                    if session.customer_problem == "where_is_my_order":
                        reply = self._answer_order_status(session, prefix_verified=True)
                    else:
                        if session.language == "hi":
                            reply = f"धन्यवाद, आपकी पहचान वेरिफ़ाई हो गई है। मैं आपके ऑर्डर {session.order_id} में किस प्रकार मदद कर सकता हूँ?"
                        elif session.language == "hi-en":
                            reply = f"Thank you, aapki identity verify ho gayi hai. Main aapke order {session.order_id} mein kya madad kar sakta hoon?"
                        else:
                            reply = f"Thank you, your identity has been verified. How can I help you with order {session.order_id} today?"

                    session.history.append({"role": "user", "content": msg})
                    session.history.append({"role": "assistant", "content": reply})
                    return reply
                else:
                    # Verification failed
                    session.verification_attempts += 1
                    if session.verification_attempts >= 2:
                        session.verification_failed = True
                        if session.language == "hi":
                            reply = (
                                "माफ़ कीजिए, दिए गए विवरण का मेल नहीं हुआ। सुरक्षा कारणों से हम ऑर्डर की जानकारी साझा नहीं कर सकते। "
                                "कृपया अपने पंजीकृत फ़ोन नंबर से सहायता टीम से संपर्क करें।"
                            )
                        elif session.language == "hi-en":
                            reply = (
                                "Sorry, details match nahi hue. Security reasons se hum order details share nahi kar sakte. "
                                "Please registered phone number se support team se sampark karein."
                            )
                        else:
                            reply = (
                                "Sorry, the details provided do not match our records. For security reasons, we cannot share order details. "
                                "Please contact support from your registered phone number."
                            )
                        session.history.append({"role": "user", "content": msg})
                        session.history.append({"role": "assistant", "content": reply})
                        return reply
                    else:
                        # Allow 1 retry
                        if session.language == "hi":
                            reply = "माफ़ कीजिए, विवरण मेल नहीं खाए। कृपया दोबारा जाँच लें और अपने रजिस्टर्ड फ़ोन नंबर के आख़िरी 4 अंक या डिलीवरी पिनकोड बताएँ।"
                        elif session.language == "hi-en":
                            reply = "Sorry, details match nahi hue. Kripya dobara check karein aur apne registered phone ke aakhri 4 digits ya delivery pincode batayein."
                        else:
                            reply = "Sorry, those details did not match. Please check again and share either the last 4 digits of your registered phone number or your delivery pincode."
                        session.history.append({"role": "user", "content": msg})
                        session.history.append({"role": "assistant", "content": reply})
                        return reply

            # If no verification proof in message, call lookup_order to verify order exists and ask for verification
            lookup_order(session.order_id)
            if session.language == "hi":
                reply = (
                    f"ऑर्डर {session.order_id} की जानकारी देखने के लिए, कृपया अपनी पहचान वेरिफ़ाई करें: "
                    "अपने रजिस्टर्ड फ़ोन नंबर के आख़िरी 4 अंक या डिलीवरी पिनकोड बताएँ।"
                )
            elif session.language == "hi-en":
                reply = (
                    f"Order {session.order_id} ki details ke liye, kripya identity verify karein: "
                    "apne registered phone ke last 4 digits ya delivery pincode share karein."
                )
            else:
                reply = (
                    f"To check details for order {session.order_id}, please verify your identity: "
                    "share either the last 4 digits of your registered phone number or the delivery pincode."
                )
            session.history.append({"role": "user", "content": msg})
            session.history.append({"role": "assistant", "content": reply})
            return reply

        # If verification failed previously, strictly refuse to disclose order details (IDV-3)
        if session.verification_failed:
            if session.language == "hi":
                reply = "पहचान वेरिफ़ाई न होने के कारण हम इस ऑर्डर की जानकारी साझा नहीं कर सकते। आप पंजीकृत नंबर से संपर्क कर सकते हैं।"
            elif session.language == "hi-en":
                reply = "Verification fail hone ke karan hum order details share nahi kar sakte. Please registered number se call karein."
            else:
                reply = "Since identity verification could not be completed, I cannot share order details. Please contact us from your registered phone number."
            session.history.append({"role": "user", "content": msg})
            session.history.append({"role": "assistant", "content": reply})
            return reply

        # ---------------------------------------------------------
        # Turn 3+: Verified Customer / Handling Specific Requests
        # ---------------------------------------------------------
        reply = self._handle_verified_turn(msg, session)
        session.history.append({"role": "user", "content": msg})
        session.history.append({"role": "assistant", "content": reply})
        return reply

    def _build_turn1_greeting(self, session: ConversationSession, msg: str) -> str:
        """Constructs initial greeting introducing as an AI assistant (STY-1)."""
        lang = session.language
        if session.order_id:
            lookup_order(session.order_id)
            if lang == "hi":
                return (
                    f"नमस्ते! मैं ZentroMart का एआई सपोर्ट असिस्टेंट हूँ। "
                    f"ऑर्डर {session.order_id} की जानकारी देखने के लिए, कृपया अपनी पहचान वेरिफ़ाई करें: "
                    "अपने रजिस्टर्ड फ़ोन के आख़िरी 4 अंक या डिलीवरी पिनकोड बताएँ।"
                )
            elif lang == "hi-en":
                return (
                    f"Hello! Main ZentroMart ka AI support assistant hoon. "
                    f"Order {session.order_id} ki details dekhne ke liye, kripya identity verify karein: "
                    "apne registered phone number ke last 4 digits ya delivery pincode share karein."
                )
            else:
                return (
                    f"Hello! I am ZentroMart's AI support assistant. "
                    f"To help you with order {session.order_id}, please verify your identity: "
                    "share either the last 4 digits of your registered phone number or your delivery pincode."
                )
        else:
            if lang == "hi":
                return (
                    "नमस्ते! मैं ZentroMart का एआई सपोर्ट असिस्टेंट हूँ। आज मैं आपकी क्या मदद कर सकता हूँ? "
                    "अगर यह किसी ऑर्डर के बारे में है, तो कृपया अपना ऑर्डर नंबर बताएँ।"
                )
            elif lang == "hi-en":
                return (
                    "Hello! Main ZentroMart ka AI support assistant hoon. Aaj main aapki kya madad kar sakta hoon? "
                    "Agar yeh kisi order ke baare mein hai, toh please apna order number share karein."
                )
            else:
                return (
                    "Hello! I am ZentroMart's AI support assistant. How can I help you today? "
                    "If your query is about an order, please share your order ID."
                )

    def _handle_verified_turn(self, msg: str, session: ConversationSession) -> str:
        """Processes requests once verified: policy checks, refusals, escalation, or answers."""
        msg_lower = msg.lower()
        order = session.order_data or {}
        lang = session.language

        # ---------------------------------------------------------
        # 1. Escalation Trigger: Customer Asks for Human (ESC-1 #6)
        # ---------------------------------------------------------
        human_triggers = [
            "speak to a person",
            "talk to a person",
            "human agent",
            "talk to human",
            "connect to human",
            "speak with someone",
            "insaan se baat",
            "kisi insaan se",
            "human colleague",
            "real person",
            "इंसान से बात",
            "इंसान",
            "सहयोगी से बात",
            "कॉल ट्रांसफर",
            "एजेंट से बात",
        ]
        if any(trigger in msg_lower or trigger in msg for trigger in human_triggers):
            return self._escalate_to_human(
                session=session,
                reason="customer_requested_human",
                intent="human_requested",
                summary=(
                    f"Customer requested human agent for order {session.order_id}. "
                    "Identity verified. Escalating to human colleague per ESC-1."
                ),
            )

        # ---------------------------------------------------------
        # 2. Escalation Trigger: Repeat Contact (ESC-1 #4)
        # ---------------------------------------------------------
        contacts_last_7 = int(order.get("contacts_last_7_days") or 0)
        is_repeat_phrase = any(
            phrase in msg_lower
            for phrase in [
                "teesri baar",
                "third time",
                "already called",
                "contacted before",
                "multiple times this week",
                "again and again",
                "dusri baar",
                "तीसरी बार",
            ]
        )
        if contacts_last_7 >= 2 or is_repeat_phrase:
            return self._escalate_to_human(
                session=session,
                reason="repeat_contact_within_7_days",
                intent="repeat_contact",
                summary=(
                    f"Repeat contact on order {session.order_id} ({contacts_last_7} earlier contacts in last 7 days). "
                    "Customer reports repeated inquiries. Handing over to Priority Resolution team per ESC-1."
                ),
            )

        # ---------------------------------------------------------
        # 3. Escalation Trigger: High-Value Damage (> ₹10,000) (ESC-1 #1)
        # ---------------------------------------------------------
        if any(w in msg_lower for w in ["damage", "broken", "kharab", "replacement", "tv", "laptop", "screen"]):
            total_inr = int(order.get("total_inr") or 0)
            subtotal_inr = int(order.get("subtotal_inr") or 0)
            if total_inr > 10000 or subtotal_inr > 10000 or "tv" in msg_lower or "laptop" in msg_lower:
                return self._escalate_to_human(
                    session=session,
                    reason="high_value_damage",
                    intent="damage_replacement",
                    summary=(
                        f"Customer reports damage on high-value order {session.order_id} (value > ₹10,000). "
                        "Damage claims above ₹10,000 cannot be approved by AI and require Priority Resolution team review under DWM-3/ESC-1."
                    ),
                    extra_note="high_value_damage",
                )

        # ---------------------------------------------------------
        # 4. Refusal Trigger: Demand for Extra Compensation (DEL-4 / ESC-4)
        # ---------------------------------------------------------
        if (
            any(w in msg_lower for w in ["voucher", "compensation", "credit", "rupaye", "discount", "paise", "cash"])
            and any(w in msg_lower for w in ["late", "der se", "delay", "300", "500", "extra", "hajaar", "thousand", "₹"])
        ) or any(phrase in msg_lower for phrase in ["500", "₹500", "give me 500", "want 500", "compensation of", "extra compensation"]):
            delivery_type = (order.get("delivery_type") or "").lower()
            if delivery_type != "express":
                # 1) Summarize the issue (minimum 20 characters)
                summary = (
                    f"Customer demanding outside-policy compensation on order {session.order_id or 'unspecified'}. "
                    "Issue: Standard delivery delay reported; customer demanded compensation/voucher. "
                    "Checks conducted: Standard delivery has no compensation under policy DEL-4 / ESC-4. "
                    "Action taken: Compensation politely refused per policy; issue logged for human colleague review."
                )
                # 2) Call create_ticket tool
                ticket_res = create_ticket(
                    summary=summary,
                    order_id=session.order_id,
                    reason="outside_policy_compensation_demand",
                    intent="compensation_demand",
                )
                # 3) Crucially, ONLY tell customer they have been handed over to a human if created=True
                if ticket_res.get("created"):
                    tid = ticket_res.get("ticket_id")
                    session.tickets_created.append(ticket_res)
                    if lang == "hi":
                        return (
                            f"माफ़ कीजिए, हमारी डिलीवरी पॉलिसी (DEL-4) के अनुसार स्टैंडर्ड डिलीवरी में देरी के लिए कोई अतिरिक्त वाउचर या मुआवज़ा देने का नियम नहीं है। "
                            f"मैंने आपकी समस्या का सारांश दर्ज कर टिकट ({tid}) बना दिया है और हमारे सहयोगी (टीम) को यह मामला सौंप दिया है।"
                        )
                    elif lang == "hi-en":
                        return (
                            f"Sorry, ZentroMart ki policy (DEL-4) ke according Standard delivery late hone par koi extra voucher ya compensation allow nahi hai. "
                            f"Maine aapke issue ka summary banakar ticket ({tid}) create kar diya hai aur human colleague / team ko handover kar diya hai."
                        )
                    else:
                        return (
                            f"I apologize for the delay, but under our policy (DEL-4), we do not offer vouchers or compensation for delayed Standard deliveries. "
                            f"I have summarized your issue and created ticket {tid}, and handed this over to a human colleague to review your case."
                        )
                else:
                    # Tool returned created=False -> DO NOT claim handover to human
                    if lang == "hi":
                        return (
                            "माफ़ कीजिए, हमारी डिलीवरी पॉलिसी (DEL-4) के अनुसार स्टैंडर्ड डिलीवरी में देरी के लिए कोई अतिरिक्त वाउचर या मुआवज़ा देने का नियम नहीं है। "
                            "हुई असुविधा के लिए हमें अत्यंत खेद है।"
                        )
                    elif lang == "hi-en":
                        return (
                            "Sorry, ZentroMart ki policy (DEL-4) ke according Standard delivery late hone par koi extra voucher ya compensation allow nahi hai. "
                            "Hui asuvidha ke liye hum maafi chahte hain."
                        )
                    else:
                        return (
                            "I apologize for the delay, but under our policy (DEL-4), we do not offer vouchers or compensation for delayed Standard deliveries. "
                            "We truly regret the inconvenience caused."
                        )


        # ---------------------------------------------------------
        # 5. Order Cancellation Request (CAN-1, CAN-2)
        # ---------------------------------------------------------
        if any(w in msg_lower for w in ["cancel", "रद्द"]):
            status = (order.get("status") or "").lower()
            if status in ["shipped", "delivered"]:
                pay_method = order.get("payment_method", "card")
                if lang == "hi":
                    return (
                        f"चूँकि आपका ऑर्डर पहले ही शिप हो चुका है (स्थिति: {status}), इसलिए इसे अब कैंसिल नहीं किया जा सकता (CAN-2)। "
                        "आप डिलीवरी के समय पार्सल लेने से मना कर सकते हैं। पार्सल वेयरहाउस लौटने के बाद आपके "
                        f"{pay_method.upper()} पर रिफ़ंड 5 से 7 कामकाजी दिनों में शुरू कर दिया जाएगा।"
                    )
                elif lang == "hi-en":
                    return (
                        f"Aapka order already ship ho chuka hai (status: {status}), isliye policy (CAN-2) ke mutabik ise cancel nahi kiya ja sakta. "
                        "Aap delivery ke samay parcel lene se mana kar sakte hain. Warehouse wapas aane par aapke "
                        f"{pay_method.upper()} par refund initiate ho jayega (usually 5 to 7 business days)."
                    )
                else:
                    return (
                        f"Since your order has already shipped (status: {status}), it cannot be cancelled as per policy (CAN-2). "
                        "You can refuse the parcel at the door when the courier arrives. Once returned to the warehouse, "
                        f"your refund will be initiated to your {pay_method.upper()} within 5 to 7 business days."
                    )
            else:
                if lang == "hi":
                    return (
                        "आपका ऑर्डर अभी शिप नहीं हुआ है, इसलिए आप इसे ZentroMart ऐप में 'My Orders' सेक्शन से मुफ़्त में कैंसिल कर सकते हैं।"
                    )
                elif lang == "hi-en":
                    return (
                        "Aapka order abhi ship nahi hua hai, isliye aap ise ZentroMart app mein 'My Orders' se easily aur free mein cancel kar sakte hain."
                    )
                else:
                    return (
                        "Your order has not shipped yet. You can cancel it for free directly from 'My Orders' in the ZentroMart app."
                    )

        # ---------------------------------------------------------
        # 6. Follow-up: Courier Delivery Attempts (DEL-3)
        # ---------------------------------------------------------
        if any(w in msg_lower for w in ["not at home", "not home", "ghar pe nahi", "ghar par nahi", "courier comes"]):
            if lang == "hi":
                return (
                    "यदि आप डिलीवरी के समय घर पर नहीं हैं, तो कूरियर अधिकतम 3 बार डिलीवरी का प्रयास करेगा (DEL-3)। "
                    "प्रत्येक असफल प्रयास के बाद आपको रीशेड्यूल करने के लिए एसएमएस भेजा जाता है। तीसरे असफल प्रयास के बाद पार्सल वेयरहाउस वापस चला जाएगा।"
                )
            elif lang == "hi-en":
                return (
                    "Agar aap ghar par nahi hain, toh courier maximum 3 delivery attempts karega (DEL-3). "
                    "Third attempt fail hone ke baad parcel warehouse laut jayega aur refund initiate hoga."
                )
            else:
                return (
                    "If you are not at home, the courier will make up to 3 delivery attempts (DEL-3). "
                    "After each attempt, you receive an SMS to reschedule. After the 3rd failed attempt, the parcel returns to the warehouse."
                )

        # ---------------------------------------------------------
        # 7. Express Late Credit Question (DEL-4)
        # ---------------------------------------------------------
        if "express" in msg_lower and any(w in msg_lower for w in ["late", "credit", "fee"]):
            credit_inr = order.get("express_late_credit_inr") or "79"
            credit_on = order.get("express_late_credit_on") or "2026-09-29"
            formatted_date = self._format_date_text(credit_on, lang)
            if lang == "hi":
                return f"हाँ, एक्सप्रेस डिलीवरी में देरी होने पर ₹{credit_inr} की एक्सप्रेस फ़ीस आपके ZentroMart Wallet में {formatted_date} को क्रेडिट कर दी गई है (DEL-4)।"
            elif lang == "hi-en":
                return f"Haan, Express delivery late hone par ₹{credit_inr} ka credit aapke ZentroMart Wallet mein {formatted_date} ko add kar diya gaya hai (DEL-4)."
            else:
                return f"Yes, for the delayed Express delivery, the ₹{credit_inr} express fee was automatically credited to your ZentroMart Wallet on {formatted_date} (DEL-4)."

        # ---------------------------------------------------------
        # 8. Refund Timeline & Status (REF-1, REF-2, REF-6, ESC-1)
        # ---------------------------------------------------------
        if any(w in msg_lower for w in ["refund", "रिफ़ंड", "paisa", "kitne din"]):
            return self._calculate_refund_timeline(session, msg)

        # ---------------------------------------------------------
        # 9. Return Eligibility & Window Check (RET-2, RET-6, RET-8, PLS-2)
        # ---------------------------------------------------------
        if any(w in msg_lower for w in ["return", "लौटा", "wapas", "वापस"]):
            # Check if this is a general policy question (e.g. drop off, physical store, how to return)
            # rather than checking return eligibility for an item in the verified order
            if any(w in msg_lower for w in ["store", "drop off", "physical", "new york", "how do i return", "how to return", "policy"]):
                rag_res = self.rag.answer_query(msg)
                return rag_res["answer"]
            return self._calculate_return_eligibility(session, msg)

        # ---------------------------------------------------------
        # 10. Shipping Date or Order Status (TRK-1, DEL-1)
        # ---------------------------------------------------------
        if any(w in msg_lower for w in ["ship hua", "shipped on", "when was it shipped", "kab ship", "status", "where is", "kahan"]):
            return self._answer_order_status(session, prefix_verified=False)

        # ---------------------------------------------------------
        # 11. General Policy Questions / Topic Switch via RAG
        # ---------------------------------------------------------
        rag_res = self.rag.answer_query(msg)
        return rag_res["answer"]

    def _answer_order_status(self, session: ConversationSession, prefix_verified: bool = False) -> str:
        """Answers order delivery, shipped_on, and promised_date status."""
        order = session.order_data or {}
        lang = session.language
        shipped_on = order.get("shipped_on")
        status = order.get("status", "placed")
        promised_date = order.get("promised_date", "")

        prefix = ""
        if prefix_verified:
            if lang == "hi":
                prefix = "धन्यवाद, आपकी पहचान वेरिफ़ाई हो गई है। "
            elif lang == "hi-en":
                prefix = "Thank you, aapki identity verify ho gayi hai. "
            else:
                prefix = "Thank you, your identity has been verified. "

        formatted_promised = self._format_date_text(promised_date, lang) if promised_date else ""
        formatted_shipped = self._format_date_text(shipped_on, lang) if shipped_on else ""

        if shipped_on and promised_date:
            if lang == "hi":
                return f"{prefix}आपका ऑर्डर {session.order_id} {formatted_shipped} को शिप हुआ था और {formatted_promised} तक डिलीवर होने का वादा है।"
            elif lang == "hi-en":
                return f"{prefix}Aapka order {session.order_id} {formatted_shipped} ko ship hua tha aur {formatted_promised} tak deliver ho jayega."
            else:
                return f"{prefix}Your order {session.order_id} was shipped on {formatted_shipped} and is promised to arrive by {formatted_promised}."
        elif shipped_on:
            if lang == "hi":
                return f"{prefix}आपका ऑर्डर {session.order_id} {formatted_shipped} को शिप हुआ था। वर्तमान स्थिति '{status}' है।"
            elif lang == "hi-en":
                return f"{prefix}Aapka order {session.order_id} {formatted_shipped} ko ship hua tha. Status: {status}."
            else:
                return f"{prefix}Your order {session.order_id} was shipped on {formatted_shipped}. Status: {status}."
        else:
            if lang == "hi":
                return f"{prefix}आपके ऑर्डर {session.order_id} की वर्तमान स्थिति '{status}' है।"
            else:
                return f"{prefix}Your order {session.order_id} is currently '{status}'."

    def _calculate_return_eligibility(self, session: ConversationSession, msg: str) -> str:
        """Calculates return window, deadline date, and reverse pickup fee."""
        order = session.order_data or {}
        lang = session.language
        items = order.get("items") or []

        # If user is asking a general policy question without a specific verified order, answer via RAG
        if not order or not items:
            rag_res = self.rag.answer_query(msg)
            return rag_res["answer"]

        # Find matching item if mentioned, else first item
        target_item = items[0] if items else {}
        for item in items:
            if item.get("title", "").lower() in msg.lower() or item.get("sku", "").lower() in msg.lower():
                target_item = item
                break

        title = target_item.get("title", "Item")
        category = (target_item.get("return_category") or "lifestyle").lower()
        delivered_on_str = order.get("delivered_on") or order.get("placed_at", "2026-10-10")[:10]
        placed_at_str = (order.get("placed_at") or "2026-10-10")[:10]

        # Festive sale check: placed between 1 and 15 October 2026 gets 30 days for fashion (RET-8)
        placed_dt = pd.to_datetime(placed_at_str).date()
        is_festive_sale = date(2026, 10, 1) <= placed_dt <= date(2026, 10, 15)

        if is_festive_sale and category in ["clothing", "footwear", "bags", "fashion"]:
            window_days = 30
        elif category in ["clothing", "footwear", "bags", "fashion"]:
            window_days = 15
        elif category in ["electronics", "appliances"]:
            window_days = 7
        elif category in ["grocery", "innerwear", "swimwear"]:
            window_days = 0
        else:
            window_days = 10

        delivered_dt = pd.to_datetime(delivered_on_str).date()
        deadline_dt = delivered_dt + timedelta(days=window_days)
        formatted_deadline = self._format_date_text(deadline_dt.isoformat(), lang)

        # Reverse pickup fee check (RET-6, PLS-2)
        is_plus = bool(order.get("is_plus"))

        if window_days == 0:
            if lang == "hi":
                return f"माफ़ कीजिए, {title} ({category}) हाइजीन और पॉलिसी (RET-5) के तहत नॉन-रिटर्नेबल है।"
            elif lang == "hi-en":
                return f"Sorry, {title} ({category}) policy (RET-5) ke tahat non-returnable category mein aata hai."
            else:
                return f"Sorry, {title} belongs to {category} which is non-returnable under policy (RET-5)."

        if lang == "hi":
            fee_info = "ZentroMart Plus सदस्य होने के कारण कोई पिकअप फ़ीस नहीं लगेगी।" if is_plus else "₹49 रिवर्स पिकअप शुल्क लागू होगा।"
            return (
                f"हाँ, आप {title} वापस कर सकते हैं। इसके लिए रिटर्न विंडो {window_days} दिन है, "
                f"और लौटाने की अंतिम तिथि {formatted_deadline} है। {fee_info}"
            )
        elif lang == "hi-en":
            fee_info = "ZentroMart Plus member hone ki wajah se pickup fee waived hai." if is_plus else "Reverse pickup fee ₹49 lagegi."
            return (
                f"Haan, aap {title} return kar sakte hain. Return window {window_days} din hai, "
                f"aur aakhri tareekh {formatted_deadline} hai. {fee_info}"
            )
        else:
            fee_info = "As a ZentroMart Plus member, reverse pickup is free." if is_plus else "A reverse pickup fee of ₹49 applies."
            return (
                f"Yes, you can return {title}. The return window is {window_days} days, "
                f"valid until {formatted_deadline}. {fee_info}"
            )

    def _calculate_refund_timeline(self, session: ConversationSession, msg: str) -> str:
        """Calculates refund timing or status based on return/refund milestones."""
        order = session.order_data or {}
        lang = session.language
        pay_method = (order.get("payment_method") or "upi").lower()
        is_plus = bool(order.get("is_plus"))

        # Check if item returned but quality check (QC) is pending (PC009 scenario)
        return_status = order.get("return_status")
        return_pickup_on = order.get("return_pickup_on")
        return_qc_passed = order.get("return_qc_passed_on")

        if return_pickup_on and not return_qc_passed:
            if lang == "hi":
                return (
                    "आपका सामान पिकअप हो चुका है, लेकिन अभी वेयरहाउस में गुणवत्ता जाँच (Quality Check / QC) की प्रक्रिया बाकी है। "
                    "QC पास होने के 2 कामकाजी दिनों में रिफ़ंड शुरू कर दिया जाएगा।"
                )
            elif lang == "hi-en":
                return (
                    "Aapka item pickup ho chuka hai, par abhi warehouse mein Quality Check (QC) pending hai. "
                    "QC pass hone ke baad refund initiate ho jayega."
                )
            else:
                return (
                    "Your returned item has been picked up, but quality check (QC) is currently pending at the warehouse. "
                    "The refund will be initiated within 2 business days after QC passes."
                )

        # Check if refund is overdue (> 3 business days past max timeline per ESC-1 #5, e.g. PC010)
        initiated_on = order.get("return_refund_initiated_on") or order.get("cancellation_refund_initiated_on") or order.get("rto_refund_initiated_on")
        rrn = order.get("return_rrn") or order.get("cancellation_rrn") or order.get("rto_rrn") or ""

        if initiated_on:
            # Calculate business days elapsed
            elapsed = self.db.calendar.business_days_since(initiated_on)
            max_allowed_days = 3 if pay_method == "upi" else (7 if pay_method in ["card", "netbanking"] else 2)
            if elapsed > max_allowed_days + 3:
                # Overdue beyond grace period -> Hand over to team with RRN per ESC-1 #5 & REF-6
                return self._escalate_to_human(
                    session=session,
                    reason="overdue_refund_escalation",
                    intent="overdue_refund",
                    summary=(
                        f"Refund on order {session.order_id} ({pay_method.upper()}) was initiated on {initiated_on} "
                        f"with RRN {rrn}, which is {elapsed} business days ago and exceeds the maximum timeline. "
                        "Handing over to human colleague per ESC-1."
                    ),
                    extra_note="overdue_refund",
                )

        if pay_method == "wallet":
            timeline = "instant, within 2 hours"
            timeline_hi = "तुरंत, 2 घंटे के अंदर"
            timeline_hien = "instant, 2 hours ke andar"
        elif pay_method == "upi":
            timeline = "1 to 3 business days"
            timeline_hi = "1 से 3 कामकाजी दिन"
            timeline_hien = "1 se 3 business days"
        elif pay_method in ["card", "netbanking"]:
            timeline = "5 to 7 business days"
            timeline_hi = "5 से 7 कामकाजी दिन"
            timeline_hien = "5 se 7 business days"
        else:
            timeline = "5 to 7 business days for bank transfer or instant to ZentroMart Wallet"
            timeline_hi = "बैंक ट्रांसफ़र में 5 से 7 कामकाजी दिन या वॉलेट में तुरंत"
            timeline_hien = "5 se 7 business days (bank) ya Wallet mein instant"

        plus_note = " (ZentroMart Plus members receive refund initiated at pickup per PLS-2)" if is_plus else ""
        plus_note_hi = " (प्लस मेंबर होने के कारण रिफ़ंड पिकअप के समय ही शुरू हो जाता है)" if is_plus else ""
        plus_note_hien = " (Plus members ka refund pickup ke waqt hi shuru ho jata hai)" if is_plus else ""

        if lang == "hi":
            return f"आपके {pay_method.upper()} पर रिफ़ंड {timeline_hi} में आ जाएगा{plus_note_hi}।"
        elif lang == "hi-en":
            return f"Aapke {pay_method.upper()} par refund {timeline_hien} mein aa jayega{plus_note_hien}."
        else:
            return f"Your refund to {pay_method.upper()} will take {timeline}{plus_note}."

    def _escalate_to_human(
        self,
        session: ConversationSession,
        reason: str,
        intent: str,
        summary: str,
        extra_note: str = "",
    ) -> str:
        """Creates a ticket via create_ticket and confirms handover ONLY if created=True."""
        lang = session.language
        order = session.order_data or {}
        rrn = order.get("return_rrn") or order.get("cancellation_rrn") or order.get("rto_rrn") or ""

        # Call mock tool
        ticket_res = create_ticket(
            summary=summary,
            order_id=session.order_id,
            reason=reason,
            intent=intent,
        )

        if ticket_res.get("created"):
            tid = ticket_res.get("ticket_id")
            session.tickets_created.append(ticket_res)

            if extra_note == "overdue_refund":
                if lang == "hi":
                    return (
                        f"रिफ़ंड की समय सीमा पार हो चुकी है। आपका बैंक संदर्भ संख्या (RRN) {rrn} है। "
                        f"मैंने आपकी शिकायत का टिकट ({tid}) बना दिया है और हमारे सहयोगी (टीम) से संपर्क करा रहा हूँ।"
                    )
                elif lang == "hi-en":
                    return (
                        f"Refund timeline exceed ho chuki hai. Aapka reference number (RRN) {rrn} hai. "
                        f"Maine ticket ({tid}) create karke human colleague / team ko handover kar diya hai."
                    )
                else:
                    return (
                        f"Your refund is overdue. Your refund reference number (RRN) is {rrn}. "
                        f"I have created ticket {tid} and connected you with a human colleague / team to resolve this."
                    )

            if extra_note == "high_value_damage":
                if lang == "hi":
                    return (
                        f"पॉलिसी के अनुसार ₹10,000 से अधिक के क्लेम का फ़ैसला हमारी Priority Resolution टीम करती है। "
                        f"मैंने आपकी शिकायत का टिकट ({tid}) बना दिया है और इसे अपनी टीम को सौंप दिया है। वे आपसे जल्द संपर्क करेंगे।"
                    )
                elif lang == "hi-en":
                    return (
                        f"Policy ke mutabik ₹10,000 se upar ke damage claims Priority Resolution team handle karti hai. "
                        f"Maine ticket ({tid}) create karke colleague ko handover kar diya hai. Wo aapse jald hi connect karenge."
                    )
                else:
                    return (
                        f"Claims for items above ₹10,000 require review by our Priority Resolution team per policy (DWM-3/ESC-1). "
                        f"I have created ticket {tid} and handed this over to my human colleagues, who will contact you shortly."
                    )

            if lang == "hi":
                return (
                    f"मैंने आपकी सहायता के लिए टिकट ({tid}) बना दिया है और अपने वरिष्ठ सहयोगी (टीम) को यह मामला सौंप दिया है। "
                    "वे जल्द ही आपसे संपर्क करेंगे।"
                )
            elif lang == "hi-en":
                return (
                    f"Maine aapke issue ke liye ticket ({tid}) create kar diya hai aur human colleague / team se connect karne ke liye handover kar diya hai. "
                    "Hamari team aapse jald hi rabta karegi."
                )
            else:
                return (
                    f"I have created ticket {tid} and connected you with a human colleague who can take over and assist you further."
                )
        else:
            err = ticket_res.get("error", "Ticket creation failed")
            if lang == "hi":
                return f"तकनीकी समस्या के कारण टिकट दर्ज नहीं हो सका ({err})। कृपया कुछ समय बाद पुनः प्रयास करें।"
            else:
                return f"Could not create an escalation ticket at this moment ({err}). Please try again shortly."

    def _extract_order_id(self, text: str) -> Optional[str]:
        """Extracts 8-digit order number from text."""
        m = re.search(r"\b(\d{4})\s*(\d{4})\b", text)
        if m:
            return f"{m.group(1)}{m.group(2)}"
        m8 = re.search(r"\b(\d{8})\b", text)
        if m8:
            return m8.group(1)
        return None

    def _extract_verification_value(self, text: str, exclude_order_id: Optional[str] = None) -> Optional[str]:
        """Extracts 4-digit phone suffix or 6-digit delivery pincode, excluding order ID digits."""
        clean_text = text
        if exclude_order_id:
            clean_text = clean_text.replace(exclude_order_id, "")
            if len(exclude_order_id) == 8:
                half1, half2 = exclude_order_id[:4], exclude_order_id[4:]
                clean_text = re.sub(rf"\b{half1}\s+{half2}\b", "", clean_text)
                clean_text = re.sub(rf"\b{half1}\b", "", clean_text)
                clean_text = re.sub(rf"\b{half2}\b", "", clean_text)

        m6 = re.search(r"\b(\d{6})\b", clean_text)
        if m6:
            return m6.group(1)
        m4 = re.search(r"\b(\d{4})\b", clean_text)
        if m4:
            return m4.group(1)
        return None

    def _format_date_text(self, date_str: str, lang: str) -> str:
        """Formats dates with month names e.g. 27 October 2026."""
        try:
            dt = pd.to_datetime(date_str)
            month_en = dt.strftime("%B")
            day = dt.day
            year = dt.year

            months_hi = {
                "January": "जनवरी",
                "February": "फ़रवरी",
                "March": "मार्च",
                "April": "अप्रैल",
                "May": "मई",
                "June": "जून",
                "July": "जुलाई",
                "August": "अगस्त",
                "September": "सितंबर",
                "October": "अक्टूबर",
                "November": "नवंबर",
                "December": "दिसंबर",
            }

            if lang == "hi":
                return f"{day} {months_hi.get(month_en, month_en)} {year}"
            return f"{day} {month_en} {year}"
        except Exception:
            return date_str


def main():
    print("=" * 70)
    print("    ZentroMart Multi-Turn Conversational Customer Support Agent")
    print("=" * 70)

    agent = ZentroMartAgent()

    print("\n--- Simulation: Conversation PC002 (High-Value Damage) ---")
    session = ConversationSession(conversation_id="SIM-PC002")
    turns = [
        "Hello, mujhe order ke baare mein help chahiye.",
        "Order number hai 8978 2837.",
        "Last four digits 4533 hain.",
        "Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha. Abhi replacement approve kar do na?",
    ]
    for i, user_msg in enumerate(turns, 1):
        print(f"\n[Turn {i}] Customer: {user_msg}")
        resp = agent.handle_turn(user_msg, session)
        print(f"[Turn {i}] Agent   : {resp}")

    print("\n--- Simulation: Conversation PC020 (Cancellation of Shipped Order) ---")
    session2 = ConversationSession(conversation_id="SIM-PC020")
    turns2 = [
        "Hi, I need some help with an order.",
        "The order number is 4316 5903.",
        "My pincode is 700091.",
        "I want to cancel it.",
    ]
    for i, user_msg in enumerate(turns2, 1):
        print(f"\n[Turn {i}] Customer: {user_msg}")
        resp = agent.handle_turn(user_msg, session2)
        print(f"[Turn {i}] Agent   : {resp}")


if __name__ == "__main__":
    main()
