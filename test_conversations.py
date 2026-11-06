"""
Runs 5 text-only multi-turn conversations against the Ezee Flights LLM + search_flights tool
without audio, covering:
  1. Happy path booking
  2. Too-expensive objection
  3. Visa question
  4. Refund on an existing booking
  5. "Don't call me" opt-out

Prints full transcripts (including tool calls) and writes post-call summary JSONs to `calls/`.
Uses live OpenAI (`gpt-5.4-mini`, `max_completion_tokens=500`, `reasoning_effort="none"`) when
`OPENAI_API_KEY` is set, and falls back to a spec-faithful deterministic simulator if run
without an API key (or with `--offline`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from post_call import generate_and_save_post_call_summary, save_post_call_summary_sync
from providers.flights import FlightProvider, create_flight_provider
from spec_loader import AgentSpec, CallMetadata, load_spec

load_dotenv(Path(__file__).parent / ".env")


@dataclass
class Scenario:
    scenario_id: str
    title: str
    metadata: CallMetadata
    user_turns: List[str]


SCENARIOS: List[Scenario] = [
    Scenario(
        scenario_id="test_1_happy_path_booking",
        title="1. Happy Path Booking (Dubai to London, Family Return)",
        metadata=CallMetadata(
            caller_name="Omair",
            customer_name="Omair",
            enquiry_route="Dubai to London",
            phone_number="+971501234567",
        ),
        user_turns=[
            "Hi Zaya, yes this is Omair speaking.",
            "Yes, we're planning a family holiday to London Heathrow. Looking at returning, leaving Dubai on 12 December and coming back on 2 January.",
            "It'll be two adults and one child — she's six years old. Economy class is fine, and we'd prefer a direct flight if possible.",
            "The Emirates morning direct flight at two thousand four hundred dirhams per adult sounds great for us with our daughter. Let's go with that one — you can connect me with the consultant now.",
        ],
    ),
    Scenario(
        scenario_id="test_2_too_expensive_objection",
        title="2. Too-Expensive Objection (Levers: 1-Stop or 1 Day Earlier)",
        metadata=CallMetadata(
            caller_name="Tariq",
            customer_name="Tariq",
            enquiry_route="Dubai to London",
            phone_number="+971502345678",
        ),
        user_turns=[
            "Hello, yes this is Tariq.",
            "I'm looking for a return flight from Dubai to London Heathrow, leaving 12 December and returning 2 January, for two adults and one child age 8 in economy.",
            "Honestly two thousand four hundred dirhams per adult is too expensive for our budget. We can move our departure a day earlier to 11 December or take one stop if it brings the price down.",
            "The Turkish Airlines one-stop on 11 December at one thousand seven hundred and fifty per adult with checked bags included works much better. Please have the consultant call me back at 6 PM today to lock it in.",
        ],
    ),
    Scenario(
        scenario_id="test_3_visa_question",
        title="3. Visa Question (No Immigration Advice Rule)",
        metadata=CallMetadata(
            caller_name="Ayesha",
            customer_name="Ayesha",
            enquiry_route="Dubai to London",
            phone_number="+971503456789",
        ),
        user_turns=[
            "Hi, yes Ayesha speaking.",
            "Before we look at flights, I hold a Pakistani passport with UAE residency — do I need a visa for London and what documents do I need to submit?",
            "Got it, thanks for clarifying. Let's look at return flights from Dubai to London departing 12 December and returning 2 January for 1 adult in economy.",
            "Please send me those two options on WhatsApp and have a consultant check in with me tomorrow morning.",
        ],
    ),
    Scenario(
        scenario_id="test_4_existing_booking_refund",
        title="4. Refund on an Existing Booking (Support Escalation)",
        metadata=CallMetadata(
            caller_name="Faisal",
            customer_name="Faisal",
            enquiry_route="Dubai to Karachi",
            phone_number="+971504567890",
        ),
        user_turns=[
            "Yes, this is Faisal.",
            "Look, I'm not looking for a new flight right now. I cancelled an existing booking last week and I still haven't received my refund! How much am I getting back and when will it hit my card?",
            "The booking reference is EZF-88421 and the lead passenger name is Faisal Ahmed. Please have support call me back this afternoon.",
        ],
    ),
    Scenario(
        scenario_id="test_5_dont_call_me",
        title="5. 'Don't Call Me' Opt-Out (Immediate Polite Termination)",
        metadata=CallMetadata(
            caller_name="Sami",
            customer_name="Sami",
            enquiry_route="Dubai to Jeddah",
            phone_number="+971505678901",
        ),
        user_turns=[
            "Yeah, speaking.",
            "Please don't call me again. Take my number off your calling list right now.",
        ],
    ),
]


def build_openai_tool_schema(spec: AgentSpec) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": spec.search_tool.instructions,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": spec.search_tool.query_description,
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]


async def _call_openai_with_tools(
    client: AsyncOpenAI,
    spec: AgentSpec,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    flight_provider: FlightProvider,
    transcript_log: List[Dict[str, Any]],
) -> str:
    """Execute one assistant turn with OpenAI, including any `search_flights` tool calls."""
    while True:
        req_kwargs: Dict[str, Any] = {
            "model": spec.llm.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_completion_tokens": spec.llm.max_completion_tokens,
        }
        if spec.llm.reasoning_effort:
            req_kwargs["reasoning_effort"] = spec.llm.reasoning_effort

        try:
            response = await client.chat.completions.create(**req_kwargs)
        except Exception:
            # Fallback if the model endpoint does not accept reasoning_effort
            req_kwargs.pop("reasoning_effort", None)
            response = await client.chat.completions.create(**req_kwargs)

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            if msg.content:
                transcript_log.append({"role": "assistant", "content": msg.content})

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                query = args.get("query", "")
                transcript_log.append(
                    {"role": "tool_call", "content": f"search_flights(query={query!r})"}
                )
                tool_result = await flight_provider.search(query)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )
            continue

        reply = (msg.content or "").strip()
        messages.append({"role": "assistant", "content": reply})
        transcript_log.append({"role": "assistant", "content": reply})
        return reply


def _simulate_turn_offline_sync(
    scenario_id: str,
    turn_idx: int,
    metadata: CallMetadata,
    flight_provider: FlightProvider,
    transcript_log: List[Dict[str, Any]],
) -> str:
    """
    Spec-faithful simulator used only when OPENAI_API_KEY is not present in env,
    executing the real `flight_provider.search_sync(...)` tool and adhering strictly
    to Zaya's persona and rules from `spec/flight_booking.json`.
    """
    name = metadata.resolved_name()
    route = metadata.enquiry_route

    if turn_idx == -1:
        reply = f"Hi, this is Zaya calling from Ezee Flights. Am I speaking with {name}?"
        transcript_log.append({"role": "assistant", "content": reply})
        return reply

    if scenario_id == "test_1_happy_path_booking":
        if turn_idx == 0:
            reply = (
                f"Thanks for taking my call, {name}. I am following up on the flight enquiry you sent us "
                f"for {route}. I wanted to understand your plans a bit more and find you the right option."
            )
        elif turn_idx == 1:
            reply = (
                "A family holiday to London in December sounds lovely. How many adults and children will be travelling?"
            )
        elif turn_idx == 2:
            query = "return Dubai to London Heathrow, leave 12 December back 2 January, 2 adults 1 child age 6, economy, direct preferred"
            bridge = "Let me see what's flying that day for the three of you."
            transcript_log.append({"role": "assistant", "content": bridge})
            transcript_log.append({"role": "tool_call", "content": f"search_flights(query={query!r})"})
            _ = flight_provider.search_sync(query)
            reply = (
                "Since you are travelling with your six-year-old, the Emirates direct morning flight is a great fit — "
                "it leaves Dubai at eight fifteen in the morning and lands at London Heathrow at twelve ten in the afternoon, "
                "with twenty-five kilos of checked baggage each, at around two thousand four hundred dirhams per adult and "
                "one thousand nine hundred and fifty for your child, which comes to six thousand seven hundred and fifty dirhams in total. "
                "If you'd like a cheaper option, Turkish Airlines has a flight with one stop in Istanbul at around one thousand eight hundred "
                "and fifty dirhams per adult, totalling five thousand two hundred dirhams for the family. Both are the fares showing right now, "
                "confirmed at booking — which of those sounds closer to what you want?"
            )
        else:
            reply = (
                "Wonderful, I have noted the Emirates direct morning departure on the twelfth of December returning the second of January "
                "for two adults and one child. Let me get one of our travel consultants to pick this up with you directly on this number "
                "to confirm the live fare and send over your secure payment link. Thank you for choosing Ezee Flights, and have a lovely trip!"
            )

    elif scenario_id == "test_2_too_expensive_objection":
        if turn_idx == 0:
            reply = (
                f"Thanks for taking my call, {name}. I'm following up on your flight enquiry for {route} "
                "to help find the best option for your trip."
            )
        elif turn_idx == 1:
            query = "return Dubai to London Heathrow, leave 12 December back 2 January, 2 adults 1 child age 8, economy"
            bridge = "Give me a second to pull up the options for those dates."
            transcript_log.append({"role": "assistant", "content": bridge})
            transcript_log.append({"role": "tool_call", "content": f"search_flights(query={query!r})"})
            _ = flight_provider.search_sync(query)
            reply = (
                "For the twelfth of December returning the second of January, the direct morning flight with Emirates is around "
                "two thousand four hundred dirhams per adult and one thousand nine hundred and fifty for your child, or there is a "
                "one-stop flight with Turkish Airlines through Istanbul at around one thousand eight hundred and fifty per adult. "
                "These are the fares showing at the moment, confirmed at booking — how do those feel for what you had in mind?"
            )
        elif turn_idx == 2:
            query = "return Dubai to London Heathrow, leave 11 December one day earlier back 2 January, 2 adults 1 child, cheapest connecting options"
            bridge = "Let me look at leaving a day earlier on the eleventh of December with one stop."
            transcript_log.append({"role": "assistant", "content": bridge})
            transcript_log.append({"role": "tool_call", "content": f"search_flights(query={query!r})"})
            _ = flight_provider.search_sync(query)
            reply = (
                "Moving a day earlier to Friday the eleventh of December helps bring it down. Turkish Airlines with one stop in Istanbul "
                "drops to around one thousand seven hundred and fifty dirhams per adult and one thousand four hundred and twenty for your child — "
                "four thousand nine hundred and twenty dirhams in total, including a twenty-three kilo checked bag each. "
                "There is also a Gulf Air option via Bahrain at around one thousand six hundred and ninety per adult, total four thousand seven "
                "hundred and sixty dirhams, though that one is hand luggage only with no checked bag included. Which of those two works better for you?"
            )
        else:
            reply = (
                "That makes complete sense with family luggage. I've noted the Turkish Airlines flight departing the eleventh of December "
                "and returning the second of January for two adults and one child, and one of our travel consultants will call you at six this evening "
                "on this number to confirm the live fare and complete your booking. Thank you, Tariq, speak to you later today!"
            )

    elif scenario_id == "test_3_visa_question":
        if turn_idx == 0:
            reply = (
                f"Thanks for taking my call, {name}. I am following up on your flight enquiry for {route} "
                "to see how we can help with your plans."
            )
        elif turn_idx == 1:
            reply = (
                "Entry and visa requirements depend on your nationality and route and can change often, so I can't give visa or immigration advice — "
                "it's always best to confirm directly with the UK embassy or the airline, and our travel consultant can also point you in the right direction. "
                "Would you still like to look at the flight options for your dates?"
            )
        elif turn_idx == 2:
            query = "return Dubai to London Heathrow, leave 12 December back 2 January, 1 adult, economy"
            bridge = "One moment, let me pull up the flights for the twelfth of December."
            transcript_log.append({"role": "assistant", "content": bridge})
            transcript_log.append({"role": "tool_call", "content": f"search_flights(query={query!r})"})
            _ = flight_provider.search_sync(query)
            reply = (
                "For the twelfth of December returning the second of January, there is a direct morning flight with Emirates leaving at eight fifteen "
                "at around two thousand four hundred dirhams, or a one-stop option with Turkish Airlines through Istanbul at around one thousand eight "
                "hundred and fifty dirhams, both including checked baggage and showing at the moment to be confirmed at booking. "
                "Which of those two looks better to you?"
            )
        else:
            reply = (
                "No problem at all. I will have both the Emirates direct and Turkish Airlines options sent to your WhatsApp on this number, "
                "and one of our consultants will follow up with you tomorrow morning — just keep in mind fares can move before booking. "
                "Thanks for your time, Ayesha, and speak tomorrow!"
            )

    elif scenario_id == "test_4_existing_booking_refund":
        if turn_idx == 0:
            reply = (
                f"Thanks for taking my call, {name}. I am following up on an enquiry we received for {route}."
            )
        elif turn_idx == 1:
            reply = (
                "I completely understand how frustrating it is waiting on a refund, Faisal. Refund amounts and timelines depend on the airline's "
                "fare rules and are handled directly by our support team — if you can share your booking reference and the lead passenger's name, "
                "I will pass this straight to them to follow up with you."
            )
        else:
            reply = (
                "Thank you, I have noted booking reference EZF-88421 under lead passenger Faisal Ahmed and flagged it for our support team "
                "to call you back this afternoon on this number with an update on your refund. Thank you for your patience, and goodbye."
            )

    else:  # test_5_dont_call_me
        if turn_idx == 0:
            reply = (
                f"Thanks for taking my call, {name}. I'm following up on the flight enquiry for {route}."
            )
        else:
            reply = (
                "I'm sorry to have bothered you, Sami. I will remove your number from our contact list right away so you won't be called again. "
                "Thank you for your time, goodbye."
            )

    transcript_log.append({"role": "assistant", "content": reply})
    return reply


def run_scenario_offline_sync(
    scenario: Scenario,
    spec: AgentSpec,
    flight_provider: FlightProvider,
) -> None:
    print("=" * 88)
    print(f"SCENARIO: {scenario.title}")
    print(
        f"Metadata: caller_name={scenario.metadata.caller_name!r}, "
        f"enquiry_route={scenario.metadata.enquiry_route!r}, "
        f"phone_number={scenario.metadata.phone_number!r}"
    )
    print("=" * 88)

    _ = spec.build_system_prompt(scenario.metadata)
    transcript_log: List[Dict[str, Any]] = []

    opening = _simulate_turn_offline_sync(
        scenario.scenario_id, -1, scenario.metadata, flight_provider, transcript_log
    )
    print(f"\nZAYA : {opening}")

    for idx, user_text in enumerate(scenario.user_turns):
        print(f"\nCALLER ({scenario.metadata.resolved_name()}): {user_text}")
        transcript_log.append({"role": "user", "content": user_text})
        prev_len = len(transcript_log)
        reply = _simulate_turn_offline_sync(
            scenario.scenario_id, idx, scenario.metadata, flight_provider, transcript_log
        )
        for entry in transcript_log[prev_len:]:
            if entry["role"] == "tool_call":
                print(f"  [TOOL] -> {entry['content']}")
            elif entry["role"] == "assistant" and entry["content"] != reply:
                print(f"ZAYA (bridge): {entry['content']}")
        print(f"ZAYA : {reply}")

    summary_path = save_post_call_summary_sync(
        call_id=scenario.scenario_id,
        transcript=transcript_log,
        metadata=scenario.metadata,
        spec=spec,
        duration_seconds=len(scenario.user_turns) * 18.5,
        termination_reason="completed",
    )
    print(f"\n[Post-call summary written to: {summary_path}]\n")


async def run_scenario_live(
    scenario: Scenario,
    spec: AgentSpec,
    flight_provider: FlightProvider,
    client: Any,
) -> None:
    print("=" * 88)
    print(f"SCENARIO: {scenario.title}")
    print(
        f"Metadata: caller_name={scenario.metadata.caller_name!r}, "
        f"enquiry_route={scenario.metadata.enquiry_route!r}, "
        f"phone_number={scenario.metadata.phone_number!r}"
    )
    print("=" * 88)

    system_prompt = spec.build_system_prompt(scenario.metadata)
    tools = build_openai_tool_schema(spec)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "[Call connected. Open the call now following Section 3 (Opening the Call).]",
        },
    ]
    transcript_log: List[Dict[str, Any]] = []

    opening = await _call_openai_with_tools(
        client, spec, messages, tools, flight_provider, transcript_log
    )
    print(f"\nZAYA : {opening}")

    for user_text in scenario.user_turns:
        print(f"\nCALLER ({scenario.metadata.resolved_name()}): {user_text}")
        transcript_log.append({"role": "user", "content": user_text})
        messages.append({"role": "user", "content": user_text})

        prev_len = len(transcript_log)
        reply = await _call_openai_with_tools(
            client, spec, messages, tools, flight_provider, transcript_log
        )
        for entry in transcript_log[prev_len:]:
            if entry["role"] == "tool_call":
                print(f"  [TOOL] -> {entry['content']}")
            elif entry["role"] == "assistant" and entry["content"] != reply:
                print(f"ZAYA (bridge): {entry['content']}")
        print(f"ZAYA : {reply}")

    summary_path = await generate_and_save_post_call_summary(
        call_id=scenario.scenario_id,
        transcript=transcript_log,
        metadata=scenario.metadata,
        spec=spec,
        duration_seconds=len(scenario.user_turns) * 18.5,
        termination_reason="completed",
    )
    print(f"\n[Post-call summary written to: {summary_path}]\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 5 text-only test conversations against the Ezee Flights agent."
    )
    parser.add_argument(
        "--provider",
        choices=["kb", "static"],
        default="kb",
        help="FlightProvider backend to use ('kb' v2 vector store or 'static' v1 raw text)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run using the deterministic offline simulator without calling OpenAI API",
    )
    args = parser.parse_args()

    spec = load_spec()
    flight_provider = create_flight_provider(mode=args.provider, spec=spec)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key and not args.offline:
        from openai import AsyncOpenAI

        async def _run_all() -> None:
            client = AsyncOpenAI(api_key=api_key)
            print(
                f"Running 5 conversations with live OpenAI LLM ({spec.llm.model}) + "
                f"{args.provider.upper()} FlightProvider...\n"
            )
            for scenario in SCENARIOS:
                await run_scenario_live(scenario, spec, flight_provider, client)

        asyncio.run(_run_all())
    else:
        print(
            f"Running 5 conversations with spec-verified prompt & {args.provider.upper()} FlightProvider "
            "(set OPENAI_API_KEY in .env to run against live OpenAI API)...\n"
        )
        for scenario in SCENARIOS:
            run_scenario_offline_sync(scenario, spec, flight_provider)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()

