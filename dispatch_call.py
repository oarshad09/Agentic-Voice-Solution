"""
CLI to dispatch an outbound call through LiveKit Agent Dispatch + SIP Trunk.

Usage:
  python dispatch_call.py --to +9715XXXXXXX --name "Omair" --route "Dubai to London"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from spec_loader import CallMetadata, load_spec

load_dotenv(Path(__file__).parent / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place an outbound Ezee Flights callback via LiveKit SIP trunk."
    )
    parser.add_argument(
        "--to",
        "--phone-number",
        dest="to",
        required=True,
        help="Destination phone number in E.164 format (e.g. +971501234567)",
    )
    parser.add_argument(
        "--name",
        "--caller-name",
        "--customer-name",
        dest="name",
        default="",
        help="Caller / customer name (maps to caller_name and customer_name)",
    )
    parser.add_argument(
        "--route",
        "--enquiry-route",
        dest="route",
        default="",
        help="Optional flight route enquired about (e.g. 'Dubai to London')",
    )
    parser.add_argument(
        "--room",
        default="",
        help="Optional LiveKit room name (auto-generated if omitted)",
    )
    parser.add_argument(
        "--agent-name",
        default=os.getenv("LIVEKIT_AGENT_NAME", "ezee-flights-outbound"),
        help="Registered LiveKit agent worker name",
    )
    parser.add_argument(
        "--trunk-id",
        default=os.getenv("SIP_OUTBOUND_TRUNK_ID", ""),
        help="LiveKit SIP outbound trunk ID (overrides SIP_OUTBOUND_TRUNK_ID env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print dispatch payload without calling LiveKit API",
    )
    return parser.parse_args()


async def dispatch_outbound_call(args: argparse.Namespace) -> int:
    spec = load_spec()
    metadata = CallMetadata(
        caller_name=args.name.strip(),
        customer_name=args.name.strip(),
        enquiry_route=args.route.strip(),
        phone_number=args.to.strip(),
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    room_name = args.room.strip() or f"ezee-call-{ts}-{uuid.uuid4().hex[:6]}"
    metadata_payload = {
        "caller_name": metadata.caller_name,
        "customer_name": metadata.customer_name,
        "enquiry_route": metadata.enquiry_route,
        "phone_number": metadata.phone_number,
    }
    metadata_json = json.dumps(metadata_payload, ensure_ascii=False)

    if args.dry_run:
        rendered_prompt = spec.build_system_prompt(metadata)
        print("=== DRY RUN: Outbound Call Dispatch ===")
        print(f"Room Name   : {room_name}")
        print(f"Agent Name  : {args.agent_name}")
        print(f"SIP Trunk   : {args.trunk_id or '(from worker env SIP_OUTBOUND_TRUNK_ID)'}")
        print(f"Metadata    : {metadata_json}")
        print(f"Prompt len  : {len(rendered_prompt)} chars (placeholders verified clean)")
        return 0

    livekit_url = os.getenv("LIVEKIT_URL", "").strip()
    livekit_api_key = os.getenv("LIVEKIT_API_KEY", "").strip()
    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()

    if not (livekit_url and livekit_api_key and livekit_api_secret):
        print(
            "Error: LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET must be set in .env or environment.",
            file=sys.stderr,
        )
        return 1

    from livekit import api

    lkapi = api.LiveKitAPI(
        url=livekit_url,
        api_key=livekit_api_key,
        api_secret=livekit_api_secret,
    )
    try:
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=args.agent_name,
                room=room_name,
                metadata=metadata_json,
            )
        )
        print(f"Dispatched agent '{args.agent_name}' to room '{room_name}' (dispatch_id={dispatch.id})")

        # If a trunk ID is provided here and worker doesn't auto-dial, ensure SIP participant is created
        if args.trunk_id and os.getenv("DIAL_FROM_DISPATCH_CLI", "false").lower() == "true":
            sip_participant = await lkapi.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=room_name,
                    sip_trunk_id=args.trunk_id,
                    sip_call_to=metadata.phone_number,
                    participant_identity=f"sip_{metadata.phone_number}",
                    participant_name=metadata.resolved_name() or "Customer",
                    wait_until_answered=True,
                )
            )
            print(
                f"Outbound SIP call connected: participant_id={sip_participant.participant_id} -> {metadata.phone_number}"
            )
        else:
            print(
                f"Worker will dial {metadata.phone_number} via SIP trunk upon joining room '{room_name}'."
            )
            print(f"Post-call summary will be saved to: calls/{room_name}.json")
        return 0
    finally:
        await lkapi.aclose()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        spec = load_spec()
        metadata = CallMetadata(
            caller_name=args.name.strip(),
            customer_name=args.name.strip(),
            enquiry_route=args.route.strip(),
            phone_number=args.to.strip(),
        )
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        room_name = args.room.strip() or f"ezee-call-{ts}-{uuid.uuid4().hex[:6]}"
        metadata_json = json.dumps(
            {
                "caller_name": metadata.caller_name,
                "customer_name": metadata.customer_name,
                "enquiry_route": metadata.enquiry_route,
                "phone_number": metadata.phone_number,
            },
            ensure_ascii=False,
        )
        rendered_prompt = spec.build_system_prompt(metadata)
        print("=== DRY RUN: Outbound Call Dispatch ===")
        print(f"Room Name   : {room_name}")
        print(f"Agent Name  : {args.agent_name}")
        print(f"SIP Trunk   : {args.trunk_id or '(from worker env SIP_OUTBOUND_TRUNK_ID)'}")
        print(f"Metadata    : {metadata_json}")
        print(f"Prompt len  : {len(rendered_prompt)} chars (placeholders verified clean)")
        sys.exit(0)
    exit_code = asyncio.run(dispatch_outbound_call(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

