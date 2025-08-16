#!/usr/bin/env python3
"""
Terminal chat for your RAG stack
Commands:
  /ingest     rebuild doc index from ./docs (shows progress bars)
  /reset      start a new empty conversation
  /session    print current session id
  /help       show commands
  /exit       quit
"""

import sys
import uuid
import traceback
import time

# optional input niceties
try:
    import readline  # noqa: F401
except Exception:
    pass

import google.generativeai as genai

from src.ingestdocs import run_ingest
from src.chain import answer_with_context
from src.conversation_history import add_turn
from src.key import GOOGLE_API_KEY

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"


def _ping_gemini() -> str:
    """Return model reply text if Gemini is reachable; raise on error."""
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content("ping")
    return resp.text.strip() if getattr(resp, "text", None) else "ok"


def banner(session_id: str, connected: bool, latency_ms: int):
    print(f"{BOLD}quickrag terminal chat{RESET}")
    print(f"{DIM}session: {session_id}{RESET}")
    conn_txt = f"{'connected' if connected else 'NOT CONNECTED'}"
    print(f"{DIM}gemini: {conn_txt} (~{latency_ms}ms){RESET}")
    print(f"{DIM}docs dir: ./docs   |   indexes: ./data/index/{RESET}")
    print("type your question, or commands:")
    print(f"  {YELLOW}/ingest{RESET}   rebuild doc index from ./docs")
    print(f"  {YELLOW}/reset{RESET}    start a new empty conversation")
    print(f"  {YELLOW}/session{RESET}  show current session id")
    print(f"  {YELLOW}/help{RESET}     show this help")
    print(f"  {YELLOW}/exit{RESET}     quit\n")


def ensure_key():
    if not GOOGLE_API_KEY or not isinstance(GOOGLE_API_KEY, str):
        print("GOOGLE_API_KEY missing or invalid in src/keys.py")
        sys.exit(1)


def new_session() -> str:
    return uuid.uuid4().hex[:12]


def main():
    ensure_key()

    # Boot-up Gemini connectivity test
    t0 = time.time()
    connected = True
    try:
        _ = _ping_gemini()
    except Exception as e:
        connected = False
        print(f"{YELLOW}warning:{RESET} could not reach Gemini: {e}")
    t1 = time.time()
    latency_ms = int((t1 - t0) * 1000)

    session_id = new_session()
    banner(session_id, connected, latency_ms)

    while True:
        try:
            user = input(f"{BLUE}you › {RESET}").strip()
            if not user:
                continue

            cmd = user.lower()

            if cmd in ("/exit", "/quit"):
                print("bye!")
                break

            if cmd in ("/help", "help"):
                banner(session_id, connected, latency_ms)
                continue

            if cmd in ("/session",):
                print(f"{DIM}current session: {session_id}{RESET}")
                continue

            if cmd in ("/reset",):
                session_id = new_session()
                print(f"{DIM}started new session: {session_id}{RESET}")
                continue

            if cmd.startswith("/ingest"):
                print(f"{DIM}ingesting docs…{RESET}")
                stats = run_ingest(force_rebuild=True)
                print(f"{DIM}ingest complete: {stats}{RESET}")
                continue

            # normal chat turn
            add_turn(session_id, role="user", text=user)
            answer = answer_with_context(question=user, session_id=session_id)
            add_turn(session_id, role="assistant", text=answer)
            print(f"{GREEN}assistant ›{RESET} {answer}\n")

        except KeyboardInterrupt:
            print("\nbye!")
            break
        except EOFError:
            print("\nbye!")
            break
        except Exception:
            print(f"{YELLOW}error:{RESET} something went wrong:")
            traceback.print_exc(limit=2)
            print()


if __name__ == "__main__":
    main()
