"""Command-line entry point for the mimOE weather agent."""

from __future__ import annotations

import sys
from urllib.error import HTTPError, URLError

from weather_agent import WeatherAgent


def run() -> int:
    agent = WeatherAgent()

    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        print("mimOE Weather Agent (type 'quit' to exit)")
        questions = iter(lambda: input("You: ").strip(), "quit")

    for question in questions:
        if question.lower() in {"quit", "exit"}:
            break
        try:
            print(f"Agent: {agent.answer(question)}")
        except (HTTPError, URLError, TimeoutError) as error:
            print(f"Agent: A required service is unavailable: {error}", file=sys.stderr)
            return 1
        except (RuntimeError, ValueError) as error:
            print(f"Agent: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
