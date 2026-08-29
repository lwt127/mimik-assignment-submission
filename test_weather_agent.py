"""Behavior-focused tests for the weather agent."""

import json
import unittest
from urllib.parse import parse_qs, urlparse

from weather_agent import Settings, WeatherAgent


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class RecordingOpener:
    def __init__(
        self,
        model_answer="Vancouver is currently 18.2 C with no precipitation.",
        classifier_answer=None,
    ):
        self.requests = []
        self.model_answer = model_answer
        self.classifier_answer = classifier_answer

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        if "geocoding-api" in request.full_url:
            return FakeResponse(
                {
                    "results": [
                        {
                            "name": "Vancouver",
                            "admin1": "British Columbia",
                            "country": "Canada",
                            "latitude": 49.24966,
                            "longitude": -123.11934,
                        }
                    ]
                }
            )
        if "api.open-meteo.com" in request.full_url:
            return FakeResponse(
                {
                    "current": {
                        "temperature_2m": 18.2,
                        "apparent_temperature": 17.8,
                        "relative_humidity_2m": 72,
                        "precipitation": 0,
                        "weather_code": 2,
                        "wind_speed_10m": 6.1,
                    },
                    "current_units": {
                        "temperature_2m": "C",
                        "precipitation": "mm",
                        "wind_speed_10m": "km/h",
                    },
                    "daily": {
                        "time": ["2026-08-29", "2026-08-30"],
                        "precipitation_probability_max": [20, 75],
                        "precipitation_sum": [0.0, 4.2],
                        "weather_code": [2, 61],
                    },
                    "daily_units": {
                        "precipitation_probability_max": "%",
                        "precipitation_sum": "mm",
                    },
                }
            )
        request_payload = json.loads(request.data.decode("utf-8"))
        if "Classify whether" in request_payload["messages"][0]["content"]:
            answer = self.classifier_answer
            if answer is None:
                question = request_payload["messages"][1]["content"].lower()
                answer = json.dumps({"is_weather": "hamlet" not in question})
            return FakeResponse(
                {"choices": [{"message": {"content": answer}}]}
            )
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": self.model_answer
                        }
                    }
                ]
            }
        )


class WeatherAgentTests(unittest.TestCase):
    def setUp(self):
        self.opener = RecordingOpener()
        self.agent = WeatherAgent(
            Settings(
                mimik_base_url="http://localhost:8083/mimik-ai/openai/v1",
                mimik_api_key="test-key",
                mimik_model="qwen3-4b",
                request_timeout=5,
            ),
            opener=self.opener,
        )

    def test_rejects_non_weather_questions_without_network_calls(self):
        answer = self.agent.answer("Who wrote Hamlet?")

        self.assertEqual(answer, "I can only answer weather questions.")
        self.assertEqual(len(self.opener.requests), 1)

    def test_asks_for_location_when_missing(self):
        answer = self.agent.answer("What is the weather today?")

        self.assertEqual(answer, "Which location would you like the weather for?")
        self.assertEqual(len(self.opener.requests), 1)

    def test_connects_weather_data_to_mimoe_chat_completion(self):
        answer = self.agent.answer("Will it rain in Vancouver today?")

        self.assertIn("Rain is not expected today", answer)
        self.assertIn("20%", answer)
        self.assertIn("0.0 mm", answer)
        self.assertEqual(len(self.opener.requests), 4)

        geocode_request, _ = self.opener.requests[1]
        geocode_query = parse_qs(urlparse(geocode_request.full_url).query)
        self.assertEqual(geocode_query["name"], ["Vancouver"])

        mimik_request, timeout = self.opener.requests[3]
        self.assertEqual(
            mimik_request.full_url,
            "http://localhost:8083/mimik-ai/openai/v1/chat/completions",
        )
        self.assertEqual(timeout, 5)
        self.assertEqual(
            mimik_request.headers["Authorization"], "Bearer test-key"
        )
        payload = json.loads(mimik_request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen3-4b")
        self.assertIn("18.2", payload["messages"][1]["content"])

    def test_qwen_semantically_accepts_an_umbrella_question(self):
        opener = RecordingOpener(classifier_answer='{"is_weather": true}')
        agent = WeatherAgent(self.agent.settings, opener=opener)

        answer = agent.answer("Should I bring an umbrella in Vancouver?")

        self.assertIn("Rain is not expected today", answer)
        self.assertIn("20%", answer)
        classifier_payload = json.loads(opener.requests[0][0].data.decode("utf-8"))
        self.assertEqual(classifier_payload["temperature"], 0)

    def test_rain_question_uses_tomorrow_forecast_when_requested(self):
        answer = self.agent.answer("Will it rain in Vancouver tomorrow?")

        self.assertIn("Rain is expected tomorrow", answer)
        self.assertIn("75%", answer)
        self.assertIn("4.2 mm", answer)

    def test_falls_back_when_local_model_invents_a_provider(self):
        opener = RecordingOpener(
            "Current temperature is 18.2 C according to the Weather Underground API."
        )
        agent = WeatherAgent(self.agent.settings, opener=opener)

        answer = agent.answer("What is the weather in Vancouver?")

        self.assertIn("Current conditions in Vancouver", answer)
        self.assertIn("18.2 C", answer)
        self.assertNotIn("Weather Underground", answer)


if __name__ == "__main__":
    unittest.main()
