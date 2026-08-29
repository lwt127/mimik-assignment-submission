"""Weather-only agent powered by a local mimOE OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

JsonOpener = Callable[[Request], Any]


@dataclass(frozen=True)
class Settings:
    mimik_base_url: str = os.getenv(
        "MIMIK_BASE_URL", "http://192.168.0.120:8083/mimik-ai/openai/v1"
    )
    mimik_api_key: str = os.getenv("MIMIK_API_KEY", "1234")
    mimik_model: str = os.getenv("MIMIK_MODEL", "qwen3-4b")
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "30"))


class WeatherAgent:
    WEATHER_TERMS = {
        "weather",
        "temperature",
        "forecast",
        "rain",
        "raining",
        "snow",
        "wind",
        "humid",
        "humidity",
        "sunny",
        "cloudy",
    }

    def __init__(
        self,
        settings: Optional[Settings] = None,
        opener: JsonOpener = urlopen,
    ) -> None:
        self.settings = settings or Settings()
        self.opener = opener

    def answer(self, question: str) -> str:
        question = question.strip()
        if not question:
            return "Please ask a weather question and include a location."
        if not self._is_weather_question(question):
            return "I can only answer weather questions."

        location = self._extract_location(question)
        if not location:
            return "Which location would you like the weather for?"

        place = self._geocode(location)
        weather = self._current_weather(place["latitude"], place["longitude"])
        return self._summarize(question, place, weather)

    def _is_weather_question(self, question: str) -> bool:
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify whether the user is asking about weather or conditions "
                    "affected by weather, such as needing an umbrella. Return only JSON "
                    'in this exact format: {"is_weather": true} or '
                    '{"is_weather": false}.'
                ),
            },
            {"role": "user", "content": question},
        ]
        try:
            answer = self._chat_completion(messages, temperature=0)
            match = re.search(
                r'\{\s*"is_weather"\s*:\s*(true|false)\s*\}',
                answer,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).lower() == "true"
        except (KeyError, IndexError, TypeError, ValueError):
            pass

        words = set(re.findall(r"[a-z]+", question.lower()))
        return bool(words & self.WEATHER_TERMS)

    def _chat_completion(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
    ) -> str:
        body = json.dumps(
            {
                "model": self.settings.mimik_model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.settings.mimik_base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.settings.mimik_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        payload = self._request_json(request)
        return str(payload["choices"][0]["message"]["content"]).strip()

    def _extract_location(self, question: str) -> Optional[str]:
        normalized = question.strip().rstrip("?.!")
        patterns = (
            r"\b(?:in|at|for|near)\s+(.+)$",
            r"\b(?:weather|forecast)\s+(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                location = re.sub(
                    r"\b(?:today|tomorrow|right now|now|this week)$",
                    "",
                    location,
                    flags=re.IGNORECASE,
                ).strip()
                if location:
                    return location
        return None

    def _request_json(self, request: Request) -> Dict[str, Any]:
        with self.opener(request, timeout=self.settings.request_timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _geocode(self, location: str) -> Dict[str, Any]:
        query = urlencode({"name": location, "count": 1, "language": "en", "format": "json"})
        request = Request(f"https://geocoding-api.open-meteo.com/v1/search?{query}")
        payload = self._request_json(request)
        results = payload.get("results") or []
        if not results:
            raise ValueError(f"I could not find the location '{location}'.")
        return results[0]

    def _current_weather(self, latitude: float, longitude: float) -> Dict[str, Any]:
        query = urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
                "daily": (
                    "precipitation_probability_max,precipitation_sum,weather_code"
                ),
                "forecast_days": 2,
                "timezone": "auto",
            }
        )
        request = Request(f"https://api.open-meteo.com/v1/forecast?{query}")
        payload = self._request_json(request)
        if "current" not in payload:
            raise RuntimeError("The weather service returned no current conditions.")
        weather = dict(payload["current"])
        weather["_units"] = payload.get("current_units", {})
        weather["_daily"] = payload.get("daily", {})
        weather["_daily_units"] = payload.get("daily_units", {})
        return weather

    def _summarize(
        self,
        question: str,
        place: Dict[str, Any],
        weather: Dict[str, Any],
    ) -> str:
        place_name = ", ".join(
            str(value)
            for value in (place.get("name"), place.get("admin1"), place.get("country"))
            if value
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise weather assistant. Directly answer the user's specific "
                    "question using only the supplied Open-Meteo observation and forecast. "
                    "For a yes/no rain question, begin with whether rain is expected and cite "
                    "the forecast probability. Otherwise, include the exact current temperature. "
                    "Never name another provider, describe the interval, or invent facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nLocation: {place_name}\n"
                    f"Current weather JSON: {json.dumps(weather, sort_keys=True)}"
                ),
            },
        ]
        try:
            answer = self._chat_completion(messages, temperature=0.2)
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("mimOE returned an unexpected response.") from error

        temperature = str(weather.get("temperature_2m", ""))
        rejected_terms = ("weather underground", "interval", "provider", "source api")
        normalized = answer.lower()
        rain_question = bool(re.search(r"\b(?:rain|raining|umbrella)\b", question.lower()))
        forecast_index = 1 if "tomorrow" in question.lower() else 0
        probabilities = weather.get("_daily", {}).get(
            "precipitation_probability_max", []
        )
        forecast_probability = (
            str(probabilities[forecast_index])
            if forecast_index < len(probabilities)
            else None
        )
        direct_rain_answer = "rain" in normalized and any(
            term in normalized for term in ("expected", "likely", "unlikely", "chance")
        )
        grounded_rain_answer = (
            direct_rain_answer
            and forecast_probability is not None
            and forecast_probability in answer
        )
        if (
            (not rain_question and temperature not in answer)
            or (not rain_question and "current" not in normalized)
            or (rain_question and not grounded_rain_answer)
            or any(term in normalized for term in rejected_terms)
            or len(answer) > 400
        ):
            return self._fallback_summary(question, place_name, weather)
        return answer

    def _fallback_summary(
        self,
        question: str,
        place_name: str,
        weather: Dict[str, Any],
    ) -> str:
        if re.search(r"\b(?:rain|raining|umbrella)\b", question.lower()):
            return self._rain_summary(question, place_name, weather)

        units = weather.get("_units", {})
        temperature_unit = units.get("temperature_2m", "C")
        wind_unit = units.get("wind_speed_10m", "km/h")
        precipitation_unit = units.get("precipitation", "mm")
        return (
            f"Current conditions in {place_name}: "
            f"{weather.get('temperature_2m')} {temperature_unit}, "
            f"{weather.get('relative_humidity_2m')}% humidity, "
            f"{weather.get('precipitation')} {precipitation_unit} precipitation, "
            f"and wind at {weather.get('wind_speed_10m')} {wind_unit}."
        )

    def _rain_summary(
        self,
        question: str,
        place_name: str,
        weather: Dict[str, Any],
    ) -> str:
        daily = weather.get("_daily", {})
        daily_units = weather.get("_daily_units", {})
        period = "tomorrow" if "tomorrow" in question.lower() else "today"
        index = 1 if period == "tomorrow" else 0
        probabilities = daily.get("precipitation_probability_max", [])
        totals = daily.get("precipitation_sum", [])
        if index >= len(probabilities) or index >= len(totals):
            return f"A {period} rain forecast is unavailable for {place_name}."

        probability = probabilities[index]
        total = totals[index]
        expected = probability >= 50 or total > 0.1
        verdict = "Rain is expected" if expected else "Rain is not expected"
        probability_unit = daily_units.get("precipitation_probability_max", "%")
        total_unit = daily_units.get("precipitation_sum", "mm")
        return (
            f"{verdict} {period} in {place_name}: the maximum rain chance is "
            f"{probability}{probability_unit}, with {total} {total_unit} forecast."
        )
