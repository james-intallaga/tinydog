"""Optional server-side teacher-model adapters; disabled unless configured."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class TeacherFeedback:
    available: bool
    diagnosis: str
    hint: str
    provider: str | None
    model: str | None
    error: str | None


class TeacherClient:
    """Ask an explicitly configured teacher for compact feedback.

    Provider output is advisory. The apprenticeship pipeline admits it into
    permanent arithmetic training only after the local exact verifier agrees.
    Users are responsible for confirming that their provider agreement permits
    the intended use of output.
    """

    def __init__(
        self, provider: str = "", model: str = "", api_key: str = "", base_url: str = ""
    ) -> None:
        self.provider = provider.strip().lower()
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "TeacherClient":
        provider = os.getenv("MLM_TEACHER_PROVIDER", "").strip().lower()
        model = os.getenv("MLM_TEACHER_MODEL", "").strip()
        if provider == "openai":
            return cls(
                provider,
                model,
                os.getenv("OPENAI_API_KEY", ""),
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        if provider == "anthropic":
            return cls(
                provider,
                model,
                os.getenv("ANTHROPIC_API_KEY", ""),
                os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
            )
        return cls()

    @property
    def enabled(self) -> bool:
        return self.provider in {"openai", "anthropic"} and bool(
            self.model and self.api_key
        )

    def status(self) -> dict[str, str | bool | None]:
        return {
            "enabled": self.enabled,
            "provider": self.provider or None,
            "model": self.model or None,
        }

    def diagnose(
        self, problem: str, student_answer: str, correct_answer: str | None = None
    ) -> TeacherFeedback:
        if not self.enabled:
            return TeacherFeedback(
                False, "", "", None, None, "Teacher API is not configured"
            )
        prompt = (
            "You are teaching a very small arithmetic model. Diagnose its mistake and give one short hint. "
            "Return JSON only with string fields diagnosis and hint. Do not invent facts.\n"
            f"Problem: {problem}\nStudent answer: {student_answer}\nVerified answer: {correct_answer or 'unknown'}"
        )
        try:
            text = (
                self._openai(prompt)
                if self.provider == "openai"
                else self._anthropic(prompt)
            )
            payload = self._parse_feedback(text)
            return TeacherFeedback(
                True,
                payload["diagnosis"],
                payload["hint"],
                self.provider,
                self.model,
                None,
            )
        except (
            ValueError,
            KeyError,
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
        ) as error:
            return TeacherFeedback(False, "", "", self.provider, self.model, str(error))

    def _openai(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "input": prompt,
            "store": False,
            "max_output_tokens": 200,
        }
        response = self._post(
            f"{self.base_url}/responses",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        texts: list[str] = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if isinstance(content.get("text"), str):
                    texts.append(content["text"])
        if not texts:
            raise ValueError("Teacher returned no text")
        return "\n".join(texts)

    def _anthropic(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self._post(
            f"{self.base_url}/messages",
            payload,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        texts = [
            item.get("text", "")
            for item in response.get("content", [])
            if item.get("type") == "text"
        ]
        if not any(texts):
            raise ValueError("Teacher returned no text")
        return "\n".join(texts)

    @staticmethod
    def _parse_feedback(text: str) -> dict[str, str]:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("Teacher did not return JSON")
        payload = json.loads(match.group(0))
        diagnosis, hint = (
            str(payload.get("diagnosis", "")).strip(),
            str(payload.get("hint", "")).strip(),
        )
        if not diagnosis and not hint:
            raise ValueError("Teacher feedback was empty")
        return {"diagnosis": diagnosis[:500], "hint": hint[:500]}

    @staticmethod
    def _post(url: str, payload: dict, headers: dict[str, str]) -> dict:
        request = Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            parsed = json.loads(response.read())
        if not isinstance(parsed, dict):
            raise ValueError("Teacher returned invalid JSON")
        return parsed
