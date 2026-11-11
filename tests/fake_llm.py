"""FakeChatModel for deterministic testing without Groq API calls."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    """Deterministic LLM replacement for testing.

    Returns pre-scripted responses in order. Supports:
    - Plain text responses
    - AIMessages with tool_calls
    - Structured output (via tool-call-based parsing)

    Records all invocations for assertion in tests.
    """

    responses: list[BaseMessage]
    supports_parallel_tools: bool = False
    invocations: list[list[BaseMessage]] = []
    _call_index: int = 0

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Return the next pre-scripted response."""
        # Record the invocation
        self.invocations.append(list(messages))

        if self._call_index >= len(self.responses):
            raise IndexError(
                f"FakeChatModel exhausted all {len(self.responses)} responses. "
                f"Call index: {self._call_index}. "
                f"Add more responses to the fixture."
            )

        response = self.responses[self._call_index]
        self._call_index += 1

        # Ensure we return an AIMessage
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response.content))

        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": "fake", "responses_count": len(self.responses)}

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> FakeChatModel:
        """Return self — tools are acknowledged but responses are pre-scripted."""
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeStructuredOutput:
        """Return a wrapper that parses pre-scripted responses as structured output."""
        return FakeStructuredOutput(model=self, schema=schema)


class FakeStructuredOutput:
    """Wrapper that returns pre-parsed structured output from FakeChatModel.

    When the pre-scripted AIMessage content is a JSON string matching the schema,
    it parses and returns the Pydantic model. When the content is already a dict,
    it validates directly.
    """

    def __init__(self, model: FakeChatModel, schema: Any) -> None:
        self.model = model
        self.schema = schema

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        """Invoke the fake model and parse the response as structured output."""
        import json

        # Get messages from input
        if isinstance(input, list):
            messages = input
        elif isinstance(input, str):
            from langchain_core.messages import HumanMessage

            messages = [HumanMessage(content=input)]
        else:
            messages = input

        result = self.model._generate(messages)
        content = result.generations[0].message.content

        # Try to parse as the schema
        if isinstance(content, str):
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                raise ValueError(
                    f"FakeStructuredOutput: response is not valid JSON: {content[:200]}"
                )
        elif isinstance(content, dict):
            data = content
        else:
            raise ValueError(f"FakeStructuredOutput: unexpected content type: {type(content)}")

        # Validate with the Pydantic schema
        if hasattr(self.schema, "model_validate"):
            return self.schema.model_validate(data)
        return data

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        """Async version — delegates to sync since FakeChatModel is synchronous."""
        return self.invoke(input, **kwargs)
