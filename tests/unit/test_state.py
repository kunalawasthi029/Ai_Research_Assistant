"""Unit tests for LangGraph state definition and reducers."""

from __future__ import annotations

from research_agent.graph.state import ResearchState, create_initial_state


class TestCreateInitialState:
    """Tests for the state factory function."""

    def test_creates_valid_state(self) -> None:
        state = create_initial_state(
            research_id="run-001",
            thread_id="thread-001",
            query="What is AI?",
        )
        assert state["research_id"] == "run-001"
        assert state["thread_id"] == "thread-001"
        assert state["research_query"] == "What is AI?"
        # Objective defaults to query when not provided
        assert state["research_objective"] == "What is AI?"

    def test_custom_objective(self) -> None:
        state = create_initial_state(
            research_id="run-002",
            thread_id="thread-002",
            query="AI safety",
            objective="Research current approaches to AI alignment",
        )
        assert state["research_objective"] == "Research current approaches to AI alignment"

    def test_all_list_fields_initialized_empty(self) -> None:
        state = create_initial_state("r", "t", "q")
        list_fields = [
            "completed_tasks",
            "tool_calls",
            "tool_results",
            "evidence",
            "findings",
            "sources",
            "validation_feedback",
            "agent_decisions",
            "error_details",
            "messages",
            "token_usage",
        ]
        for field in list_fields:
            assert state[field] == [], f"{field} should be an empty list"

    def test_optional_fields_initialized_none(self) -> None:
        state = create_initial_state("r", "t", "q")
        optional_fields = [
            "research_plan",
            "active_task",
            "analyst_summary",
            "report",
            "validation_result",
        ]
        for field in optional_fields:
            assert state[field] is None, f"{field} should be None"

    def test_counters_initialized_to_zero(self) -> None:
        state = create_initial_state("r", "t", "q")
        assert state["plan_revision_count"] == 0
        assert state["revision_count"] == 0
        assert state["research_iteration"] == 0

    def test_execution_status_initialized(self) -> None:
        state = create_initial_state("r", "t", "q")
        assert state["execution_status"] == "initialized"


class TestStateReducers:
    """Tests that verify reducer behavior on the state TypedDict.

    These tests verify the annotation metadata is correct by checking
    that the TypedDict annotations use operator.add where expected.
    LangGraph uses these annotations at runtime to determine how to
    merge node outputs into the state.
    """

    def test_append_fields_have_annotated_metadata(self) -> None:
        """Verify fields meant to append have Annotated with operator.add."""
        import operator
        from typing import Annotated, get_args, get_origin, get_type_hints

        hints = get_type_hints(ResearchState, include_extras=True)

        append_fields = [
            "completed_tasks",
            "tool_calls",
            "tool_results",
            "evidence",
            "sources",
            "validation_feedback",
            "agent_decisions",
            "error_details",
            "token_usage",
        ]

        for field_name in append_fields:
            hint = hints[field_name]
            # Should be Annotated[list[...], operator.add]
            assert get_origin(hint) is Annotated, f"{field_name} should be Annotated but is {hint}"
            args = get_args(hint)
            # The second arg should be the reducer function
            assert args[1] is operator.add, (
                f"{field_name} reducer should be operator.add but is {args[1]}"
            )

    def test_overwrite_fields_are_not_annotated(self) -> None:
        """Verify fields that overwrite don't have Annotated metadata."""
        from typing import Annotated, get_origin, get_type_hints

        hints = get_type_hints(ResearchState, include_extras=True)

        overwrite_fields = [
            "research_id",
            "thread_id",
            "research_query",
            "research_plan",
            "active_task",
            "task_queue",
            "analyst_summary",
            "report",
            "validation_result",
            "revision_count",
            "research_iteration",
            "execution_status",
            "current_agent",
            "timestamps",
        ]

        for field_name in overwrite_fields:
            hint = hints[field_name]
            if get_origin(hint) is Annotated:
                # If Annotated, the reducer should NOT be operator.add
                # (it could be something else like add_messages for messages)
                import operator
                from typing import get_args

                args = get_args(hint)
                assert args[1] is not operator.add, f"{field_name} should overwrite, not append"

    def test_messages_use_add_messages_reducer(self) -> None:
        """Verify messages field uses LangChain's add_messages reducer."""
        from typing import Annotated, get_args, get_origin, get_type_hints

        from langgraph.graph import add_messages

        hints = get_type_hints(ResearchState, include_extras=True)
        hint = hints["messages"]

        assert get_origin(hint) is Annotated
        args = get_args(hint)
        assert args[1] is add_messages, f"messages reducer should be add_messages but is {args[1]}"

    def test_state_is_typed_dict(self) -> None:
        """Verify ResearchState is a TypedDict (not a regular class)."""

        # TypedDict classes have __annotations__ and are dict subclasses conceptually
        assert hasattr(ResearchState, "__annotations__")
        # A TypedDict creates dicts, not class instances
        state = create_initial_state("r", "t", "q")
        assert isinstance(state, dict)
