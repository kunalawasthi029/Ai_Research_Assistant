"""Pydantic schemas for research plans and tasks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SubQuestion(BaseModel):
    """A sub-question derived from the main research query."""

    model_config = ConfigDict(str_strip_whitespace=True, hide_input_in_errors=True)
    question_id: str = Field(
        min_length=1, max_length=80, description="Unique identifier for this sub-question"
    )
    question: str = Field(min_length=1, max_length=1000, description="The sub-question text")
    priority: int = Field(ge=1, le=5, description="Priority level (1=highest, 5=lowest)")
    expected_source_types: list[str] = Field(
        default_factory=list,
        description="Expected source types: web, academic, wikipedia, calculated",
    )


class ResearchTask(BaseModel):
    """A concrete research task for the Researcher agent to execute."""

    model_config = ConfigDict(str_strip_whitespace=True, hide_input_in_errors=True)
    task_id: str = Field(min_length=1, max_length=80, description="Unique identifier for this task")
    question_id: str = Field(
        min_length=1, max_length=80, description="ID of the parent sub-question"
    )
    description: str = Field(min_length=1, max_length=2000, description="What to research")
    search_queries: list[str] = Field(
        min_length=1, max_length=5, description="Suggested search queries for this task"
    )
    required_evidence: str = Field(
        min_length=1, max_length=2000, description="What evidence is needed to answer this task"
    )
    expected_source_types: list[str] = Field(
        default_factory=list, description="Which source types to target"
    )
    completion_criteria: str = Field(
        min_length=1, max_length=2000, description="How to know when this task is complete"
    )
    priority: int = Field(ge=1, le=5, description="Priority level (1=highest)")
    status: Literal["pending", "in_progress", "completed", "partial", "skipped"] = Field(
        default="pending", description="pending|in_progress|completed|partial|skipped"
    )


class ResearchPlan(BaseModel):
    """Structured research plan produced by the Planner agent."""

    model_config = ConfigDict(str_strip_whitespace=True, hide_input_in_errors=True)
    research_objective: str = Field(
        min_length=1, max_length=4000, description="Refined research objective"
    )
    sub_questions: list[SubQuestion] = Field(
        min_length=1, max_length=10, description="Decomposed sub-questions"
    )
    research_tasks: list[ResearchTask] = Field(
        min_length=1, max_length=20, description="Concrete research tasks"
    )
    scope: Literal["narrow", "balanced", "broad"] = Field(
        default="balanced", description="narrow|balanced|broad"
    )
    estimated_complexity: Literal["low", "moderate", "high"] = Field(
        default="moderate", description="low|moderate|high"
    )
    notes: str | None = Field(default=None, description="Additional planning notes")

    @model_validator(mode="after")
    def linked_unique_tasks(self) -> ResearchPlan:
        """Reject duplicate IDs, orphan tasks and questions with no research work."""
        questions = {question.question_id for question in self.sub_questions}
        tasks = {task.task_id for task in self.research_tasks}
        covered = {task.question_id for task in self.research_tasks}
        if len(questions) != len(self.sub_questions) or len(tasks) != len(self.research_tasks):
            raise ValueError("Plan identifiers must be unique")
        if questions != covered:
            raise ValueError(
                "Every task must reference an existing question and every question needs a task"
            )
        if any(
            not query.strip() or len(query) > 1000
            for task in self.research_tasks
            for query in task.search_queries
        ):
            raise ValueError("Search queries must be nonempty and bounded")
        return self
