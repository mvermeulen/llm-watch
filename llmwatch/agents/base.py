"""
Base classes and the global agent registry for llm-watch.

New agents are registered automatically by calling `registry.register(agent)` or
by using the `@registry.agent` decorator.  The orchestrator iterates the registry
to discover every agent that should participate in a run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Structured result returned by every agent."""

    agent_name: str
    category: str  # "watcher", "lookup", or "reporter"
    data: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    new_sources: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.errors


class BaseAgent:
    """
    Abstract base class for all llm-watch agents.

    Subclasses must implement :meth:`run` and set the class attributes
    `name` (a short identifier) and `category` ("watcher", "lookup", or
    "reporter").
    """

    name: str = "base"
    category: str = "base"

    def run(self, context: dict[str, Any] | None = None) -> AgentResult:
        """
        Execute the agent and return an :class:`AgentResult`.

        Parameters
        ----------
        context:
            Optional dict passed in by the orchestrator.  Watchers usually
            ignore it; lookup agents and the reporter use it to access results
            from upstream agents.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.run() is not implemented")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _result(
        self,
        data: list[dict[str, Any]] | None = None,
        errors: list[str] | None = None,
        new_sources: list[str] | None = None,
    ) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            category=self.category,
            data=data or [],
            errors=errors or [],
            new_sources=new_sources or [],
        )


class AgentRegistry:
    """
    A simple registry that maps agent names to agent instances.

    Usage
    -----
    Register an agent::

        registry.register(MyAgent())

    Or use the decorator form::

        @registry.agent
        class MyAgent(BaseAgent):
            ...

    Retrieve agents::

        for agent in registry.agents():
            result = agent.run()
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, agent: BaseAgent) -> BaseAgent:
        """Register *agent* and return it (allows use as a decorator)."""
        if agent.name in self._agents:
            logger.warning("Replacing already-registered agent '%s'", agent.name)
        self._agents[agent.name] = agent
        logger.debug("Registered agent: %s (%s)", agent.name, agent.category)
        return agent

    def agent(self, cls: type[BaseAgent]) -> type[BaseAgent]:
        """Class decorator that instantiates *cls* and registers it."""
        self.register(cls())
        return cls

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def agents(self, category: str | None = None) -> list[BaseAgent]:
        """Return all registered agents, optionally filtered by *category*."""
        agents = list(self._agents.values())
        if category is not None:
            agents = [a for a in agents if a.category == category]
        return agents

    def get(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        names = list(self._agents.keys())
        return f"AgentRegistry({names!r})"


# Module-level singleton registry – all agents should register here.
registry = AgentRegistry()
