"""
Unit tests for the agent base classes and registry.
"""

import pytest

from llmwatch.agents.base import AgentRegistry, AgentResult, BaseAgent


class EchoAgent(BaseAgent):
    name = "echo"
    category = "watcher"

    def run(self, context=None):
        return self._result(data=[{"msg": "hello"}])


class BrokenAgent(BaseAgent):
    name = "broken"
    category = "watcher"

    def run(self, context=None):
        return self._result(errors=["something went wrong"])


class TestAgentResult:
    def test_ok_when_no_errors(self):
        r = AgentResult(agent_name="a", category="watcher")
        assert r.ok() is True

    def test_not_ok_when_errors(self):
        r = AgentResult(agent_name="a", category="watcher", errors=["oops"])
        assert r.ok() is False

    def test_defaults(self):
        r = AgentResult(agent_name="a", category="watcher")
        assert r.data == []
        assert r.errors == []
        assert r.new_sources == []


class TestBaseAgent:
    def test_run_raises_not_implemented(self):
        agent = BaseAgent()
        with pytest.raises(NotImplementedError):
            agent.run()

    def test_result_helper(self):
        agent = EchoAgent()
        result = agent.run()
        assert result.agent_name == "echo"
        assert result.category == "watcher"
        assert result.data == [{"msg": "hello"}]
        assert result.ok() is True

    def test_broken_agent_not_ok(self):
        agent = BrokenAgent()
        result = agent.run()
        assert result.ok() is False
        assert "something went wrong" in result.errors


class TestAgentRegistry:
    def test_register_and_retrieve(self):
        reg = AgentRegistry()
        agent = EchoAgent()
        reg.register(agent)
        assert len(reg) == 1
        assert reg.get("echo") is agent

    def test_filter_by_category(self):
        reg = AgentRegistry()
        reg.register(EchoAgent())
        # create a simple lookup agent
        lookup = BaseAgent()
        lookup.name = "my_lookup"
        lookup.category = "lookup"
        reg.register(lookup)

        watchers = reg.agents(category="watcher")
        lookups = reg.agents(category="lookup")
        assert len(watchers) == 1
        assert len(lookups) == 1
        assert watchers[0].name == "echo"
        assert lookups[0].name == "my_lookup"

    def test_all_agents_no_filter(self):
        reg = AgentRegistry()
        reg.register(EchoAgent())
        all_agents = reg.agents()
        assert len(all_agents) == 1

    def test_get_missing_returns_none(self):
        reg = AgentRegistry()
        assert reg.get("nonexistent") is None

    def test_decorator_registers_class(self):
        reg = AgentRegistry()

        @reg.agent
        class SpecialAgent(BaseAgent):
            name = "special"
            category = "watcher"

            def run(self, context=None):
                return self._result()

        assert reg.get("special") is not None

    def test_duplicate_registration_replaces(self):
        reg = AgentRegistry()
        a1 = EchoAgent()
        a2 = EchoAgent()
        reg.register(a1)
        reg.register(a2)
        assert reg.get("echo") is a2
        assert len(reg) == 1

    def test_repr(self):
        reg = AgentRegistry()
        reg.register(EchoAgent())
        assert "echo" in repr(reg)
