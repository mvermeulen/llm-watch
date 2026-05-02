"""
Agent package for llm-watch.

Agents are divided into three categories:
- watchers: monitor data sources for new/trending LLM activity
- lookup:   look up supplementary information (e.g. arXiv papers)
- reporter: aggregate findings and produce weekly reports
"""

from llmwatch.agents.base import AgentResult, BaseAgent, registry

__all__ = ["BaseAgent", "AgentResult", "registry"]
