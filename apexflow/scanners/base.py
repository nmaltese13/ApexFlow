from __future__ import annotations
import logging
from abc import ABC, abstractmethod

from apexflow.models import Signal
from apexflow.providers import DataProvider, get_flow_client, get_squeeze_client, get_news_client

log = logging.getLogger(__name__)


class BaseScanner(ABC):
    name: str = "base"
    label: str = "Base Scanner"

    def __init__(self, provider: DataProvider):
        self.provider = provider
        # Optional supplementary clients; None when no key is configured.
        self.flow_client = get_flow_client()
        self.squeeze_client = get_squeeze_client()
        self.news_client = get_news_client()

    @abstractmethod
    def scan(self, universe: list[str]) -> list[Signal]:
        ...

    def scan_safe(self, universe: list[str]) -> list[Signal]:
        try:
            return self.scan(universe)
        except Exception as e:
            log.exception("%s crashed: %s", self.name, e)
            return []
