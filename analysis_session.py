"""In-memory analysis state for the currently loaded custom dataset."""

from dataclasses import dataclass, field


@dataclass
class AnalysisSession:
    """Store explicit custom-analysis selections and completed results."""

    regression: dict | None = None
    classification: dict | None = None
    hypothesis: dict | None = None
    results: dict = field(default_factory=dict)

    def reset(self):
        """Clear configuration and results when the active dataset changes."""
        self.regression = None
        self.classification = None
        self.hypothesis = None
        self.results.clear()

    def remember(self, name, result):
        """Store a completed result only when an analysis returned one."""
        if result is not None:
            self.results[name] = result
        return result
