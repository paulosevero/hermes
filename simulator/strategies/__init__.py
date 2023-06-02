"""Automatic Python configuration file."""
__version__ = "1.0.0"

# Baseline maintenance strategies
from .greedy_least_batch import greedy_least_batch
from .salus import salus
from .lamp import lamp
from .nsgaii import nsgaii

# Proposed maintenance strategy
from .hermes import hermes
from .hermes_v2 import hermes_v2
from .hermes_v3 import hermes_v3