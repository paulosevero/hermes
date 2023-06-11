"""Automatic Python configuration file."""
__version__ = "1.0.0"

# Baseline maintenance strategies
from .greedy_least_batch import greedy_least_batch
from .salus import salus
from .lamp import lamp

# NSGA-II
from .nsgaii import nsgaii
from .nsgaii_v2 import nsgaii_v2
from .nsgaii_v3 import nsgaii_v3
from .nsgaii_v4 import nsgaii_v4

# Proposed maintenance strategy
from .hermes import hermes
from .hermes_v2 import hermes_v2
from .lamp_v2 import lamp_v2
