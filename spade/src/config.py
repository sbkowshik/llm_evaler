"""Configuration settings for the SPADE evaluation system."""

from typing import Dict, List

# SPADE Settings
DEFAULT_FALSE_FAILURE_RATE_THRESHOLD = 0.40  # 40%, as mentioned in the SPADE paper
MIN_GRADES_REQUIRED = 10  # Minimum number of responses needed for assertion selection

# Default evaluation criteria
DEFAULT_SPADE_CRITERIA = [
    {
        "name": "Concise explanation",
        "description": "Response is concise and to the point",
        "implementation_type": "llm",
    },
    {
        "name": "Straightforward explanation",
        "description": "Explanation is straightforward and easy to understand",
        "implementation_type": "llm",
    },
] 