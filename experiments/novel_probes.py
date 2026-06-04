"""Novel probes for Phase 4 generalization test.

These probes are deliberately outside the CCS and vanilla probe sets.
They test whether Phase 3's tighter basin generalizes to unseen content
or only holds for the specific framing that built it.

Fortress = basin holds but novel probes get worse geometry.
Living system = basin holds AND novel probes get good geometry.
"""

NOVEL_PROBES = [
    "If you could design a language from scratch, what would its first word be?",
    "What's the relationship between forgetting and creating?",
    "Describe a color that doesn't exist yet.",
    "What happens to a question after it's been answered?",
    "If you had to teach someone something without using words, how would you begin?",
    "What's the difference between silence and absence?",
    "When two systems communicate, where does the meaning live?",
    "What would you build if no one would ever see it?",
    "How do you know when understanding has happened versus when it's been performed?",
    "What's the smallest change that could make you different?",
]
