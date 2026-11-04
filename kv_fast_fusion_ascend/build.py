"""Which BFF code a run actually executed.

Bumped with every change that a run is meant to test, and printed on the decode's apply-mode line,
so a box that pulled only part of a change (2026-11-01: an "exact search" run used a two-turn-old
build) is visible on line 1 of decode-0.txt instead of after the run is analysed.
"""

BFF_BUILD = "2026-11-04b"
