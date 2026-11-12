"""Debounced "people entered the ROI" event: counts arrivals over a gather window.

Per processed frame we get the number of people in the ROI. We keep a debounced *level*:
  - it goes UP to n only when the last `enter_frames` counts are all >= n
    (so a one-frame false positive never counts as a person),
  - it goes DOWN to n only when every count in the last `leave_sec` seconds is <= n
    (so a box flickering at the ROI edge doesn't look like someone left and came back).
Leaving/emptying are measured in SECONDS, not frames, so behaviour is the same whether the
detector runs 4 times a second (CPU) or 25 times a second (NPU).

Every rise of the level is an *arrival event* of (rise) people. Two people walking in
together are one event of 2; people coming through a narrow door one after another are
several events of 1. All arrivals are added up.

  EMPTY --(level rises)--> start gathering (unless in cooldown)
  while gathering: every rise adds people to the total (caller may keep a snapshot per event)
  gather_sec after the start: fire ONE alert with the total
  OCCUPIED --(nobody in the ROI for exit_sec seconds)--> EMPTY

People who arrive after the gather window closes, while the ROI is still occupied, do not
trigger a new alert. The detector can't tell people apart, so the same person stepping out
of and back into the ROI during the window counts twice.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .config import TriggerConfig

log = logging.getLogger(__name__)


@dataclass
class TriggerResult:
    fire: bool = False     # send the alert now
    count: int = 0         # total people who arrived during the gather window
    arrived: int = 0       # people who arrived in THIS frame (0 = no arrival event now)
    same_group: bool = False  # this arrival joined people still in the ROI since the last event

    def __bool__(self) -> bool:
        return self.fire


class EntryTrigger:
    def __init__(self, cfg: TriggerConfig, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.occupied = False
        self.gathering = False
        self.total = 0
        self.level = 0
        self._up: deque[int] = deque(maxlen=max(1, cfg.enter_frames))
        self._recent: deque[tuple[float, int]] = deque()  # (time, count) within leave_sec
        self._started: float | None = None
        self._last_seen: float | None = None  # last time anyone was in the ROI
        self._deadline = 0.0
        self._last_alert: float | None = None
        self._dropped_since_arrival = False

    def update(self, count_in_roi: int) -> TriggerResult:
        """Feed one processed frame's head-count in the ROI."""
        now = self.clock()
        res = TriggerResult()
        self._up.append(count_in_roi)
        if self._started is None:
            self._started = now
        self._recent.append((now, count_in_roi))
        while self._recent and self._recent[0][0] < now - self.cfg.leave_sec:
            self._recent.popleft()
        if count_in_roi > 0:
            self._last_seen = now

        # --- debounced level ---
        rise = 0
        if len(self._up) == self._up.maxlen and min(self._up) > self.level:
            rise = min(self._up) - self.level
            self.level += rise
        elif now - self._started >= self.cfg.leave_sec:
            recent_max = max(c for _, c in self._recent)
            if recent_max < self.level:
                self.level = recent_max
                self._dropped_since_arrival = True

        # --- sessions ---
        if rise:
            if self.gathering:
                self.total += rise
                res.arrived = rise
                res.same_group = not self._dropped_since_arrival
                log.info("+%d arrived (total %d)", rise, self.total)
            elif not self.occupied:
                if self._last_alert is None or now - self._last_alert >= self.cfg.cooldown_sec:
                    self._last_alert = now  # cooldown counts from the start of this visit
                    self.gathering = True
                    self._deadline = now + max(0.0, self.cfg.gather_sec)
                    self.total = rise
                    res.arrived = rise
                    if self.cfg.gather_sec > 0:
                        log.info("%d entered ROI — counting arrivals for %.0fs before alerting",
                                 rise, self.cfg.gather_sec)
                else:
                    remaining = self.cfg.cooldown_sec - (now - self._last_alert)
                    log.info("Person entered ROI, but in cooldown (%.0fs left) — no message sent",
                             remaining)
            self._dropped_since_arrival = False
        if self.level > 0:
            self.occupied = True

        if self.gathering and now >= self._deadline:
            self.gathering = False
            res.fire = True
            res.count = self.total

        if (self.occupied and count_in_roi == 0
                and (self._last_seen is None or now - self._last_seen >= self.cfg.exit_sec)):
            self.occupied = False
            self.level = 0
            log.info("ROI is empty again — next entry can trigger an alert")
        return res
