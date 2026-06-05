# Blueprint: ha_socket — filtered `subscribe_entities` instead of the state_changed firehose

**Status:** Backlog / design only. No code yet. Efficiency + log-noise improvement; not
urgent (the current firehose is functionally correct, just wasteful).

**Objective:** Stop receiving every Home Assistant entity's `state_changed` event when
Darkstar only cares about ~a dozen sensors. Subscribe to exactly the monitored entities
so HA filters server-side; temperature/`event.*`/unrelated changes never arrive.

---

## Problem

`backend/ha_socket.py` subscribes to the **global firehose** (line ~221):

```python
{"id": sub_id, "type": "subscribe_events", "event_type": "state_changed"}
```

HA then pushes **every** entity's change. Darkstar filters in code (line ~308):

```python
if entity_id in self.monitored_entities:
    self._handle_state_change(entity_id, new_state)
```

Everything non-monitored is deserialised and dropped. Cost:

- **Log noise** — the debug line at ~303 logs `Received state_changed … monitored=False`
  for anything whose id contains `"ev"`, which also matches `d`**`ev`**`ice_temperature`
  and the `"`**`ev`**`ent."` domain. (Pure noise; `log_level: info` already silences it.)
- **Wasted CPU/network** — JSON-decoding a constant stream of irrelevant events on a busy
  HA. Small per event, but continuous.

Neither affects correctness — monitored values flow correctly. This is an efficiency /
cleanliness refactor.

---

## Current implementation (what changes)

1. `subscribe_events(state_changed)` — the firehose subscription (~217-225).
2. `get_states` — a separate full-snapshot request to prime initial values (~230), then
   filtered to monitored entities (~253-261) plus the initial EV-array emit (~263-294).
3. Receive loop (~297-309): `event.data.{entity_id, new_state}` → `_handle_state_change`.
4. **Stable interface:** `_handle_state_change(entity_id, new_state_dict)` (~315) and the
   `self.monitored_entities` mapping (entity_id → internal key). **This must not change** —
   all downstream parsing (grid/battery/PV/EV/water handlers) keys off it.

---

## Proposed: `subscribe_entities`

Send one filtered subscription using the already-built monitored list:

```python
{"id": sub_id, "type": "subscribe_entities",
 "entity_ids": list(self.monitored_entities.keys())}
```

HA replies with **compressed** event messages (this is the part that needs care):

- **First message — full snapshot** under `event.a` (added/all):
  `{"a": {entity_id: {"s": state, "a": {attrs}, "lc": ts, "lu": ts, "c": ctx}, …}}`
  → replaces `get_states` for these entities.
- **Subsequent — changes** under `event.c` (changed):
  `{"c": {entity_id: {"+": {changed compressed fields}, "-": {removed}}}}`
  The `+` carries only the fields that changed (often just `s`; sometimes `a`).

**Compressed key map:** `s`→state, `a`→attributes, `lc`→last_changed, `lu`→last_updated,
`c`→context. A small `_decompress(compressed) -> {"entity_id", "state", "attributes",
"last_changed", "last_updated"}` adapts each entity back to the dict shape
`_handle_state_change` already expects, so nothing downstream changes.

**Delta handling:** because `c` sends only changed fields, keep a per-entity cache of the
last full (decompressed) state. On `c[entity]["+"]`, merge into the cached state (so an
unchanged `attributes` is preserved — important for status sensors that read a `power`
attribute), then call `_handle_state_change`.

---

## Migration steps

1. Build `entity_ids` from `self.monitored_entities` (already available).
2. Replace `subscribe_events` + `get_states` with `subscribe_entities`. The first `a`
   message is the initial snapshot — move the initial EV-array emit (~263-294) to run
   after processing `a`.
3. Add `_decompress(compressed_state) -> dict` and a `self._entity_state_cache:
   dict[str, dict]` for applying `c` deltas.
4. Adapt the receive loop: branch on `event.get("a")` (snapshot, prime cache + handle all)
   and `event.get("c")` (per-entity delta → merge cache → handle).
5. Leave `_handle_state_change` and every domain handler untouched.
6. **Reconnect:** on each (re)connect, re-subscribe; the fresh `a` snapshot re-primes the
   cache. Clear the cache on disconnect.

---

## Edge cases & risks

- **A parsing bug = stale/missing live data → wrong control decisions.** This is the live
  read path, so the decompress + delta-merge must be unit-tested hard.
- **Attributes preservation:** status/program sensors read `attributes[power]`; a `c`
  delta that omits `a` must keep the cached attributes (the merge handles this).
- **Entities created later** (not yet existing at subscribe time) still arrive via `c`.
- **HA version:** `subscribe_entities` (compressed state) needs HA ≥ 2022.4 — fine for any
  current install.
- **Safety / rollout:** gate behind `ha_socket.subscribe_entities: true` (config) with the
  existing `subscribe_events` path kept as a fallback. Optionally shadow-run: subscribe
  both ways briefly and assert the monitored values match, then flip the default.

## Quick related win (independent, 1-liner)
The debug log at ~303 uses `"ev" in entity_id.lower()`, which matches `device`/`event`.
Replace with a precise check (`entity_id in self.monitored_entities`, or restrict to the
EV plug/soc/power keys) so EV debugging doesn't spam unrelated entities. Safe to do now,
regardless of the subscribe refactor.

## Testing
- Unit-test `_decompress` and the `c`-delta merge with sample `a`/`c` payloads (incl. an
  attributes-only change and a state-only change).
- Reconnect test: cache cleared + re-primed from a new `a`.
- Parity check: same `latest_values` produced as the old `subscribe_events` path for a
  recorded event stream.

## Effort
Medium. The subscription swap is small; the compressed-format parser + delta cache +
reconnect handling + tests are the real work. ~1 focused session. Low value-to-urgency
(noise is already silenced by `log_level: info`), so schedule it when next editing
`ha_socket.py`.
