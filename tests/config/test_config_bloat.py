"""Regression: comment-token duplication in template_aware_merge (config bloat).

The live incident: four list keys outside ARRAY_UNIQUE_KEYS (deferrable_loads,
phase_observer.devices, executor.excess_pv.sinks, executor.ev_surplus.chargers) each
carried a template comment block. Binding the round-trip-loaded user list into the
template tree kept the template's comment token AND the user copy — +1 copy of every
block per write, ~7 kB/write, a 428 kB config.yaml, and the June onboarding-reset when
a non-atomic write of that file was interrupted.

The fix strips ruamel comment attachments from the USER tree before merging: comments
can only ever come from the template, exactly once, no matter how many write cycles run.
"""

import io

from ruamel.yaml import YAML

from backend.config_migration import template_aware_merge

TEMPLATE = """\
system:
  timezone: "Europe/Stockholm"

# ---------------------------------------------------------------
# Deferrable loads: appliances Darkstar may schedule (dishwasher,
# washing machine, ...). This block is template documentation that
# must appear EXACTLY ONCE no matter how many times config is saved.
# ---------------------------------------------------------------
deferrable_loads: []

executor:
  # Sinks soak surplus PV when the battery is nearly full.
  # Another template-owned comment block.
  excess_pv:
    sinks: []
"""


def _load(yaml: YAML, text: str):
    return yaml.load(io.StringIO(text))


def _dump(yaml: YAML, data) -> str:
    buf = io.StringIO()
    yaml.dump(data, buf)
    return buf.getvalue()


def _one_cycle(yaml: YAML, user_text: str) -> str:
    """One save cycle: fresh template + round-trip-loaded user config -> merged dump."""
    template = _load(yaml, TEMPLATE)
    user = _load(yaml, user_text)
    template_aware_merge(template, user)
    return _dump(yaml, template)


def test_repeated_merge_cycles_do_not_duplicate_template_comments():
    yaml = YAML()
    yaml.preserve_quotes = True

    user_text = TEMPLATE.replace(
        "deferrable_loads: []",
        "deferrable_loads:\n  - id: washer\n    power_sensor: sensor.washer_power",
    ).replace("sinks: []", "sinks:\n      - id: villavagn_ac\n        enabled: false")

    sizes = []
    for _ in range(6):
        user_text = _one_cycle(yaml, user_text)
        sizes.append(len(user_text))

    # The bloat bug grew the file every cycle; fixed output is size-stable.
    assert sizes[-1] == sizes[1], f"config grows per write cycle: {sizes}"
    # The template comment blocks appear exactly once.
    assert user_text.count("EXACTLY ONCE") == 1, user_text
    assert user_text.count("soak surplus PV") == 1, user_text
    # And the user's values survived every cycle.
    assert "id: washer" in user_text
    assert "id: villavagn_ac" in user_text


def test_user_values_win_and_template_comments_survive_once():
    yaml = YAML()
    template = _load(yaml, TEMPLATE)
    user = _load(
        yaml,
        "system:\n  timezone: 'Europe/Helsinki'\ndeferrable_loads:\n  - id: washer\n",
    )
    template_aware_merge(template, user)
    out = _dump(yaml, template)
    assert "Europe/Helsinki" in out
    assert out.count("EXACTLY ONCE") == 1


def test_atomic_write_through_symlink_updates_the_target(tmp_path):
    """Regression: the add-on's config.yaml is a SYMLINK (/app/config.yaml ->
    /config/darkstar/config.yaml). os.replace on the link path replaces the LINK
    with the temp file — content lands in an ephemeral local file, the real
    (bind-mounted) config is never written, and every save is lost at restart
    while reporting success. _write_config must resolve the link first."""
    from ruamel.yaml import YAML

    from backend.config_migration import _write_config

    real = tmp_path / "mounted" / "config.yaml"
    real.parent.mkdir()
    real.write_text("system:\n  timezone: 'Europe/Stockholm'\nmarker: old\n")
    link = tmp_path / "app" / "config.yaml"
    link.parent.mkdir()
    link.symlink_to(real)

    yaml = YAML()
    new_cfg = yaml.load("system:\n  timezone: 'Europe/Stockholm'\nmarker: new\n")

    assert _write_config(link, new_cfg, yaml, strict_validation=False) is True

    # The REAL file got the new content...
    assert "marker: new" in real.read_text()
    # ...and reading via the link agrees (link still points at the real file).
    assert "marker: new" in link.read_text()
