"""Display names: config identifiers (hip_abduction, jump_landing_static, knee_act) -> Title Case text."""
import re

_SMALL = {"to", "of", "at", "and", "or", "in", "on", "for"}      # kept lower-case inside a name


def pretty(name: str) -> str:
    """'hip_abduction' -> 'Hip Abduction', 'knee_act' -> 'Knee Actuator', 'sit_to_stand' -> 'Sit to Stand'."""
    name = re.sub(r"_act$", " actuator", str(name))
    words = [w for w in re.split(r"[_\s]+", name.strip()) if w]
    return " ".join(w if (any(c.isdigit() for c in w) or w.isupper() or (k and w in _SMALL)) else w[:1].upper() + w[1:]
                    for k, w in enumerate(words))
