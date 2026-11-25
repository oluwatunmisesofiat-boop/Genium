"""Text and perceptual image diffing for packaging change detection."""
from typing import Dict, List


def diff_packaging_specs(old_spec: Dict, new_spec: Dict) -> List[str]:
    changes = []
    for k, v in new_spec.items():
        if k in old_spec and old_spec[k] != v:
            changes.append(f"Attribute '{k}' updated from '{old_spec[k]}' to '{v}'.")
        elif k not in old_spec:
            changes.append(f"Attribute '{k}' newly introduced.")
    return changes
