"""Snapshot local G1+hands assets without modifying the source repository."""

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/firedust/monkeyBars_CRL/assets/g1_brachiation/menagerie/unitree_g1")
DEST = ROOT / "assets/g1_dex3"
DEST.mkdir(parents=True, exist_ok=True)
root = ET.parse(SOURCE / "g1_with_hands.xml").getroot()
files = ["g1_with_hands.xml", "scene_with_hands.xml", "LICENSE", "README.md"]
files += sorted({"assets/" + m.attrib["file"] for m in root.findall("./asset/mesh")})
records = []
for name in files:
    src, dst = SOURCE / name, DEST / name
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Refusing to overwrite modified asset: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    records.append({"file": name, "sha256": digest, "bytes": src.stat().st_size})
report = {
    "source": str(SOURCE),
    "model_name": root.attrib["model"],
    "identity_note": "Three fingers and 7 joints per hand; Dex3-1 candidate, exact hardware revision not verified.",
    "files": records,
    "actuator_count_xml": len(root.findall("./actuator/*")),
}
(DEST / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(f"Snapshotted {len(records)} files; {report['actuator_count_xml']} actuators")
