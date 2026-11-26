# ==========================================================
# SCRIPT 1: DATA INLEZEN
# ==========================================================

import json
from pathlib import Path
import pandas as pd


# ----------------------------------------------------------
# INSTELLINGEN
# ----------------------------------------------------------
ROOT = Path(r"D:\Vives\BDD100K")

SPLITS = ["train", "val", "test"]

FOTO_MAP = ROOT / "bdd100k_images_100k" / "100k"
LABEL_MAP = ROOT / "bdd100k_labels"
UITVOER_MAP = ROOT / "CODE" / "Fase2" / "data2"

UITVOER_MAP.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# ELKE SPLIT VERWERKEN
# ----------------------------------------------------------
for split in SPLITS:
    print("Bezig met:", split)

    foto_map = FOTO_MAP / split
    label_map = LABEL_MAP / split

    if not foto_map.exists() or not label_map.exists():
        raise FileNotFoundError(f"Map ontbreekt voor split '{split}': {foto_map} of {label_map}")

    foto_namen = set()
    for foto in foto_map.glob("*.jpg"):
        foto_namen.add(foto.name)

    afbeeldingen = []
    boxes = []

    json_bestanden = list(label_map.glob("*.json"))
    print("  Aantal JSON-bestanden:", len(json_bestanden))

    teller = 0
    for json_bestand in json_bestanden:
        teller += 1
        if teller % 2000 == 0:
            print(f"  {teller} van {len(json_bestanden)} verwerkt")

        with open(json_bestand, "r", encoding="utf-8") as f:
            data = json.load(f)

        naam = Path(data["name"]).stem + ".jpg"

        if naam not in foto_namen:
            continue

        eigenschappen = data.get("attributes", {})
        weer = eigenschappen.get("weather")
        scene = eigenschappen.get("scene")
        tijdstip = eigenschappen.get("timeofday")

        aantal_objecten = 0
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                if "box2d" not in obj:
                    continue
                rechthoek = obj["box2d"]
                aantal_objecten += 1
                boxes.append({
                    "name": naam,
                    "split": split,
                    "weather": weer,
                    "scene": scene,
                    "timeofday": tijdstip,
                    "category": obj.get("category"),
                    "x1": rechthoek["x1"],
                    "y1": rechthoek["y1"],
                    "x2": rechthoek["x2"],
                    "y2": rechthoek["y2"],
                })

        afbeeldingen.append({
            "name": naam,
            "split": split,
            "path": str(foto_map / naam),
            "weather": weer,
            "scene": scene,
            "timeofday": tijdstip,
            "number_of_objects": aantal_objecten,
        })

    afbeeldingen_tabel = pd.DataFrame(afbeeldingen)
    boxes_tabel = pd.DataFrame(boxes)

    if afbeeldingen_tabel.empty:
        raise SystemExit(f"Geen afbeeldingen gevonden voor '{split}'. Controleer de paden.")

    print("  Aantal foto's:", len(afbeeldingen_tabel))
    print("  Aantal objecten:", len(boxes_tabel))
    print("  Verdeling per weer:")
    print(afbeeldingen_tabel["weather"].value_counts(dropna=False).to_string())

    afbeeldingen_tabel.to_pickle(UITVOER_MAP / f"images_{split}.pkl")
    boxes_tabel.to_pickle(UITVOER_MAP / f"boxes_{split}.pkl")
    print("  Opgeslagen!\n")

print("KLAAR")
