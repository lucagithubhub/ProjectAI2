# ==========================================================
# SCRIPT 2: EXPERIMENT OPBOUWEN
# ==========================================================

import shutil
from pathlib import Path
import pandas as pd


# ----------------------------------------------------------
# INSTELLINGEN
# ----------------------------------------------------------
DATA_MAP = Path(r"D:\Vives\BDD100K\CODE\Fase2\data2")
UITVOER_MAP = Path(r"D:\Vives\BDD100K\CODE\Fase2\yolo_data2")

BREEDTE = 1280
HOOGTE = 720

KLASSEN = {
    "person": 0,
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 2,
    "bus": 2,
}

CONDITIES = ["clear", "rain", "fog", "night"]

SAMPLE_PER_CONDITIE = None


# ----------------------------------------------------------
# HULPFUNCTIES
# ----------------------------------------------------------
def laad_split(split):
    afbeeldingen = pd.read_pickle(DATA_MAP / f"images_{split}.pkl")
    boxes = pd.read_pickle(DATA_MAP / f"boxes_{split}.pkl")

    boxes = boxes[boxes["category"].isin(KLASSEN)].copy()
    boxes["klasse"] = boxes["category"].map(KLASSEN)

    afbeeldingen = afbeeldingen[afbeeldingen["name"].isin(boxes["name"])].copy()
    return afbeeldingen, boxes


def kies_conditie(afbeeldingen, conditie):
    weer = afbeeldingen["weather"]
    tijd = afbeeldingen["timeofday"]

    if conditie == "clear":
        return afbeeldingen[(weer == "clear") & (tijd == "daytime")]
    if conditie == "rain":
        return afbeeldingen[(weer == "rainy") & (tijd == "daytime")]
    if conditie == "fog":
        return afbeeldingen[weer == "foggy"]
    if conditie == "night":
        return afbeeldingen[(weer == "clear") & (tijd == "night")]


def exporteer(afbeeldingen, boxes, doelmap):
    foto_uit = doelmap / "images"
    label_uit = doelmap / "labels"
    foto_uit.mkdir(parents=True, exist_ok=True)
    label_uit.mkdir(parents=True, exist_ok=True)

    boxes_per_foto = dict(list(boxes.groupby("name")))

    for naam, pad in zip(afbeeldingen["name"], afbeeldingen["path"]):
        shutil.copy2(pad, foto_uit / naam)

        regels = []
        for b in boxes_per_foto[naam].itertuples():
            x1 = min(max(b.x1, 0), BREEDTE)
            x2 = min(max(b.x2, 0), BREEDTE)
            y1 = min(max(b.y1, 0), HOOGTE)
            y2 = min(max(b.y2, 0), HOOGTE)

            breedte = x2 - x1
            hoogte = y2 - y1
            if breedte <= 1 or hoogte <= 1:
                continue

            x_midden = (x1 + x2) / 2 / BREEDTE
            y_midden = (y1 + y2) / 2 / HOOGTE
            regels.append(
                f"{b.klasse} {x_midden:.6f} {y_midden:.6f} "
                f"{breedte / BREEDTE:.6f} {hoogte / HOOGTE:.6f}"
            )

        (label_uit / f"{Path(naam).stem}.txt").write_text("\n".join(regels))


# ----------------------------------------------------------
# STAP 1: data inlezen
# ----------------------------------------------------------
print("STAP 1: data inlezen")
train_foto, train_boxes = laad_split("train")
val_foto, val_boxes = laad_split("val")
test_foto, test_boxes = laad_split("test")
print("  train:", len(train_foto), "| val:", len(val_foto), "| test:", len(test_foto))


# ----------------------------------------------------------
# STAP 2: testsets per conditie samenstellen
# ----------------------------------------------------------
print("\nSTAP 2: testsets per conditie")
testsets = {}
for conditie in CONDITIES:
    pool = kies_conditie(test_foto, conditie)
    print(f"  {conditie}: {len(pool)} beelden beschikbaar")

    if len(pool) == 0:
        print("  WAARSCHUWING: geen beelden, conditie wordt overgeslagen")
        continue

    if SAMPLE_PER_CONDITIE is not None:
        aantal = min(SAMPLE_PER_CONDITIE, len(pool))
        pool = pool.sample(aantal, random_state=0)
        print(f"    -> willekeurig {aantal} gekozen")

    testsets[conditie] = pool


# ----------------------------------------------------------
# STAP 3: alles naar YOLO-mappen schrijven
# ----------------------------------------------------------
print("\nSTAP 3: bestanden plaatsen in", UITVOER_MAP)

if UITVOER_MAP.exists():
    shutil.rmtree(UITVOER_MAP)

for conditie, fotos in testsets.items():
    print("  test/" + conditie)
    exporteer(fotos, test_boxes, UITVOER_MAP / "test" / conditie)

print("  train")
exporteer(train_foto, train_boxes, UITVOER_MAP / "train")

print("  val")
exporteer(val_foto, val_boxes, UITVOER_MAP / "val")


# ----------------------------------------------------------
# STAP 4: samenvatting van de testsets
# ----------------------------------------------------------
print("\nSAMENVATTING: aantal objecten per conditie")
namen = ["pedestrian", "cyclist", "vehicle"]
overzicht = []
for conditie, fotos in testsets.items():
    objecten = test_boxes[test_boxes["name"].isin(fotos["name"])]
    telling = objecten["klasse"].value_counts()
    rij = {"conditie": conditie, "afbeeldingen": len(fotos)}
    for nummer in range(3):
        rij[namen[nummer]] = int(telling.get(nummer, 0))
    overzicht.append(rij)
print(pd.DataFrame(overzicht).to_string(index=False))

print("\nKLAAR")
