# ==========================================================
# SCRIPT 3: TRAINEN EN TESTEN
# ==========================================================

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


# ----------------------------------------------------------
# INSTELLINGEN
# ----------------------------------------------------------
ROOT = Path(r"D:\Vives\BDD100K\CODE\Fase2\yolo_data2")

GETRAIND_MODEL = None

EPOCHS = 30
PATIENCE = 5
IMGSZ = 640
BATCH = 16
DEVICE = 0
WORKERS = 4

CONDITIES = ["clear", "rain", "fog", "night"]
KLASSEN = ["pedestrian", "cyclist", "vehicle"]


# ----------------------------------------------------------
# HULPFUNCTIES
# ----------------------------------------------------------
def maak_yaml(bestand, map_pad, train_pad, val_pad):
    tekst = f"path: {map_pad.as_posix()}\n"
    tekst += f"train: {train_pad}\n"
    tekst += f"val: {val_pad}\n"
    tekst += "names:\n"
    for nummer, naam in enumerate(KLASSEN):
        tekst += f"  {nummer}: {naam}\n"
    bestand.write_text(tekst)
    return bestand


def tel_fotos(map_pad):
    if not map_pad.exists():
        return 0
    return len(list(map_pad.glob("*.jpg")))


# ----------------------------------------------------------
# HOOFDPROGRAMMA
# ----------------------------------------------------------
def main():
    # --- Stap 1: tellen hoeveel foto's er zijn ---
    print("Aantal foto's per map:")
    aantal_train = tel_fotos(ROOT / "train" / "images")
    aantal_val = tel_fotos(ROOT / "val" / "images")
    print("  train:", aantal_train)
    print("  val:  ", aantal_val)

    aantal_test = {}
    for conditie in CONDITIES:
        aantal_test[conditie] = tel_fotos(ROOT / "test" / conditie / "images")
        print(f"  test/{conditie}:", aantal_test[conditie])

    if GETRAIND_MODEL is None and (aantal_train == 0 or aantal_val == 0):
        raise SystemExit("De train- of val-map is leeg. Draai eerst script 2.")

    actieve_condities = [c for c in CONDITIES if aantal_test[c] > 0]

    # --- Stap 2: trainen (of een bestaand model laden) ---
    if GETRAIND_MODEL is None:
        yaml_train = maak_yaml(ROOT / "train.yaml", ROOT, "train/images", "val/images")

        model = YOLO("yolov8n.pt")
        model.train(
            data=str(yaml_train),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            patience=PATIENCE,
            workers=WORKERS,
            seed=0,
            deterministic=True,
            project=str(ROOT / "runs"),
            name="train",
        )
        model = YOLO(str(model.trainer.best))
    else:
        model = YOLO(GETRAIND_MODEL)

    # --- Stap 3: testen per weerconditie ---
    resultaten = []
    for conditie in actieve_condities:
        test_map = ROOT / "test" / conditie
        yaml_test = maak_yaml(ROOT / f"test_{conditie}.yaml", test_map, "images", "images")

        uitslag = model.val(
            data=str(yaml_test),
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            conf=0.001,
            iou=0.6,
            plots=False,
            verbose=False,
        )

        for plek, klasse_nummer in enumerate(uitslag.box.ap_class_index):
            resultaten.append({
                "conditie": conditie,
                "klasse": KLASSEN[int(klasse_nummer)],
                "AP50": float(uitslag.box.ap50[plek]),
                "precision": float(uitslag.box.p[plek]),
                "recall": float(uitslag.box.r[plek]),
                "aantal_beelden": aantal_test[conditie],
            })

        resultaten.append({
            "conditie": conditie,
            "klasse": "alle",
            "mAP50": float(uitslag.box.map50),
            "precision": float(uitslag.box.mp),
            "recall": float(uitslag.box.mr),
            "aantal_beelden": aantal_test[conditie],
        })

    # --- Stap 4: resultaten tonen en opslaan ---
    tabel_lang = pd.DataFrame(resultaten)
    tabel_lang.to_csv(ROOT / "results.csv", index=False)

    tabel = tabel_lang.pivot(index="klasse", columns="conditie", values="AP50")
    tabel = tabel[actieve_condities]
    print("\nAP50 per klasse en conditie:")
    print(tabel.round(3))

    tabel.plot(kind="bar", figsize=(9, 5))
    plt.ylabel("AP50")
    plt.title("Detectienauwkeurigheid per klasse en weerconditie")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ROOT / "results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
