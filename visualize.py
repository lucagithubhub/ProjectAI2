import cv2
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. DATA INLEZEN
# ==========================================

df = pd.read_pickle(r"D:\Vives\BDD100K\CODE\Fase2\data2\images_train.pkl")
boxes_df = pd.read_pickle(r"D:\Vives\BDD100K\CODE\Fase2\data2\boxes_train.pkl")


# ==========================================
# 2. EERSTE AFBEELDING SELECTEREN
# ==========================================

row = df.iloc[0]

image_name = row["name"]
image_path = row["path"]

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(
        f"Afbeelding kon niet worden ingelezen: {image_path}"
    )

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ==========================================
# 3. BOUNDING BOXES OPHALEN
# ==========================================

image_boxes = boxes_df[boxes_df["name"] == image_name]

for _, box in image_boxes.iterrows():

    x1 = int(box["x1"])
    y1 = int(box["y1"])
    x2 = int(box["x2"])
    y2 = int(box["y2"])

    category = str(box["category"])

    # Rechthoek tekenen
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    # Objectcategorie tonen
    cv2.putText(
        img,
        category,
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )


# ==========================================
# 4. AFBEELDING TONEN
# ==========================================

plt.figure(figsize=(12, 7))

plt.imshow(img)

plt.title(
    f"Weer: {row['weather']} | "
    f"Scène: {row['scene']} | "
    f"Tijdstip: {row['timeofday']}"
)

plt.axis("off")
plt.show()
