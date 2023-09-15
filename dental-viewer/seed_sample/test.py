import cv2
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon


# Image
original = cv2.imread("1502_left.png")
image_filled = original.copy()
image_bordered = original.copy()

# Annotation
annotation = json.load(open("1502_left.json"))

# Opacity
alpha = 0.6

# Colors
colors = {
    "11": (0.122, 0.467, 0.706),
    "12": (0.682, 0.780, 0.910),
    "13": (1.000, 0.498, 0.055),
    "14": (1.000, 0.733, 0.471),
    "15": (0.173, 0.627, 0.173),
    "16": (0.596, 0.875, 0.541),
    "21": (0.839, 0.153, 0.157),
    "22": (1.000, 0.596, 0.588),
    "23": (0.580, 0.404, 0.741),
    "24": (0.773, 0.690, 0.835),
    "25": (0.549, 0.337, 0.294),
    "26": (0.769, 0.612, 0.580),
    "31": (0.890, 0.467, 0.761),
    "32": (0.969, 0.714, 0.824),
    "33": (0.498, 0.498, 0.498),
    "34": (0.780, 0.780, 0.780),
    "35": (0.737, 0.741, 0.133),
    "36": (0.859, 0.859, 0.553),
    "41": (0.090, 0.745, 0.812),
    "42": (0.620, 0.855, 0.898),
    "43": (0.702, 0.886, 0.804),
    "44": (0.992, 0.804, 0.675),
    "45": (0.957, 0.792, 0.894),
    "46": (0.902, 0.961, 0.788),
}

# Hulls
hulls = {}
for tooth in annotation['tooth']:
    # Label
    label = tooth["teeth_num"]
    # Segmentation
    points = np.array(tooth["segmentation"])
    hull = cv2.convexHull(points)
    hulls[label] = hull
hulls = dict(sorted(hulls.items(), key=lambda item: item[0]))

# Cropping
polys = {}
for label in hulls.keys():
    poly = Polygon(np.squeeze(hulls[label]))
    for label_ in hulls.keys():
        if label <= label_:
            continue
        poly_ = Polygon(np.squeeze(hulls[label_]))
        poly = poly.difference(poly_)
    polys[label] = poly
polys = dict(sorted(polys.items(), key=lambda item: item[0], reverse=True))

# Contours
def add_contours(image, image_type=""):
    if image_type == "bordered":
        color = (
            int(255 * 0.173),
            int(255 * 0.627),
            int(255 * 0.173),
        )
        thickness = 5
    elif image_type == "filled":
        color = (
            int(255 * colors[label][2]),
            int(255 * colors[label][1]),
            int(255 * colors[label][0]),
        )
        thickness = cv2.FILLED
    cv2.drawContours(
        image       = image,
        contours    = [coords_3d],
        contourIdx  = -1,
        color       = color,
        thickness   = thickness,
    )

# Texts
def add_texts(image):
    cx = int(np.mean(coords_3d[:, :, 0]) - 50)
    cy = int(np.mean(coords_3d[:, :, 1]) + 30)
    cv2.putText(
        img         = image,
        text        = label,
        org         = (cx, cy),
        fontFace    = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale   = 3,
        color       = (255, 255, 255),
        thickness   = 10,
    )

# Add contours and texts
int_coords = lambda x: np.array(x).round().astype(np.int32)
for label, poly in polys.items():
    coords_3d = np.expand_dims(int_coords(poly.exterior.coords), axis=1)
    # Filled image
    add_contours(image_filled, "filled")
    add_texts(image_filled)
    # Bordered image
    add_contours(image_bordered, "bordered")
    add_texts(image_bordered)

# Filled
image_filled = cv2.addWeighted(
    src1    = image_filled, 
    alpha   = alpha, 
    src2    = original, 
    beta    = 1 - alpha, 
    gamma   = 0,
)

# Save
cv2.imwrite("output_filled.png", image_filled)
cv2.imwrite("output_bordered.png", image_bordered)
