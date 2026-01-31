import time
from collections import defaultdict, deque
import numpy as np
import cv2
from ultralytics import YOLO

# ================= CONFIG =================

STABILITY_TIME = 5.0  # seconds

# Expected product per slot (by class ID)
EXPECTED_MAPPING = {
    0: 0,
    1: 0, 
    2: 0,
    3: 1,
    4: 2,
    5: 6,
    6: 7,
    7: 3,
    8: 4,
    9: 5
}

# ================= HELPERS =================

def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xA >= xB or yA >= yB:
        return 0.0

    return (xB - xA) * (yB - yA)

def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def product_inside_slot(product_box, slot_box, thresh=0.7):
    inter = intersection_area(product_box, slot_box)
    return inter / box_area(product_box) >= thresh

# ================= TEMPORAL SLOT TRACKER =================

class SlotTemporalValidator:
    def __init__(self, expected_mapping, stability_time, min_samples=5):
        self.expected_mapping = expected_mapping
        self.stability_time = stability_time
        self.min_samples = min_samples
        self.history = defaultdict(deque)
        self.confirmed = {}

    def update(self, slot_id, detected_product, timestamp):
        hist = self.history[slot_id]
        hist.append((timestamp, detected_product))

        # Remove old entries
        while hist and timestamp - hist[0][0] > self.stability_time:
            hist.popleft()

        products = [p for _, p in hist]
        if len(products) < self.min_samples:
            return None

        most_common = max(set(products), key=products.count)

        # Majority agreement
        if products.count(most_common) > len(products) / 2:
            expected = self.expected_mapping.get(slot_id)
            return most_common == expected  # ✅ FIXED

        return None
# ================= MAIN LOGIC =================

def process_frame(slot_results, product_results, validator):
    timestamp = time.time()

    # Parse slot detections
    slots = []
    for box, cls in zip(slot_results.boxes.xyxy, slot_results.boxes.cls):
        slots.append({
            "slot_id": int(cls),
            "bbox": box.cpu().numpy()
        })

    # Parse product detections
    products = []
    for box, cls in zip(product_results.boxes.xyxy, product_results.boxes.cls):
        products.append({
            "product_id": int(cls),
            "bbox": box.cpu().numpy()
        })

    # Slot → product assignment
    slot_assignment = {}

    for slot in slots:
        slot_id = slot["slot_id"]
        slot_bbox = slot["bbox"]

        assigned_product = None
        best_coverage = 0.0

        for prod in products:
            inter = intersection_area(prod["bbox"], slot_bbox)
            coverage = inter / box_area(prod["bbox"])

            if coverage > best_coverage:
                best_coverage = coverage
                assigned_product = prod["product_id"]

        if best_coverage < 0.7:
            assigned_product = None

        slot_assignment[slot_id] = assigned_product

    # Temporal validation
    decisions = {}
    confirmed_slots = set()

    for slot_id, product_id in slot_assignment.items():
        if product_id is None:
            continue

        decision = validator.update(slot_id, product_id, timestamp)
        if decision is not None:
            decisions[slot_id] = decision
            confirmed_slots.add(slot_id)
            validator.history[slot_id].clear()
    return decisions

def draw_results(frame, slot_results, decisions):
    for box, cls in zip(slot_results.boxes.xyxy, slot_results.boxes.cls):
        slot_id = int(cls)
        x1, y1, x2, y2 = map(int, box.cpu().numpy())

        if slot_id in decisions:
            ok = decisions[slot_id]
            color = (0, 255, 0) if ok else (0, 0, 255)
            label = f"Slot {slot_id}: {'OK' if ok else 'NG'}"
        else:
            color = (255, 255, 0)
            label = f"Slot {slot_id}: checking"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

slot_model = YOLO(r"C:\Users\VBK computer\Downloads\slot_mosaic.pt")
product_model = YOLO(r"C:\Users\VBK computer\Downloads\product.pt")

validator = SlotTemporalValidator(EXPECTED_MAPPING, STABILITY_TIME)

cap = cv2.VideoCapture(r"C:\Users\VBK computer\Downloads\lower_full_2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    slot_results = slot_model(frame, conf=0.5)[0]
    product_results = product_model(frame, conf=0.5)[0]

    decisions = process_frame(slot_results, product_results, validator)

    # ---- DRAW RESULTS ----
    draw_results(frame, slot_results, decisions)

    cv2.imshow("Slot Inspection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()

