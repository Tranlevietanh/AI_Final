import time
from collections import defaultdict, deque
import numpy as np
import cv2
from ultralytics import YOLO

# ================= CONFIG =================

CAMERA_FPS = 15
SAMPLE_FPS = 3
STABILITY_TIME = 5.0
NUM_SLOTS = 10

FRAME_STRIDE = CAMERA_FPS // SAMPLE_FPS   
REQUIRED_SAMPLES = SAMPLE_FPS * int(STABILITY_TIME)

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
slot_states = {slot_id: None for slot_id in range(NUM_SLOTS)}
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
    def __init__(self, expected_mapping, required_samples):
        self.expected_mapping = expected_mapping
        self.required_samples = required_samples
        self.history = defaultdict(deque)

    def update(self, slot_id, detected_product):
        hist = self.history[slot_id]
        hist.append(detected_product)

        # Keep only required samples
        if len(hist) > self.required_samples:
            hist.popleft()

        # Not enough evidence yet
        if len(hist) < self.required_samples:
            return None

        most_common = max(set(hist), key=hist.count)

        # Majority vote
        if hist.count(most_common) > len(hist) / 2:
            if most_common is None:
                return None
            expected = self.expected_mapping.get(slot_id)
            return most_common == expected

        return None
# ================= MAIN LOGIC =================

def process_frame(slot_results, product_results, validator):
    # 1. Create a lookup for detected slots
    # format: {slot_id: bbox}
    detected_slots_map = {}
    for box, cls in zip(slot_results.boxes.xyxy, slot_results.boxes.cls):
        detected_slots_map[int(cls)] = box.cpu().numpy()

    # 2. Parse product detections into a list
    products = []
    for box, cls in zip(product_results.boxes.xyxy, product_results.boxes.cls):
        products.append({
            "product_id": int(cls),
            "bbox": box.cpu().numpy()
        })

    decisions = {}

    # 3. Iterate over ALL possible slot IDs (0-9)
    for slot_id in range(NUM_SLOTS):
        assigned_product = None
        
        # Check if the slot model actually found this slot in this frame
        if slot_id in detected_slots_map:
            slot_bbox = detected_slots_map[slot_id]
            best_coverage = 0.0

            # Find the best product match for THIS slot
            for prod in products:
                inter = intersection_area(prod["bbox"], slot_bbox)
                coverage = inter / box_area(prod["bbox"])

                if coverage > best_coverage:
                    best_coverage = coverage
                    assigned_product = prod["product_id"]

            # If the product isn't "inside" enough, it's effectively an empty slot
            if best_coverage < 0.7:
                assigned_product = None
        else:
            # The slot model didn't even see the slot (occlusion/blur)
            # We pass None to validator to record a "missing" state
            assigned_product = None

        # 4. Update the validator for EVERY slot, every frame
        # This prevents the history from "freezing"
        decision = validator.update(slot_id, assigned_product)
        if decision is not None:
            decisions[slot_id] = decision
            
    return decisions


def draw_text_only(frame, slot_states):
    start_x = 20
    start_y = 40
    line_h = 30

    for slot_id in range(10):
        state = slot_states.get(slot_id)

        if state is None:
            text = f"Slot {slot_id}: CHECKING"
            color = (255, 255, 0)  # yellow
        elif state is True:
            text = f"Slot {slot_id}: OK"
            color = (0, 255, 0)
        else:
            text = f"Slot {slot_id}: NG"
            color = (0, 0, 255)

        y = start_y + slot_id * line_h

        cv2.putText(
            frame,
            text,
            (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
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

validator = SlotTemporalValidator(EXPECTED_MAPPING, REQUIRED_SAMPLES)

cap = cv2.VideoCapture(r"C:\Users\VBK computer\Downloads\camera_0_1764404594.mp4")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    if frame_idx % FRAME_STRIDE != 0:
        continue
    slot_results = slot_model(frame, conf=0.5)[0]
    product_results = product_model(frame, conf=0.5)[0]

    decisions = process_frame(slot_results, product_results, validator)
    for slot_id, ok in decisions.items():
        slot_states[slot_id] = ok
    # ---- DRAW RESULTS ----
    draw_text_only(frame, slot_states)
    draw_results (frame, slot_results, decisions)

    cv2.imshow("Slot Inspection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()

