import time
from collections import defaultdict, deque
import numpy as np
import cv2
from ultralytics import YOLO
from state import StateController, process_from_slot_states
from cam_view import CameraReader


SAMPLE_FPS = 5 #FPS mỗi lần chạy mô hình YOLO
STABILITY_TIME = 5.0 #Độ dài queue để xét liệu object có thực sự nằm trong slot
NUM_SLOTS = 10
INFER_INTERVAL = 1.0 / SAMPLE_FPS #Thời gian giữa mỗi lần chạy YOLO
last_infer_time = 0.0

STATE_TO_CAMERA = {
    0: 0,  # camera 0 checks state 0 -> 1
    1: 1,  # camera 1 checks state 1 -> 2
    2: 2,  # camera 2 checks state 2 -> 3
    3: 3,  # camera 3 checks state 3 -> 4
} #mapping giữa vị trí camera và state

CAMERA_URLS = [
    "rtsp://admin:CPSFLT@192.168.1.160:554/ch1/main",
    "rtsp://admin:DVCLRQ@192.168.1.116:554/ch1/main",
    "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main",
    "rtsp://admin:KXILGD@192.168.1.152:554/ch1/main"
] #địa chỉ camera

REQUIRED_SAMPLES = SAMPLE_FPS * int(STABILITY_TIME) #Số lượng mẫu (frame) trong queue xét obj có thực sự trong slot (5*5=25)

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
} #mapping giữa slot và object

slot_states_per_camera = {
    cam_id: {slot_id: None for slot_id in range(NUM_SLOTS)}
    for cam_id in range(len(CAMERA_URLS))
} #State của các slot theo camera

def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xA >= xB or yA >= yB:
        return 0.0

    return (xB - xA) * (yB - yA) #Hàm tính diện tích giao giữa box của object và box của slot

def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1]) #Hàm tính diện tích của box 

def product_inside_slot(product_box, slot_box, thresh=0.7):
    inter = intersection_area(product_box, slot_box)
    return inter / box_area(product_box) >= thresh #Object được tính là trong slot trong 1 frame khi diện tích intersection >= 0.7 diện tích box


class SlotTemporalValidator:
    def __init__(self, expected_mapping, required_samples):
        self.expected_mapping = expected_mapping
        self.required_samples = required_samples
        self.history = defaultdict(deque) #khai báo mapping giữa object và slot, số sample của queue, queue

    def update(self, slot_id, detected_product):
        hist = self.history[slot_id]
        hist.append(detected_product)  #Thêm vào queue mỗi lượt detect (1 frame)

        # Keep only required samples
        if len(hist) > self.required_samples:
            hist.popleft() #Khi queue quá số mẫu (25) sẽ pop mẫu cũ nhất

        # Not enough evidence yet
        if len(hist) < self.required_samples:
            return None #Khi queue chưa đủ số mẫu (25) chưa đưa ra quyết định là obj có thật sự nằm trong slot không

        most_common = max(set(hist), key=hist.count) #lấy kết quả phổ biến nhất trong queue

        # Majority vote
        if hist.count(most_common) > len(hist) / 2: #nếu kết quả phổ biến nhất chiếm hơn 50% queue
            if most_common is None:
                return None #trường hợp không có object
            expected = self.expected_mapping.get(slot_id)
            return most_common == expected #trường hợp có object -> So sánh với mapping: Trả về True/False

        return None

validators = {
    cam_id: SlotTemporalValidator(EXPECTED_MAPPING, REQUIRED_SAMPLES)
    for cam_id in range(len(CAMERA_URLS))
} #Các class kiểm tra cho các cam

def process_frame(slot_results, product_results, validator):

    detected_slots_map = {}
    for box, cls in zip(slot_results.boxes.xyxy, slot_results.boxes.cls):
        detected_slots_map[int(cls)] = box.cpu().numpy() #Trả lại kết quả box của slot theo cls

    products = []
    for box, cls in zip(product_results.boxes.xyxy, product_results.boxes.cls):
        products.append({
            "product_id": int(cls),
            "bbox": box.cpu().numpy()
        }) #Trả lại kết quả box của product theo cls

    decisions = {}

    for slot_id in range(NUM_SLOTS):
        assigned_product = None

        if slot_id in detected_slots_map:
            slot_bbox = detected_slots_map[slot_id]
            best_coverage = 0.0

            for prod in products:
                inter = intersection_area(prod["bbox"], slot_bbox)
                coverage = inter / box_area(prod["bbox"])

                if coverage > best_coverage:
                    best_coverage = coverage
                    assigned_product = prod["product_id"]

            if best_coverage < 0.7:
                assigned_product = None
        else:
            assigned_product = None

        decision = validator.update(slot_id, assigned_product) #update queue kiểm tra xem object có thực sự trong slot không
        decisions[slot_id] = decision #viết vào một list theo slot_id
            
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
        ) #viết state của các slot trên màn hình 
def draw_results(frame, slot_results, decisions):
    for box, cls in zip(slot_results.boxes.xyxy, slot_results.boxes.cls):
        slot_id = int(cls)
        x1, y1, x2, y2 = map(int, box.cpu().numpy())

        if slot_id in decisions:
            state = decisions[slot_id]

            if state is True:
                color = (0, 255, 0)
                label = f"Slot {slot_id}: OK"

            elif state is False:
                color = (0, 0, 255)
                label = f"Slot {slot_id}: NG"

            else:  # state is None
                color = (255, 255, 0)
                label = f"Slot {slot_id}: CHECKING"

        else:
            # Slot not evaluated this frame
            color = (255, 255, 0)
            label = f"Slot {slot_id}: CHECKING"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        ) #viết quyết định của các slot theo từng frame

slot_model = YOLO(r"C:\Users\VBK computer\Downloads\slot.pt")
product_model = YOLO(r"C:\Users\VBK computer\Downloads\product.pt") #nạp mô hình


# State controller (locks per-state once processed)
state_controller = StateController(initial_state=0) #khai báo class quản lý trạng thái

# cap = cv2.VideoCapture(r"D:\projects\ai&app\cky\AI_Final\camera_0_1764404513.mp4")
reader = CameraReader(CAMERA_URLS, width=1280, height=720) #khai báo class lấy dữ liệu từ cam
reader.start()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
latest_slot_results = {}
latest_decisions = {}

try:
    while True:
        latest_advanced = False
        frames = reader.get_frames() #lấy frame từ cam. frames là một list gồm 4 vị trí frames[0], [1],.. tương ứng với 4 cam
        now = time.time()

        current_state = state_controller.state #state hiện tại (0, 1, 2, 3, 4)
        active_cam = STATE_TO_CAMERA.get(current_state) #Cam tương ứng với state

        if active_cam is not None: 
            frame = frames[active_cam] #Lấy frame của cam tương ứng với state

            # ---------- INFERENCE (ONE CAMERA, LOW FPS) ----------
            if frame is not None and now - last_infer_time >= INFER_INTERVAL: #nếu đã đủ thời gian ngắt quãng giữa các thời gian xử lý, tiếp tục xử lý
                last_infer_time = now
                frame_small = cv2.resize(frame, None, fx=0.5, fy=0.5) #resize từ 1280 về 640

                slot_results = slot_model(frame_small, conf=0.5)[0] #lấy kết quả từ model slot với confidence score = 0.5
                product_results = product_model(frame_small, conf=0.5)[0] #lấy kết quả từ model object với confidence score = 0.5

                decisions = process_frame(
                    slot_results,
                    product_results,
                    validators[active_cam]
                ) #xử lý slot và object -> đưa ra quyết định trong frame này object có trong slot không

                slot_states_per_camera[active_cam].update(decisions) #quyết định này đưa vào trong queue 25 quyết định
                latest_slot_results[active_cam] = slot_results
                latest_decisions[active_cam] = decisions

                # ---------- STATE ADVANCE (ONLY ACTIVE CAMERA) ----------

                latest_advanced = False
                if active_cam is not None:
                    _, latest_advanced, _ = process_from_slot_states(
                        state_controller,
                        slot_states_per_camera[active_cam]
                    ) #từ các state của các slot suy luận ra state hiện tại của mô hình: 0, 1, 2 hay 3

                    if latest_advanced:
                        validators[active_cam].history.clear() #khi chuyển state -> chuyển cam, và xóa lịch sử queue của các slot để xử lý thông tin mới

        slot_results = latest_slot_results.get(active_cam)
        if frame is not None and slot_results is not None:
            frame_small = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
            draw_results(
                frame_small,
                latest_slot_results.get(active_cam),
                latest_decisions.get(active_cam, {})
            ) #viết ra màn hình

            draw_text_only(frame_small, slot_states_per_camera[active_cam]) #viết ra màn hình

            status_text = f"State:{state_controller.state} | {'Accepted' if latest_advanced else 'Waiting'}"
            cv2.putText(frame_small, status_text, (20, frame_small.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2) #viết ra màn hình 

            cv2.imshow("Slot Inspection", frame_small)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    reader.stop()
    cv2.destroyAllWindows()


# cap.release()
# cv2.destroyAllWindows()
