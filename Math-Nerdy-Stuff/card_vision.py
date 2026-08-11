"""
Camera Card Counter  -  Hi-Lo  -  3-Deck Shoe
pip install opencv-python numpy
Run: python3 card_vision.py
"""

import cv2
import numpy as np
import time
import mss

# ── Hi-Lo constants ───────────────────────────────────────────────────────────
NUM_DECKS      = 3
CARDS_PER_DECK = 52
TOTAL_CARDS    = NUM_DECKS * CARDS_PER_DECK

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
HILO  = {
    "2": +1, "3": +1, "4": +1, "5": +1, "6": +1,
    "7":  0, "8":  0, "9":  0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}

RANK_MAP = {          # what we match in the cropped pip region
    "A": ["A"],
    "2": ["2"],
    "3": ["3"],
    "4": ["4"],
    "5": ["5"],
    "6": ["6"],
    "7": ["7"],
    "8": ["8"],
    "9": ["9"],
    "10": ["10", "0"],
    "J": ["J"],
    "Q": ["Q"],
    "K": ["K"],
}

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
C_BG      = (30,  20,  15)
C_GREEN   = (80, 200, 80)
C_RED     = (60,  60, 220)
C_YELLOW  = (20, 200, 220)
C_WHITE   = (230, 230, 230)
C_DIM     = (120, 120, 120)
C_PANEL   = (50,  40,  35)

# ── State ─────────────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.running_count = 0
        self.cards_seen    = 0
        self.remaining     = {r: NUM_DECKS * 4 for r in RANKS}
        self.history       = []
        self.last_seen     = {}      # rank -> timestamp, for cooldown
        self.cooldown      = 2.5     # seconds before same rank fires again
        self.paused        = False

    @property
    def cards_left(self):
        return TOTAL_CARDS - self.cards_seen

    @property
    def decks_remaining(self):
        return max(self.cards_left / CARDS_PER_DECK, 0.5)

    @property
    def true_count(self):
        return self.running_count / self.decks_remaining

    def log_card(self, rank):
        now = time.time()
        if rank in self.last_seen and now - self.last_seen[rank] < self.cooldown:
            return False                   # still in cooldown
        if self.remaining[rank] <= 0:
            return False
        self.remaining[rank]  -= 1
        self.running_count    += HILO[rank]
        self.cards_seen       += 1
        self.history.append((rank, now))
        self.last_seen[rank]   = now
        return True

    def undo(self):
        if not self.history:
            return
        rank, _ = self.history.pop()
        self.remaining[rank]  += 1
        self.running_count    -= HILO[rank]
        self.cards_seen       -= 1
        self.last_seen.pop(rank, None)

    def reset(self):
        self.__init__()


# ── Card detection ────────────────────────────────────────────────────────────
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def find_card_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8000 or area > 120000:
            continue
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            cards.append(approx)
    return cards


def warp_card(frame, contour):
    pts = contour.reshape(4, 2).astype("float32")
    # Order: top-left, top-right, bottom-right, bottom-left
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype="float32")
    dst = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], dtype="float32")
    M   = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(frame, M, (200, 300))


def read_rank(warped):
    """
    Crop the top-left pip region and use OCR-lite template matching via
    contour shape — or fall back to a simple digit/letter recognizer using
    the pixel pattern of the rank area.
    """
    # Grab top-left corner (rank is printed there)
    corner = warped[5:60, 5:40]
    gray   = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    _, bw  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the largest contour in the pip region = rank character
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Use Hu moments to classify the rank character
    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 30:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    char_roi   = bw[y:y+h, x:x+w]

    return classify_rank(char_roi, w, h)


def classify_rank(roi, w, h):
    """
    Classify by aspect ratio + pixel density heuristics.
    This is a lightweight approach — works well for standard card fonts.
    """
    if roi is None or roi.size == 0:
        return None

    ratio  = w / max(h, 1)
    pixels = cv2.countNonZero(roi) / max(roi.size, 1)

    # Resize to fixed size for moment comparison
    resized = cv2.resize(roi, (20, 30))
    m       = cv2.moments(resized)
    hu      = cv2.HuMoments(m).flatten()

    # Very wide = 10 (two digits)
    if ratio > 1.2:
        return "10"

    # Tall and narrow = A, J, Q, K, or number
    # Use pixel density and aspect ratio bands
    if ratio < 0.55:
        # Narrow — likely 1, I-shaped: A or J
        if pixels > 0.45:
            return "A"
        return "J"

    if 0.55 <= ratio < 0.75:
        if pixels < 0.30:
            return "7"
        if pixels < 0.38:
            return "2"
        return "Q"

    if 0.75 <= ratio < 0.95:
        if pixels > 0.55:
            return "8"
        if pixels > 0.45:
            return "6"
        if pixels > 0.38:
            return "9"
        return "3"

    # ~square
    if pixels > 0.55:
        return "5"
    if pixels > 0.45:
        return "4"
    return "K"


# ── HUD drawing ───────────────────────────────────────────────────────────────
def bet_advice(tc):
    if tc >= 4:  return "BET MAX",          C_RED
    if tc >= 3:  return "BET HIGH",         (0, 100, 255)
    if tc >= 2:  return "BET UP",           C_YELLOW
    if tc >= 1:  return "BET SLIGHTLY UP",  C_GREEN
    if tc <= -1: return "BET LOW",          (200, 150, 80)
    return              "TABLE MIN",        C_DIM


def draw_hud(frame, state, detected_ranks):
    h, w = frame.shape[:2]

    # Semi-transparent panel on the right
    panel_x = w - 280
    overlay  = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (20, 15, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def text(msg, x, y, color=C_WHITE, scale=0.6, thick=1):
        cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    px = panel_x + 10
    y  = 30

    text("CARD COUNTER", px, y, C_YELLOW, 0.65, 2);  y += 28
    text("Hi-Lo  |  3-Deck Shoe", px, y, C_DIM, 0.45); y += 30

    cv2.line(frame, (panel_x, y), (w, y), (80, 70, 60), 1); y += 18

    rc = state.running_count
    rc_col = C_GREEN if rc > 0 else C_RED if rc < 0 else C_WHITE
    text("Running Count", px, y, C_DIM, 0.45);          y += 20
    text(f"  {rc:+d}", px, y, rc_col, 0.9, 2);          y += 36

    tc = state.true_count
    tc_col = C_YELLOW if tc >= 1 else (200, 150, 80) if tc <= -1 else C_WHITE
    text("True Count", px, y, C_DIM, 0.45);             y += 20
    text(f"  {tc:+.2f}", px, y, tc_col, 0.9, 2);        y += 36

    text(f"Decks left: {state.decks_remaining:.1f}", px, y, C_WHITE, 0.5); y += 22
    text(f"Cards seen: {state.cards_seen}", px, y, C_WHITE, 0.5);          y += 30

    cv2.line(frame, (panel_x, y), (w, y), (80, 70, 60), 1); y += 18

    advice, adv_col = bet_advice(tc)
    text("BET ADVICE", px, y, C_DIM, 0.45);             y += 20
    text(advice, px, y, adv_col, 0.7, 2);               y += 36

    cv2.line(frame, (panel_x, y), (w, y), (80, 70, 60), 1); y += 18

    # Last 5 cards seen
    text("Last cards:", px, y, C_DIM, 0.45);            y += 20
    recent = [r for r, _ in state.history[-5:]][::-1]
    for rank in recent:
        col = C_GREEN if HILO[rank] > 0 else C_RED if HILO[rank] < 0 else C_DIM
        text(f"  {rank}  ({HILO[rank]:+d})", px, y, col, 0.55); y += 22

    y = h - 90
    cv2.line(frame, (panel_x, y), (w, y), (80, 70, 60), 1); y += 14

    status = "PAUSED  (P)" if state.paused else "SCANNING"
    st_col = (80, 80, 200) if state.paused else C_GREEN
    text(status, px, y, st_col, 0.55, 2);               y += 26
    text("[U] Undo  [R] Reset  [Q] Quit", px, y, C_DIM, 0.38)

    # Highlight detected cards on main frame
    for rank in detected_ranks:
        col = C_GREEN if HILO[rank] > 0 else C_RED if HILO[rank] < 0 else C_DIM
        text(f">> {rank} detected", 10, 40, col, 0.8, 2)


def draw_card_outlines(frame, contours, state, newly_logged):
    for cnt in contours:
        col = (0, 255, 0) if newly_logged else (200, 200, 60)
        cv2.drawContours(frame, [cnt], -1, col, 2)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera index.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    state   = State()
    print("Camera card counter running.")
    print("Keys:  P = pause/resume | U = undo | R = reset shoe | Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_ranks = []
        newly_logged   = False

        if not state.paused:
            thresh   = preprocess(frame)
            contours = find_card_contours(thresh)

            for cnt in contours:
                warped = warp_card(frame, cnt)
                rank   = read_rank(warped)
                if rank and rank in HILO:
                    detected_ranks.append(rank)
                    logged = state.log_card(rank)
                    if logged:
                        newly_logged = True

            draw_card_outlines(frame, contours, state, newly_logged)

        draw_hud(frame, state, detected_ranks)
        cv2.imshow("Card Counter  |  Q=quit  P=pause  U=undo  R=reset", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            state.paused = not state.paused
        elif key == ord('u'):
            state.undo()
        elif key == ord('r'):
            state.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()