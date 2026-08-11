import tkinter as tk
from tkinter import font as tkfont
from typing import Tuple, List

# ── Constants ────────────────────────────────────────────────────────────────
NUM_DECKS = 4
CARDS_PER_DECK = 52
TOTAL_CARDS = NUM_DECKS * CARDS_PER_DECK  # 208

# Each rank starts with 4 copies per deck × 4 decks = 16
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
INITIAL_COUNT = {r: NUM_DECKS * 4 for r in RANKS}

# Hi-Lo values
HILO = {
    "2": +1, "3": +1, "4": +1, "5": +1, "6": +1,
    "7":  0, "8":  0, "9":  0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}

# Bet recommendation thresholds (based on true count)
def bet_advice(tc: float) -> Tuple[str, str]:
    if tc >= 4:
        return "BET MAX 🔥", "#ff4444"
    elif tc >= 3:
        return "BET HIGH ⬆", "#ff8800"
    elif tc >= 2:
        return "BET UP  ↑", "#ffcc00"
    elif tc >= 1:
        return "BET SLIGHTLY UP", "#aaffaa"
    elif tc <= -1:
        return "BET LOW ↓", "#aaaaff"
    else:
        return "BET TABLE MIN", "#cccccc"


# ── App ───────────────────────────────────────────────────────────────────────
class CardCounter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Blackjack Card Counter — 4-Deck Shoe")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")

        self._init_state()
        self._build_ui()
        self._refresh()

    # ── State ─────────────────────────────────────────────────────────────────
    def _init_state(self):
        self.remaining = dict(INITIAL_COUNT)   # cards left per rank
        self.running_count = 0
        self.cards_seen = 0

    def _reset(self):
        self._init_state()
        self._refresh()

    # ── Core logic ────────────────────────────────────────────────────────────
    @property
    def cards_left(self) -> int:
        return TOTAL_CARDS - self.cards_seen

    @property
    def decks_remaining(self) -> float:
        return max(self.cards_left / CARDS_PER_DECK, 0.5)  # floor at 0.5

    @property
    def true_count(self) -> float:
        return self.running_count / self.decks_remaining

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        PAD = 14
        BG = "#1a1a2e"
        PANEL = "#16213e"
        ACCENT = "#0f3460"

        title_font  = tkfont.Font(family="Helvetica", size=16, weight="bold")
        label_font  = tkfont.Font(family="Helvetica", size=11)
        big_font    = tkfont.Font(family="Helvetica", size=22, weight="bold")
        card_font   = tkfont.Font(family="Helvetica", size=14, weight="bold")
        small_font  = tkfont.Font(family="Helvetica", size=9)

        # ── Title bar ─────────────────────────────────────────────────────────
        tk.Label(self, text="♠ Blackjack Card Counter ♥", font=title_font,
                 bg=BG, fg="#e2e2e2").pack(pady=(PAD, 4))
        tk.Label(self, text=f"4-Deck Shoe  ·  Hi-Lo System",
                 font=small_font, bg=BG, fg="#888888").pack(pady=(0, PAD))

        # ── Stats row ─────────────────────────────────────────────────────────
        stats_frame = tk.Frame(self, bg=BG)
        stats_frame.pack(padx=PAD, pady=(0, PAD), fill="x")

        def stat_box(parent, label, var, col, fg="#ffffff"):
            f = tk.Frame(parent, bg=PANEL, padx=16, pady=10,
                         relief="flat", bd=0)
            f.grid(row=0, column=col, padx=6, sticky="nsew")
            parent.columnconfigure(col, weight=1)
            tk.Label(f, text=label, font=small_font, bg=PANEL,
                     fg="#000000").pack()
            lbl = tk.Label(f, textvariable=var, font=big_font, bg=PANEL,
                     fg=fg)
            lbl.pack()
            return lbl

        self.var_running = tk.StringVar()
        self.var_true    = tk.StringVar()
        self.var_decks   = tk.StringVar()
        self.var_seen    = tk.StringVar()

        self.lbl_running = stat_box(stats_frame, "Running Count", self.var_running, 0, "#e2e2e2")
        self.lbl_true    = stat_box(stats_frame, "True Count",    self.var_true,    1, "#f5c518")
        self.lbl_decks   = stat_box(stats_frame, "Decks Left",    self.var_decks,   2, "#90cdf4")
        self.lbl_seen    = stat_box(stats_frame, "Cards Seen",    self.var_seen,    3, "#90cdf4")

        # ── Bet advice ────────────────────────────────────────────────────────
        self.bet_frame = tk.Frame(self, bg=PANEL, padx=12, pady=10)
        self.bet_frame.pack(padx=PAD, pady=(0, PAD), fill="x")
        self.var_bet  = tk.StringVar()
        self.lbl_bet  = tk.Label(self.bet_frame, textvariable=self.var_bet,
                                 font=tkfont.Font(family="Helvetica", size=14, weight="bold"),
                                 bg=PANEL, fg="#cccccc")
        self.lbl_bet.pack()

        # ── Card buttons ──────────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(padx=PAD, pady=(0, 6))

        self.card_btns = {}
        self.card_rem_vars = {}

        colors = {
            "+1": ("#000000", "#2a2a2a"),   # black (2-6)
            " 0": ("#000000", "#2a2a2a"),   # black (7-9)
            "-1": ("#000000", "#2a2a2a"),   # black (10,J,Q,K,A)
        }

        def hilo_label(r):
            v = HILO[r]
            return f"{v:+d}" if v != 0 else " 0"

        for i, rank in enumerate(RANKS):
            hl = hilo_label(rank)
            bg_dark, bg_lit = colors[hl]

            col_frame = tk.Frame(btn_frame, bg=BG)
            col_frame.grid(row=0, column=i, padx=4)

            # Remaining count above button
            rem_var = tk.StringVar()
            self.card_rem_vars[rank] = rem_var
            tk.Label(col_frame, textvariable=rem_var, font=small_font,
                     bg=BG, fg="#000000").pack()

            # Card button
            btn = tk.Button(
                col_frame, text=f"{rank}\n{hl}",
                font=card_font, width=4, height=2,
                bg=bg_dark, fg="black", activebackground=bg_lit,
                activeforeground="black", relief="flat", cursor="hand2",
                command=lambda r=rank: self._press(r)
            )
            btn.pack()
            self.card_btns[rank] = btn

        # ── Undo / Reset row ──────────────────────────────────────────────────
        ctrl_frame = tk.Frame(self, bg=BG)
        ctrl_frame.pack(padx=PAD, pady=(10, PAD))

        tk.Button(ctrl_frame, text="↩  Undo Last Card",
                  font=label_font, bg=ACCENT, fg="white",
                  activebackground="#1a4a80", relief="flat", cursor="hand2",
                  padx=14, pady=6,
                  command=self._undo_card).pack(side="left", padx=8)

        tk.Button(ctrl_frame, text="🔄  New Shoe",
                  font=label_font, bg="#4a1a1a", fg="white",
                  activebackground="#7a2a2a", relief="flat", cursor="hand2",
                  padx=14, pady=6,
                  command=self._reset).pack(side="left", padx=8)

        # ── Legend ────────────────────────────────────────────────────────────
        leg = tk.Frame(self, bg=BG)
        leg.pack(pady=(0, PAD))
        legend_items = [
            ("■ 2–6  +1 (favor player)", "#27ae60"),
            ("■ 7–9   0 (neutral)",      "#555577"),
            ("■ 10–A -1 (favor dealer)", "#c0392b"),
        ]
        for txt, clr in legend_items:
            tk.Label(leg, text=txt, font=small_font, bg=BG,
                     fg=clr).pack(side="left", padx=10)

        # History stack for undo
        self.history: List[str] = []

    # ── Override press to track history ──────────────────────────────────────
    def _press(self, rank: str):
        if self.remaining[rank] <= 0:
            return
        self.remaining[rank] -= 1
        self.running_count += HILO[rank]
        self.cards_seen += 1
        self.history.append(rank)
        self._refresh()

    def _undo_card(self):
        if not self.history:
            return
        rank = self.history.pop()
        self.remaining[rank] += 1
        self.running_count -= HILO[rank]
        self.cards_seen -= 1
        self._refresh()

    # ── Refresh display ───────────────────────────────────────────────────────
    def _refresh(self):
        rc = self.running_count
        tc = self.true_count
        dl = self.decks_remaining
        cs = self.cards_seen

        # Running count coloring
        rc_color = "#4caf50" if rc > 0 else "#ef5350" if rc < 0 else "#e2e2e2"
        self.var_running.set(f"{rc:+d}")
        self.lbl_running.config(fg=rc_color)

        # True count
        tc_color = "#f5c518" if tc >= 1 else "#ef5350" if tc <= -1 else "#e2e2e2"
        self.var_true.set(f"{tc:+.2f}")
        self.lbl_true.config(fg=tc_color)

        self.var_decks.set(f"{dl:.1f}")
        self.var_seen.set(str(cs))

        # Bet advice
        advice, adv_color = bet_advice(tc)
        self.var_bet.set(f"Bet Advice:  {advice}")
        self.lbl_bet.config(fg=adv_color)

        # Card buttons remaining
        for rank in RANKS:
            left = self.remaining[rank]
            self.card_rem_vars[rank].set(f"{left}/16")
            # Dim button if rank is exhausted
            btn = self.card_btns[rank]
            if left == 0:
                btn.config(state="disabled", bg="#333333")
            else:
                btn.config(state="normal")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = CardCounter()
    app.mainloop()