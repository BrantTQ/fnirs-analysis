# run_enem_blocks.py
# PsychoPy >= 2022.2 recommended

# ---- prefs BEFORE imports that create windows ----
from psychopy import prefs
prefs.general['measureFrameRate'] = False
prefs.general['shutdownKey'] = 'escape'
prefs.general['autoLog'] = False

from psychopy import visual, core, event, gui, data, logging
from psychopy.hardware import keyboard
import csv, os, time, random, sys, json
import re

logging.console.setLevel(logging.ERROR)

PUNCT_KEYMAP = {
    "comma": ",", "period": ".", "minus": "-", "slash": "/",
    "backslash": "\\", "apostrophe": "'", "quote": "\"",
    "semicolon": ";", "equal": "=", "space": " ",
    "leftbracket": "[", "rightbracket": "]",
    "leftbrace": "{", "rightbrace": "}",
}

# ===== paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ===== config =====
# NOTE: Set USE_FNIRS=True and USE_LSL=True to actually send markers
USE_FNIRS = True
USE_LSL = True
USE_TTL = False
LSL_STREAM_NAME = "Trigger"        # corrected name
LSL_STREAM_TYPE = "Markers"
PARALLEL_PORT_ADDR = 0x0378

TRIGGER_MAP = {
    # session / utility
    "EXP_START": 1,
    "EXP_END": 2,
    "PING": 3,

    # task flow
    "Q_TEXT_ON": 11,
    "Q_FULL_ON": 12,     # kept for compatibility (unused here)
    "BUTTON_CLICK": 13,
    "Q_STEM_ON": 14,     # question (stem) is shown
    "Q_OPTIONS_ON": 15,  # options are shown
    "ANS_A": 21, "ANS_B": 22, "ANS_C": 23, "ANS_D": 24, "ANS_E": 25,
    "BLK_ON": 91, "BLK_OFF": 92, "BLOCK_REST": 93, "ITI": 99,

    # questionnaire
    "QUESTIONNAIRE_ON": 71, "QUESTIONNAIRE_OFF": 72,
}

MIN_ITI_SECS = 3.0
MAX_ITI_SECS = 5.0
FULLSCREEN = True
WIN_SIZE = [1920, 1100]
BLOCK_DURATION_SECS = 7 * 60

# ---- files ----
QUESTIONS_JSON = r"C:\Users\thiago-ext\Documents\FNIRS\psychopy\filtered_questions.json"
SOCIO_Q_FILE = os.path.join(BASE_DIR, "demographics_questions.json")  # adjust if needed

BLOCKS_PER_TYPE = 5
QUESTIONS_PER_BLOCK = 3
N_BLOCKS = BLOCKS_PER_TYPE * 2

STEM_TEXT_HEIGHT = 26
GEN_TEXT_HEIGHT = 24
OPTION_TEXT_HEIGHT = 20

RUN_QUESTIONNAIRE_BEFORE = True
INSERT_QUESTIONNAIRE_AFTER_BLOCK = None

# ===== core/window =====
global_clock = core.MonotonicClock()

print("Initializing window (this may take a moment)...")
try:
    win = visual.Window(
        size=WIN_SIZE, fullscr=FULLSCREEN, color=[1, 1, 1], units="pix",
        waitBlanking=False, autoLog=False
    )
    win.recordFrameIntervals = False
    print("Window created successfully!")
except Exception as e:
    print(f"ERROR creating window: {e}")
    print("Trying alternative window creation...")
    try:
        win = visual.Window(size=WIN_SIZE, fullscr=False, color=[-1,-1,-1], units="pix")
        print("Window created with fallback method!")
    except Exception as e2:
        print(f"FATAL ERROR: Could not create window: {e2}")
        sys.exit(1)

kb = keyboard.Keyboard()
mouse = event.Mouse(win=win)

# ===== layout (fixed left margin; no overlap) =====
SCREEN_W, SCREEN_H = win.size
LEFT_X = -SCREEN_W//2 + 60   # visible left margin
WRAP_PIX = int(SCREEN_W * 0.86)

TEXT_Y     = 330
QUESTION_Y = 80
OPTIONS_Y0 =  10
OPTION_STEP = -70
BUTTON_Y   = -320

question_text = visual.TextStim(
    win, text="", color="black", height=STEM_TEXT_HEIGHT,
    wrapWidth=WRAP_PIX, alignText='left', pos=(LEFT_X, TEXT_Y),
    anchorHoriz='left', anchorVert='center'
)
question_itself = visual.TextStim(
    win, text="", color="black", height=GEN_TEXT_HEIGHT,
    wrapWidth=WRAP_PIX, alignText='left', pos=(LEFT_X, QUESTION_Y),
    anchorHoriz='left', anchorVert='center'
)
msg_text = visual.TextStim(win, text="", color="black", height=GEN_TEXT_HEIGHT, pos=(0, 0))

button_show = visual.Rect(
    win, width=360, height=64, fillColor=[-0.2,-0.2,-0.2],
    lineColor="black", pos=(0, BUTTON_Y)
)
button_show_lbl = visual.TextStim(
    win, text="Show question", color="black", height=GEN_TEXT_HEIGHT, pos=(0, BUTTON_Y)
)

opt_texts, opt_boxes = [], []
for i in range(5):
    y = OPTIONS_Y0 + i*OPTION_STEP
    t = visual.TextStim(
        win, text="", color="black", height=OPTION_TEXT_HEIGHT,
        wrapWidth=WRAP_PIX, alignText='left', pos=(LEFT_X + 54, y),
        anchorHoriz='left', anchorVert='center'
    )
    opt_texts.append(t)
    b = visual.Rect(
        win, width=46, height=46, fillColor=[-0.2,-0.2,-0.2],
        lineColor="black", pos=(LEFT_X + 18, y)
    )
    opt_boxes.append(b)

# ===== markers I/O =====
class NoMarkerOutlet:
    def push_sample(self, *args, **kwargs): pass

outlet = NoMarkerOutlet()
pport = None

if USE_FNIRS:
    if USE_LSL:
        try:
            from pylsl import StreamInfo, StreamOutlet
            # int32 numeric markers
            info = StreamInfo(LSL_STREAM_NAME, LSL_STREAM_TYPE, 1, 0, 'int32',
                              f'psychopy_{int(time.time())}')
            outlet = StreamOutlet(info)
            print("[LSL] Marker stream created.")
            # numeric startup sample (no-op)
            outlet.push_sample([0])
        except Exception as e:
            print("[LSL] ERROR:", e); USE_LSL = False
    if USE_TTL:
        try:
            from psychopy import parallel
            pport = parallel.ParallelPort(address=PARALLEL_PORT_ADDR)
            pport.setData(0); print("[TTL] Parallel port ready at", hex(PARALLEL_PORT_ADDR))
        except Exception as e:
            print("[TTL] ERROR:", e); USE_TTL = False

def send_marker(code_name: str):
    """Send a numeric marker over LSL and/or TTL; return (name, int_code, local_time_for_logs)."""
    t = global_clock.getTime()
    code_int = TRIGGER_MAP.get(code_name, 0)

    # LSL: push integer codes (let LSL timestamp internally)
    if USE_FNIRS and USE_LSL:
        try:
            outlet.push_sample([code_int])
        except Exception as e:
            print("[LSL] send error:", e)

    # TTL: send integer code
    if USE_FNIRS and USE_TTL and pport is not None and code_int > 0:
        try:
            pport.setData(code_int); core.wait(0.005); pport.setData(0)
        except Exception as e:
            print("[TTL] send error:", e)

    return code_name, code_int, t

# ===== logging =====
exp_info = {"participant": "", "session": "001"}
dlg = gui.DlgFromDict(exp_info, title="ENEM fNIRS (Blocks)")
if not dlg.OK: core.quit()

timestamp = time.strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"enem_blocks_{exp_info['participant']}_{timestamp}.csv")
try:
    log_f = open(log_path, "w", newline="", encoding="utf-8")
except Exception as e:
    print(f"[LOG] Could not open log file: {e}")
    try:
        msg_text.text = "Error: cannot open log file. Check write permissions."
        msg_text.draw(); win.flip(); core.wait(2.0)
    except Exception: pass
    sys.exit(1)

print(f"[LOG] Writing to: {os.path.abspath(log_path)}")
log_writer = csv.writer(log_f)
log_writer.writerow([
    "t_abs","phase","block","trial_idx_in_block","question_number","question_year",
    "question_type","question_field","marker_name","marker_code",
    "rt_from_phase","choice","correct","button_click_time","option_view_time","note"
])
log_f.flush()

def log_event(phase, block_label, trial_idx, q_data, marker_name, code, t_phase_start,
              choice="", correct="", button_click_t="", opt_view_t="", note=""):
    t_abs = global_clock.getTime()
    rt = (t_abs - t_phase_start) if t_phase_start is not None else ""
    if isinstance(q_data, dict):
        q_num=q_data.get("question_number",""); q_year=q_data.get("year","")
        q_type=q_data.get("type",""); q_field=q_data.get("field","")
    else:
        q_num=q_year=q_type=q_field=""
    log_writer.writerow([
        f"{t_abs:.6f}", phase, block_label, trial_idx, q_num, q_year, q_type, q_field,
        marker_name, code, f"{rt}", choice, correct, button_click_t, opt_view_t, note
    ])
    log_f.flush()

# ===== helpers =====
def wait_secs_draw(secs, drawlist=None):
    if secs <= 0: return
    t0 = core.Clock()
    while t0.getTime() < secs:
        if drawlist:
            for stim in drawlist: stim.draw()
        win.flip(); core.wait(0.001)

def show_message(text, key_to_continue="space"):
    msg_text.text = text
    kb.clearEvents()
    while True:
        msg_text.draw(); win.flip(); core.wait(0.001)
        keys = kb.getKeys([key_to_continue,'escape'], waitRelease=False)
        if keys:
            if keys[0].name == 'escape': cleanup_and_quit()
            break

def wait_for_mouse_release():
    # Debounce: wait until all mouse buttons are released
    while any(mouse.getPressed()):
        win.flip(); core.wait(0.01)

def debounce_after_trigger():
    # Short refractory period after a reveal to avoid double-advance with held keys
    event.clearEvents(); kb.clearEvents(); mouse.clickReset()
    wait_for_mouse_release()
    core.wait(0.12)

def cleanup_and_quit():
    try:
        if not log_f.closed: log_f.flush(); log_f.close()
    except Exception: pass
    try: win.close()
    except Exception: pass
    core.quit()

# ===== data load =====
def load_questions():
    if not os.path.exists(QUESTIONS_JSON):
        print(f"ERROR: Questions file not found: {QUESTIONS_JSON}"); cleanup_and_quit()
    with open(QUESTIONS_JSON,'r',encoding='utf-8') as f:
        all_questions = json.load(f)
    concrete = [q for q in all_questions if q.get("type")=="concrete"]
    abstract = [q for q in all_questions if q.get("type")=="abstract"]
    random.shuffle(concrete); random.shuffle(abstract)
    return concrete, abstract

def build_block_list(concrete_questions, abstract_questions):
    need_per_type = BLOCKS_PER_TYPE * QUESTIONS_PER_BLOCK  # 15
    if len(concrete_questions) < need_per_type:
        print(f"[WARN] Not enough CONCRETE ({len(concrete_questions)}) for {need_per_type}. Truncating.")
    if len(abstract_questions) < need_per_type:
        print(f"[WARN] Not enough ABSTRACT ({len(abstract_questions)}) for {need_per_type}. Truncating.")
    conc_pool = concrete_questions[:need_per_type]
    abst_pool = abstract_questions[:need_per_type]
    concrete_blocks, abstract_blocks = [], []
    for i in range(BLOCKS_PER_TYPE):
        concrete_blocks.append(conc_pool[i*QUESTIONS_PER_BLOCK:(i+1)*QUESTIONS_PER_BLOCK])
        abstract_blocks.append(abst_pool[i*QUESTIONS_PER_BLOCK:(i+1)*QUESTIONS_PER_BLOCK])
    order = ['concrete','abstract']; random.shuffle(order)
    if order[0]=='concrete':
        blocks = [('C',i+1,blk) for i,blk in enumerate(concrete_blocks)] + \
                 [('A',i+1,blk) for i,blk in enumerate(abstract_blocks)]
        first_part = "concrete"
    else:
        blocks = [('A',i+1,blk) for i,blk in enumerate(abstract_blocks)] + \
                 [('C',i+1,blk) for i,blk in enumerate(concrete_blocks)]
        first_part = "abstract"
    print(f"[PLAN] Block order: first {first_part} (5 blocks), then the other type (5 blocks).")
    return blocks

def load_socio_questions(path=SOCIO_Q_FILE):
    if not os.path.exists(path):
        print(f"[QNR] Socioeconomic questions file not found: {path}")
        cleanup_and_quit()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("[QNR] Invalid format: expected a list of question objects.")
        cleanup_and_quit()
    return data

def should_show_question(q, answers):
    cond = q.get("show_if")
    if not cond:
        return True
    ref_val = str(answers.get(cond.get("id", ""), "")).lower()
    pattern = cond.get("regex", "")
    try:
        return re.search(pattern, ref_val) is not None
    except re.error:
        return True  # if regex invalid, fail open

# ===== questionnaire =====
def text_input_loop(prompt_text, required=True, numeric_only=False):
    """
    Simple text input field with CAPS LOCK + Shift support and punctuation.
      - ENTER confirms (enforces required & numeric if requested)
      - BACKSPACE deletes
      - ESC quits
      - CAPS LOCK toggles caps mode
      - Shift (lshift/rshift) applies to the next character only
    """
    input_box = visual.Rect(win, width=WRAP_PIX, height=60, fillColor=[-0.2,-0.2,-0.2],
                            lineColor="black", pos=(0, -150))
    input_text = visual.TextStim(win, text="", color="black", height=GEN_TEXT_HEIGHT,
                                 pos=(0, -150), wrapWidth=WRAP_PIX * 0.95)
    prompt = visual.TextStim(win, text=f"{prompt_text}\n(Type your answer. ENTER to confirm.)",
                             color="black", height=GEN_TEXT_HEIGHT, wrapWidth=WRAP_PIX, pos=(0, 60))

    typed = ""
    caps_mode = False      # toggled by 'capslock'
    shift_next = False     # set by lshift/rshift, consumed by next char

    while True:
        prompt.draw(); input_box.draw(); input_text.text = typed; input_text.draw()
        win.flip(); core.wait(0.001)

        keys = kb.getKeys(waitRelease=False)
        for k in keys:
            name = k.name

            # quit
            if name == 'escape':
                cleanup_and_quit()

            # confirm
            elif name in ('return', 'num_enter'):
                val = typed.strip()
                if required and len(val) == 0:
                    continue
                if numeric_only and len(val) > 0:
                    # accept . or , as decimal separators
                    v = val.replace('.', '', 1).replace(',', '', 1)
                    if not v.isdigit():
                        continue
                return val

            # edits / modifiers
            elif name == 'backspace':
                typed = typed[:-1]

            elif name == 'capslock':
                caps_mode = not caps_mode

            elif name in ('lshift', 'rshift'):
                shift_next = True
                # don't append anything yet; applies to the NEXT printable key

            # punctuation
            elif name in PUNCT_KEYMAP:
                typed += PUNCT_KEYMAP[name]
                shift_next = False  # consume shift if it was set

            # alphanumerics (single char names)
            elif len(name) == 1:
                ch = name
                if ch.isalpha():
                    # apply caps logic: CAPS XOR SHIFT
                    if caps_mode ^ shift_next:
                        ch = ch.upper()
                typed += ch
                shift_next = False  # consume shift

            # ignore other control keys

def run_questionnaire(block_label="QNR"):
    socio_questions = load_socio_questions(SOCIO_Q_FILE)
    if not socio_questions:
        return

    mname, mcode, _ = send_marker("QUESTIONNAIRE_ON")
    log_event("questionnaire", block_label, -1, {"type":"QNR","field":"start"},
              mname, mcode, None, note="Questionnaire start")

    show_message("QUESTIONNAIRE\n\nAnswer the following questions.\nPress SPACE to continue.")

    answers = {}
    for i, q in enumerate(socio_questions, 1):
        # parse
        qid = str(q.get("id", "")).strip()
        if not qid:
            qid = f"qnr_{i:02d}"  # fallback ID so logging never breaks
        prompt = str(q.get("prompt", "")).strip()
        qtype  = str(q.get("type", "text") or "text").strip().lower()
        required = bool(q.get("required", False))
        if not prompt:
            continue  # skip malformed entries

        # conditional show
        if not should_show_question(q, answers):
            continue

        # collect answer
        numeric = (qtype == "number")
        answer = text_input_loop(prompt, required=required, numeric_only=numeric)

        # optional strict gender guard
        if qid == "gender":
            while str(answer).strip().lower() not in ("male", "female"):
                answer = text_input_loop("Please enter exactly 'Male' or 'Female':", required=True)

        answers[qid] = answer

        # log into existing CSV fields
        log_event(
            "questionnaire_item",
            block_label,
            -1,
            {
                "question_number": qid,
                "year": "",
                "type": "QNR",
                "field": prompt
            },
            "QNR_ITEM",
            0,
            None,
            choice=str(answer),
            note=f"qid={qid}; type={qtype}; required={required}"
        )

    # save responses JSON
    socio_out = os.path.join(LOG_DIR, f"socio_{exp_info['participant']}_{timestamp}.json")
    try:
        with open(socio_out, "w", encoding="utf-8") as sf:
            json.dump(answers, sf, ensure_ascii=False, indent=2)
        print(f"[QNR] Saved responses to {socio_out}")
    except Exception as e:
        print(f"[QNR] Could not save socio responses: {e}")

    mname, mcode, _ = send_marker("QUESTIONNAIRE_OFF")
    log_event("questionnaire", block_label, -1, {"type":"QNR","field":"end"},
              mname, mcode, None, note="Questionnaire end")

# ===== trial =====
def run_trial(block_label, idx_in_block, question_data):
    """
    Three-phase cumulative presentation with debounce:
      1) Show TEXT only.
      2) On click/space (released), add QUESTION (stem).
      3) On next click/space (released), add OPTIONS.
      Then wait for answer.
    """
    # ITI
    iti_duration = random.uniform(MIN_ITI_SECS, MAX_ITI_SECS)
    iti_start = global_clock.getTime()
    msg_text.text = "+"
    wait_secs_draw(iti_duration, [msg_text])
    send_marker("ITI")
    log_event("iti", block_label, idx_in_block, question_data, "ITI", 99, iti_start,
              note=f"ITI duration: {iti_duration:.2f}s")

    # Content
    stem_text  = question_data["question_text_translated"]
    question_t = question_data["question_itself_translated"]
    option_keys = [
        "question_option_A_translated","question_option_B_translated",
        "question_option_C_translated","question_option_D_translated",
        "question_option_E_translated"
    ]

    # reset states
    event.clearEvents(); kb.clearEvents(); mouse.clickReset(); wait_for_mouse_release()

    # PHASE 1: TEXT only
    question_text.text = stem_text
    question_itself.text = ""
    for i in range(5): opt_texts[i].text = ""

    t_on = global_clock.getTime()
    send_marker("Q_TEXT_ON")
    log_event("q_text_on", block_label, idx_in_block, question_data, "Q_TEXT_ON", 11, t_on)

    button_show_lbl.text = "Show question"

    # --- wait for first (debounced) reveal ---
    while True:
        question_text.draw()
        button_show.draw(); button_show_lbl.draw()
        win.flip(); core.wait(0.001)

        # mouse (press-and-release)
        if mouse.isPressedIn(button_show, buttons=[0]):
            wait_for_mouse_release()
            send_marker("BUTTON_CLICK")
            click_t = global_clock.getTime()
            log_event("button_click", block_label, idx_in_block, question_data,
                      "BUTTON_CLICK", TRIGGER_MAP["BUTTON_CLICK"], t_on,
                      button_click_t=f"{click_t - t_on:.6f}")
            debounce_after_trigger()
            break

        # keyboard (space released)
        keys = kb.getKeys(['space','escape'], waitRelease=True)
        if keys:
            if keys[0].name=='escape': cleanup_and_quit()
            send_marker("BUTTON_CLICK")
            click_t = global_clock.getTime()
            log_event("button_click", block_label, idx_in_block, question_data,
                      "BUTTON_CLICK", TRIGGER_MAP["BUTTON_CLICK"], t_on,
                      button_click_t=f"{click_t - t_on:.6f}")
            debounce_after_trigger()
            break

    # PHASE 2: add QUESTION
    question_itself.text = question_t
    stem_on = global_clock.getTime()
    send_marker("Q_STEM_ON")
    log_event("q_stem_on", block_label, idx_in_block, question_data, "Q_STEM_ON", 14, stem_on)

    button_show_lbl.text = "Show options"

    # --- wait for second (debounced) reveal ---
    while True:
        question_text.draw()
        question_itself.draw()
        button_show.draw(); button_show_lbl.draw()
        win.flip(); core.wait(0.001)

        if mouse.isPressedIn(button_show, buttons=[0]):
            wait_for_mouse_release()
            send_marker("BUTTON_CLICK")
            click2_t = global_clock.getTime()
            log_event("button_click", block_label, idx_in_block, question_data,
                      "BUTTON_CLICK", TRIGGER_MAP["BUTTON_CLICK"], stem_on,
                      button_click_t=f"{click2_t - stem_on:.6f}")
            debounce_after_trigger()
            break

        keys = kb.getKeys(['space','escape'], waitRelease=True)
        if keys:
            if keys[0].name=='escape': cleanup_and_quit()
            send_marker("BUTTON_CLICK")
            click2_t = global_clock.getTime()
            log_event("button_click", block_label, idx_in_block, question_data,
                      "BUTTON_CLICK", TRIGGER_MAP["BUTTON_CLICK"], stem_on,
                      button_click_t=f"{click2_t - stem_on:.6f}")
            debounce_after_trigger()
            break

    # PHASE 3: add OPTIONS
    for i,k in enumerate(option_keys):
        letter = chr(65+i)
        opt_texts[i].text = f"{letter}) {question_data[k]}"

    options_on = global_clock.getTime()
    send_marker("Q_OPTIONS_ON")
    log_event("q_options_on", block_label, idx_in_block, question_data, "Q_OPTIONS_ON", 15, options_on)

    # Wait for answer
    chosen = None
    event.clearEvents(); kb.clearEvents(); mouse.clickReset(); wait_for_mouse_release()

    while chosen is None:
        question_text.draw(); question_itself.draw()
        for i in range(5):
            opt_boxes[i].draw(); opt_texts[i].draw()
        win.flip(); core.wait(0.001)

        if any(mouse.getPressed()):
            for i,box in enumerate(opt_boxes):
                if box.contains(mouse):
                    wait_for_mouse_release()
                    chosen = chr(65+i); break

        keys = kb.getKeys(['a','b','c','d','e','1','2','3','4','5','escape'], waitRelease=True)
        if keys:
            name = keys[0].name
            if name=='escape': cleanup_and_quit()
            elif name in list('abcde'): chosen = name.upper()
            elif name in ['1','2','3','4','5']: chosen = chr(64+int(name))

    # Log answer
    ans_marker = f"ANS_{chosen}"
    send_marker(ans_marker)
    answer_time = global_clock.getTime()
    log_event("answer", block_label, idx_in_block, question_data, ans_marker, TRIGGER_MAP.get(ans_marker,0),
              options_on, choice=chosen, opt_view_t=f"{answer_time - options_on:.6f}")
    msg_text.text = "Response recorded"
    wait_secs_draw(0.5, [msg_text])

# ===== block runner =====
def run_block(block_label, questions_in_block):
    send_marker("BLK_ON")
    log_event("block_start", block_label, -1, {}, "BLK_ON", 91, None,
              note=f"{block_label} start (target {BLOCK_DURATION_SECS}s)")
    block_clock = core.Clock(); block_clock.reset()
    trial_idx = 0

    for q in questions_in_block:
        trial_idx += 1
        run_trial(block_label, trial_idx, q)
        if block_clock.getTime() >= BLOCK_DURATION_SECS:
            break

    remaining = BLOCK_DURATION_SECS - block_clock.getTime()
    if remaining > 0:
        send_marker("BLOCK_REST")
        log_event("block_rest_wait", block_label, -1, {}, "BLOCK_REST", 93, None,
                  note=f"Waiting {remaining:.1f}s to complete 7-min block")
        rest_clock = core.Clock()
        while rest_clock.getTime() < remaining:
            left = remaining - rest_clock.getTime()
            msg_text.text = f"Rest\n\nNext block in {int(left)} seconds..."
            msg_text.draw(); win.flip()
            keys = kb.getKeys(['escape'], waitRelease=False)
            if keys: cleanup_and_quit()

    send_marker("BLK_OFF")
    log_event("block_end", block_label, -1, {}, "BLK_OFF", 92, None,
              note=f"{block_label} end (actual {block_clock.getTime():.1f}s)")

# ===== main =====
# EXP_START: send first, then log (tighter alignment)
mname, mcode, _ = send_marker("EXP_START")
log_event("experiment", "START", -1, {}, mname, mcode, None,
          note=f"Experiment started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

show_message(
    "Welcome!\n\n"
    "You will complete 10 blocks.\n\n"
    "Each block has 3 questions and lasts 7 minutes.\n\n"
    "• If you finish the 3 questions before 7 minutes, a REST screen will appear and you’ll wait until the block ends.\n\n"
    "• If you are still answering when the 7 minutes end, the experiment immediately moves to the next block.\n\n"
    "Between blocks, there is a short rest period of about 20 seconds.\n\n"
    "Press SPACE to begin."
)

# sanity ping: send first, then log (and use the actual PING code)
mname, mcode, _ = send_marker("PING")
log_event("experiment", "PING", -1, {}, mname, mcode, None,
          note="Post-welcome ping")

if RUN_QUESTIONNAIRE_BEFORE:
    run_questionnaire(block_label="PRE")

concrete_q, abstract_q = load_questions()
plan = build_block_list(concrete_q, abstract_q)  # list of (type_tag, within_idx, [questions])

for type_tag, within_idx, questions in plan:
    label_prefix = "C" if type_tag == "C" else "A"
    block_label = f"{label_prefix}{within_idx}"
    show_message(f"BLOCK {block_label}\n\nPress SPACE to continue.")
    run_block(block_label, questions)

# EXP_END: send first, then log (tighter alignment)
mname, mcode, _ = send_marker("EXP_END")
log_event("experiment", "END", -1, {}, mname, mcode, None,
          note=f"Experiment ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")

show_message("Thank you for participating!\n\nPress SPACE to finish.")
cleanup_and_quit()
