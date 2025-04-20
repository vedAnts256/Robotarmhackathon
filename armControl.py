# Imports
import base64
import cv2
from openai import OpenAI

import logging
import os
import time
from dataclasses import asdict
from pprint import pformat

import rerun as rr

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    is_headless,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser
from lerobot.common.policies.act.modeling_act import ACTPolicy

import subprocess

def lookatcards():
    cmd = [
    'python', 'lerobot/scripts/control_robot.py',
    '--robot.type=so100',
    '--control.type=replay',
    '--control.fps=30',
    '--control.repo_id=vednot25t/lookatcardsfinal',
    '--control.episode=0'
    ]
    subprocess.run(cmd)

def lookattable():
    cmd = [
    'python', 'lerobot/scripts/control_robot.py',
    '--robot.type=so100',
    '--control.type=replay',
    '--control.fps=30',
    '--control.repo_id=vednot25t/lookattable',
    '--control.episode=0'
    ]
    subprocess.run(cmd)

def goback():
    cmd = [
    'python', 'lerobot/scripts/control_robot.py',
    '--robot.type=so100',
    '--control.type=replay',
    '--control.fps=30',
    '--control.repo_id=vednot25t/goback',
    '--control.episode=0'
    ]
    subprocess.run(cmd)

def tap():
    cmd = [
    'python', 'lerobot/scripts/control_robot.py',
    '--robot.type=so100',
    '--control.type=replay',
    '--control.fps=30',
    '--control.repo_id=vednot25t/tap',
    '--control.episode=0'
    ]
    subprocess.run(cmd)

def money():
    cmd = [
    "python", "lerobot/scripts/control_robot.py",
    "--robot.type=so100",
    "--control.type=record",
    "--control.fps=30",
    '--control.single_task=Grasp two cards at a time and place them in a cup on the side',
    f"--control.repo_id=vednot25t/eval_act_so100_test2",
    '--control.tags=["so100","card_cup"]',
    "--control.warmup_time_s=5",
    "--control.episode_time_s=30",
    "--control.reset_time_s=30",
    "--control.num_episodes=1",
    "--control.push_to_hub=false",
    "--control.policy.path=/home/ved/.cache/huggingface/hub/models--vednot25t--s0100_test/snapshots/18980f8d4615cda0da3c380f6c13d75bf2311a01"
    ]
    subprocess.run(cmd)

def money2():
    cmd = [
    'python', 'lerobot/scripts/control_robot.py',
    '--robot.type=so100',
    '--control.type=replay',
    '--control.fps=30',
    '--control.repo_id=vednot25t/money2',
    '--control.episode=0'
    ]
    subprocess.run(cmd)


# ------------------------------------------------------------------------------
# Configuration: set your API key here or via the OPENAI_API_KEY env var
# ------------------------------------------------------------------------------
API_KEY = os.getenv("OPENAI_API_KEY") or "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"
""
if not API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")
client = OpenAI(api_key=API_KEY)

# ------------------------------------------------------------------------------
# Camera capture helper
# ------------------------------------------------------------------------------

def capture_image(stage: str, cam_index: int = 6) -> str:
    """
    Waits for Enter, captures one frame from camera, writes to "{stage}.jpg",
    and returns the filename.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cam_index}")
    input(f"\nPress Enter to capture the {stage} image...")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image from camera")
    filename = f"{stage}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[capture_image] saved → {filename}")
    return filename

# ------------------------------------------------------------------------------
# Image ↔ data‑URL & card parsing
# ------------------------------------------------------------------------------

def encode_image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def parse_card_list(raw: str) -> list[str]:
    return [tok.strip() for tok in raw.split(";") if tok.strip()]

# ------------------------------------------------------------------------------
# OpenAI calls
# ------------------------------------------------------------------------------

def detect_cards(image_path: str) -> list[str]:
    data_url = encode_image_to_data_url(image_path)
    resp = client.responses.create(
        model="gpt-4.1",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": (
                     "Give me back ONLY a semicolon‑separated string of the cards "
                     "from left to right, top to bottom. Use 2‑char codes (e.g. 'KH')."
                 )},
                {"type": "input_image", "image_url": data_url}
            ]
        }]
    )
    cards = parse_card_list(resp.output_text)
    print(f"[detect_cards] {os.path.basename(image_path)} → {cards}")
    return cards

def should_proceed(hole: list[str], table: list[str]) -> bool:
    prompt = (
        f"My hole cards: {', '.join(hole)}.\n"
        f"Community cards: {', '.join(table)}.\n"
        "Based purely on poker strategy with high risk tolerance, should I proceed? "
        "Answer 'Yes' or 'No'."
    )
    comp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    ans = comp.choices[0].message.content.strip().lower()
    print(f"[should_proceed] → {ans}")
    return ans.startswith("y")

def determine_winner(my_hole: list[str], opp_hole: list[str], table: list[str]) -> str:
    prompt = (
        f"My hole cards: {', '.join(my_hole)}.\n"
        f"Opponent's hole cards: {', '.join(opp_hole)}.\n"
        f"Community cards: {', '.join(table)}.\n"
        "Who wins? Reply 'Me', 'Opponent', or 'Tie'."
    )
    comp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    result = comp.choices[0].message.content.strip()
    print(f"[determine_winner] → {result}")
    return result

# ------------------------------------------------------------------------------
# Main poker flow
# ------------------------------------------------------------------------------

def play_hand():
    # 1) My hole cards
    print("capture_image: my_hole")
    lookatcards()
    time.sleep(0.5)
    hole_img = capture_image("my_hole")
    my_hole = detect_cards(hole_img)
    time.sleep(0.5)

    # 2) Flop
    print("capture_image: flop" \
    "")
    lookattable()
    time.sleep(0.5)
    flop_img = capture_image("flop")
    table = detect_cards(flop_img)
    if not should_proceed(my_hole, table):
        print("→ Fold on the flop")
        tap()
        return
    money()
    time.sleep(0.5)
    goback()

    # 3) Turn
    lookattable()
    time.sleep(0.5)
    turn_img = capture_image("turn")
    table += detect_cards(turn_img)
    if not should_proceed(my_hole, table):
        print("→ Fold on the turn")
        tap()
        return
    money2()
    time.sleep(0.5)

    # 4) River
    river_img = capture_image("river")
    table += detect_cards(river_img)
    if not should_proceed(my_hole, table):
        print("→ Fold on the river")
        tap()
        return

    print("→ Play to showdown!")

    # 5) Opponent's hole cards
    opp_img = capture_image("opp_hole")
    opp_hole = detect_cards(opp_img)

    # 6) Determine winner
    result = determine_winner(my_hole, opp_hole, table)
    if result.lower() == "me":
        print("🎉 I win!")
    elif result.lower() == "opponent":
        print("😞 Opponent wins.")
    else:
        print("🤝 It's a tie!")

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main():
    print("=== Poker Robot ===")
    print("At each stage, press Enter to capture only the image you need now.")
    play_hand()
    
    
if __name__ == "__main__":
    main()

    
