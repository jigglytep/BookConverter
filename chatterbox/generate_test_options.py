import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from chatterbox.tts import ChatterboxTTS
import config

device = "cuda:0"
print(f"Loading Chatterbox model on {device}...")
model = ChatterboxTTS.from_pretrained(device)

ref_path = "references/practical_engineering/practical_engineering.wav"
if Path(ref_path).exists():
    print(f"Loading reference voice from {ref_path}...")
    model.prepare_conditionals(ref_path)
else:
    print("Using default voice (no reference)...")

# text_opt_a = "Chapter One. — The - morning - light - filtered - slowly - through - the tall laboratory windows, illuminating the instruments on the workbench."
text_opt_a = "AND CHAPTER THE ONE."
# Option A: Direct synthesis with em-dash prosodic pause
print("\n--- Generating Option A (Prosodic Punctuation: ' — ') ---")
print(f"Text: {text_opt_a}")
wav_a = model.generate(
    text_opt_a,
    exaggeration=config.EXAGGERATION,
    cfg_weight=config.CFG_WEIGHT,
    temperature=config.TEMPERATURE,
    top_p=config.TOP_P,
    repetition_penalty=config.REPETITION_PENALTY,
)
wav_a_np = wav_a.squeeze().cpu().numpy()
sf.write("test_option_a.wav", wav_a_np, model.sr, subtype="PCM_16")
print(f"Saved test_option_a.wav ({len(wav_a_np)/model.sr:.2f}s)")

# Option B: Explicit tag [pause] with programmatic silence splicing (350ms)
text_opt_b = "There Chapter the One"
# [pause] The morning light filtered slowly through the tall laboratory windows, illuminating the instruments on the workbench."
print("\n--- Generating Option B (Explicit [pause] tag with 350ms audio silence splice) ---")
print(f"Text: {text_opt_b}")

parts = [p.strip() for p in text_opt_b.split("[pause]") if p.strip()]
wav_b_parts = []
pause_samples = int(model.sr * 0.35)  # 350ms silence
silence = np.zeros(pause_samples, dtype=np.float32)

for idx, part in enumerate(parts):
    print(f"Synthesizing part {idx+1}: '{part}'")
    w = model.generate(
        part,
        exaggeration=config.EXAGGERATION,
        cfg_weight=config.CFG_WEIGHT,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        repetition_penalty=config.REPETITION_PENALTY,
    )
    wav_b_parts.append(w.squeeze().cpu().numpy())
    if idx < len(parts) - 1:
        wav_b_parts.append(silence)

wav_b_np = np.concatenate(wav_b_parts)
sf.write("test_option_b.wav", wav_b_np, model.sr, subtype="PCM_16")
print(f"Saved test_option_b.wav ({len(wav_b_np)/model.sr:.2f}s)")

print("\nFinished successfully!")
