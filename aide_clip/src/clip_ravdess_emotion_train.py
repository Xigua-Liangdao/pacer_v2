import argparse
import atexit
import copy
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
import numpy as np
from PIL import Image

try:
    from qcpa import QCPAHead
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from qcpa import QCPAHead

# RAVDESS-specific dataset definitions added here while preserving the aide_clip strict-frozen CLIP + adapter pipeline.
RAVDESS_LABEL_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
EMOTION_LABELS = list(RAVDESS_LABEL_MAP.values())
EMOTION_DISPLAY_MAP = {
    "neutral": "neutral",
    "calm": "calm",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fearful",
    "disgust": "disgusted",
    "surprised": "surprised",
}
RAVDESS_EMOTION_CODE_MAP = dict(RAVDESS_LABEL_MAP)
RAVDESS_MODALITY_CODE_MAP = {
    "01": "full_av",
    "02": "video_only",
    "03": "audio_only",
}
RAVDESS_VOCAL_CHANNEL_CODE_MAP = {
    "01": "speech",
    "02": "song",
}
RAVDESS_INTENSITY_CODE_MAP = {
    "01": "normal",
    "02": "strong",
}
DEFAULT_VISUAL_MODALITY_CODES = {"02"}
DEFAULT_VOCAL_CHANNEL_CODES = {"01"}
DEFAULT_VIDEO_EXTENSIONS = {".mp4"}
DEFAULT_SAMPLING_WINDOW_START = 0.4
DEFAULT_SAMPLING_WINDOW_END = 0.9
DEFAULT_DIFF_ALPHA = 0.6
DEFAULT_DIFF_BETA = 0.4
DEFAULT_MIN_GAP_RATIO = 0.08
DEFAULT_SCORE_SMOOTH_WINDOW = 3
DEFAULT_FRAME_DIFF_METRIC = "gray_l1"
DEFAULT_SAMPLING_DEBUG_SAMPLES = 6
DEFAULT_LEGACY_STRICT_BASELINE_CACHE = "/data1/yanjing/talk2bev/aide_clip/cache/ravdess_features/strict_features_f02d08f36ca4e172.pt"

LOCKED_BASELINE_PRESET = {
    "prompt_set": "ravdess_8_facial_cues",
    "epochs": 80,
    "extract_batch_size": 32,
    "train_batch_size": 32,
    "lr": 1.5e-4,
    "weight_decay": 5e-4,
    "max_grad_norm": 1.0,
    "num_frames": 5,
    "frame_sampling_mode": "middle_late",
    "feature_layout": "sequence",
    "adapter_hidden_dim": 256,
    "adapter_dropout": 0.2,
    "temporal_head": "transformer",
    "temporal_num_heads": 4,
    "temporal_num_layers": 1,
    "temporal_pooling": "mean",
    "use_class_weight": False,
    "label_smoothing": 0.0,
    "loss_type": "focal",
    "focal_gamma": 1.5,
    "use_test_ensemble": True,
    "ensemble_group_size": 2,
    "strict_frozen_clip": True,
    "use_global_logit_scale": False,
    "use_prompt_weight": False,
    "use_class_temperature": False,
    "use_class_bias": False,
    "use_qcpa": False,
    "qcpa_num_heads": 4,
    "attention_scope": "local",
    "dual_stage": False,
    "no_residual_gate": False,
    "no_mahalanobis": False,
    "no_lowrank_bias": False,
    "bias_rank": 16,
    "run_zero_shot_eval": True,
    "zero_shot_only": False,
    "report_train_metrics": True,
    "seed": 45,
    "benchmark_test_fold": 0,
    "benchmark_val_fold": 1,
    "split_mode": "benchmark_5fold",
    "allowed_modalities": "02",
    "allowed_vocal_channels": "01",
    "allowed_intensities": "01,02",
    "sampling_window_start": 0.4,
    "sampling_window_end": 0.9,
    "diff_alpha": 0.6,
    "diff_beta": 0.4,
    "min_gap_ratio": 0.08,
    "score_smooth_window": 3,
    "frame_diff_metric": "gray_l1",
    "use_amp": False,
    "early_stopping_patience": 0,
    "early_stopping_min_delta": 0.0,
    "lr_scheduler": "none",
    "legacy_feature_cache_path": None,
}

EXPERIMENT_PRESETS = {
    "exp_baseline_locked": dict(LOCKED_BASELINE_PRESET),
    "exp_baseline_locked_capc": dict(LOCKED_BASELINE_PRESET),
    "exp_baseline_locked_qcpa": {
        **LOCKED_BASELINE_PRESET,
        "use_qcpa": True,
        "qcpa_num_heads": 4,
        "attention_scope": "local",
        "dual_stage": False,
        "no_residual_gate": False,
        "no_mahalanobis": False,
        "no_lowrank_bias": False,
        "bias_rank": 16,
    },
    "exp_middle_late": dict(LOCKED_BASELINE_PRESET),
    "exp_diff_guided": {
        **LOCKED_BASELINE_PRESET,
        "frame_sampling_mode": "diff_guided",
    },
}

RAVDESS_PROMPT_GROUPS = {
    "neutral": [
        "The person appears neutral.",
        "The facial expression looks neutral.",
        "This clip shows a neutral expression.",
        "The speaker's face appears neutral.",
        "The visible emotion is neutral.",
        "The person shows a neutral emotional expression.",
        "The expression in this video looks neutral.",
    ],
    "calm": [
        "The person appears calm.",
        "The facial expression looks calm.",
        "This clip shows a calm expression.",
        "The speaker's face appears calm.",
        "The visible emotion is calm.",
        "The person shows a calm emotional state.",
        "The expression in this video looks calm.",
    ],
    "happy": [
        "The person appears happy.",
        "The facial expression looks happy.",
        "This clip shows a happy expression.",
        "The speaker's face appears happy.",
        "The visible emotion is happy.",
        "The person shows a happy emotional state.",
        "The expression in this video looks happy.",
    ],
    "sad": [
        "The person appears sad.",
        "The facial expression looks sad.",
        "This clip shows a sad expression.",
        "The speaker's face appears sad.",
        "The visible emotion is sad.",
        "The person shows a sad emotional state.",
        "The expression in this video looks sad.",
    ],
    "angry": [
        "The person appears angry.",
        "The facial expression looks angry.",
        "This clip shows an angry expression.",
        "The speaker's face appears angry.",
        "The visible emotion is angry.",
        "The person shows an angry emotional state.",
        "The expression in this video looks angry.",
    ],
    "fearful": [
        "The person appears fearful.",
        "The facial expression looks fearful.",
        "This clip shows a fearful expression.",
        "The speaker's face appears fearful.",
        "The visible emotion is fearful.",
        "The person shows a fearful emotional state.",
        "The expression in this video looks fearful.",
    ],
    "disgust": [
        "The person appears disgusted.",
        "The facial expression looks disgusted.",
        "This clip shows a disgusted expression.",
        "The speaker's face appears disgusted.",
        "The visible emotion is disgust.",
        "The person shows a disgusted emotional reaction.",
        "The expression in this video looks disgusted.",
    ],
    "surprised": [
        "The person appears surprised.",
        "The facial expression looks surprised.",
        "This clip shows a surprised expression.",
        "The speaker's face appears surprised.",
        "The visible emotion is surprise.",
        "The person shows a surprised emotional reaction.",
        "The expression in this video looks surprised.",
    ],
}

RAVDESS_FACIAL_CUE_PROMPT_GROUPS = {
    "neutral": [
        "The face looks neutral with a relaxed mouth and steady gaze.",
        "The expression is neutral, with level eyebrows and no strong facial movement.",
        "The person shows a neutral face with balanced features and a calm mouth.",
        "The facial expression appears neutral, without a smile, frown, or visible tension.",
        "The face stays neutral with even eyes, a relaxed brow, and a composed mouth.",
        "The visible expression is neutral, with little facial activation or emphasis.",
        "The person maintains a neutral face with stable eyes and relaxed facial muscles.",
    ],
    "calm": [
        "The person appears calm with soft eyes and a gently relaxed face.",
        "The face looks calm and composed, with smooth brows and a settled mouth.",
        "The expression is calm, gentle, and quietly controlled in the face.",
        "The person shows a calm face with relaxed eyelids and very light tension.",
        "The facial cues suggest calmness, composure, and an easy expression.",
        "The face appears serene and composed, with smooth features and a soft gaze.",
        "The visible expression looks calm, resting, and emotionally settled.",
    ],
    "happy": [
        "The face looks happy with a smile, lifted cheeks, and bright eyes.",
        "The person appears cheerful with raised cheeks and smiling mouth corners.",
        "The facial expression shows happiness through a visible smile and lively eyes.",
        "The person looks joyful with an open, smiling, and energetic face.",
        "The face shows happiness with lifted cheeks, softened eyes, and a warm smile.",
        "The visible expression is happy, upbeat, and clearly smiling.",
        "The person appears happy with expressive cheeks and a bright facial expression.",
    ],
    "sad": [
        "The facial expression looks sad with downturned mouth corners and drooping eyelids.",
        "The person appears sad with a downcast gaze and softly lowered eyebrows.",
        "The face shows sadness through low energy, heavy eyes, and a faint frown.",
        "The visible expression is sad, subdued, and gently pulled downward in the mouth.",
        "The person has a sad face with reduced movement and a sorrowful eye area.",
        "The facial cues suggest sadness, lowered eyes, and a withdrawn, low-energy expression.",
        "The face appears sorrowful and tired, with drooped eyes and mouth corners.",
    ],
    "angry": [
        "The face looks angry with furrowed brows, a tight jaw, and tense eyes.",
        "The person appears angry with a hard stare and tightened facial muscles.",
        "The facial expression shows anger through knitted brows and a compressed mouth.",
        "The face appears angry, forceful, and tense around the eyes and mouth.",
        "The person shows an angry face with a stern glare and strong brow tension.",
        "The expression looks angry with tightened lips and a confrontational gaze.",
        "The visible facial cues suggest anger, irritation, and pronounced facial tension.",
    ],
    "fearful": [
        "The person appears fearful with widened eyes and lips stretched in tension.",
        "The face looks fearful with alarmed eyes and eyebrows pulled upward in distress.",
        "The facial expression shows fear through wide eyes, tense eyelids, and a tight mouth.",
        "The person looks scared with anxious eyes and a strained, guarded expression.",
        "The face appears fearful with visible tension, lifted brows, and an uneasy mouth shape.",
        "The visible expression suggests fear, anxiety, and a defensive facial reaction.",
        "The person shows a fearful face with worried eyes and tightly stretched lips.",
    ],
    "disgust": [
        "The face shows disgust with a wrinkled nose and raised upper lip.",
        "The person appears disgusted with a grimace and an aversive facial reaction.",
        "The facial expression looks disgusted, with the nose tightened and the lip curled.",
        "The face appears disgusted with a visibly unpleasant, rejecting reaction.",
        "The person shows disgust through a curled lip and compressed nose area.",
        "The visible expression suggests disgust, aversion, and a rejecting grimace.",
        "The face looks disgusted with strong nose-wrinkling and upper-lip tension.",
    ],
    "surprised": [
        "The person looks surprised with raised eyebrows, round eyes, and a dropped-open mouth.",
        "The face shows surprise through high brows and a sudden jaw-drop reaction.",
        "The facial expression appears surprised, reactive, and open-mouthed rather than tense.",
        "The person looks startled with widened eyes and a clearly opened mouth.",
        "The face appears surprised with lifted brows, round eyes, and a quick open-mouth reaction.",
        "The visible expression suggests surprise through raised features and a dropped jaw.",
        "The person shows a surprised facial expression with wide eyes and an open, rounded mouth.",
    ],
}

RAVDESS_PAIRWISE_CUE_PROMPT_GROUPS = {
    "neutral": [
        "The face looks expressionless with little visible emotion and a steady mouth.",
        "The facial expression is neutral, flat, and without a smile, frown, or strong tension.",
        "The person shows a neutral face with balanced features and minimal emotional activation.",
        "The visible expression looks emotionally neutral rather than calm, happy, fearful, or sad.",
        "The face stays neutral with level brows, a relaxed mouth, and almost no expressive change.",
        "The person has a plain neutral expression with little movement around the eyes or lips.",
    ],
    "calm": [
        "The face looks calm and composed with soft eyes and a peaceful, relaxed expression.",
        "The person appears calm rather than expressionless, with gentle facial relaxation and a settled mouth.",
        "The visible expression suggests calm composure, smooth brows, and quiet ease in the face.",
        "The face appears relaxed and serene, more peaceful than neutral and without strong emotional tension.",
        "The person shows a calm facial expression with soft eyelids and a composed, restful look.",
        "The expression is calm and controlled, with gentle relaxation instead of a flat neutral face.",
    ],
    "happy": [
        "The face looks happy with a clear smile, lifted cheeks, and bright cheerful eyes.",
        "The person appears happy and joyful, with smiling lips and raised cheeks rather than surprise.",
        "The visible expression shows happiness through a warm smile instead of an open startled mouth.",
        "The face appears cheerful and upbeat, with smiling features rather than fearful or tense cues.",
        "The person shows a joyful facial expression with positive energy and clear smiling cues.",
        "The expression is happy and smiling, not startled, not fearful, and not emotionally flat.",
    ],
    "sad": [
        "The face looks sad with drooping eyelids, downturned lips, and low-energy facial movement.",
        "The person appears sad with a downcast gaze and a heavy expression, not wide-eyed like fear.",
        "The visible expression suggests sadness through lowered eyes and a softly pulled-down mouth.",
        "The face appears sorrowful and tired, with withdrawn low-energy cues rather than neutral calmness.",
        "The person shows a sad facial expression with drooped features, not tense like anger or fear.",
        "The expression is sad and subdued, with downward facial cues instead of widened eyes or a smile.",
    ],
    "angry": [
        "The face looks angry with furrowed brows, tight lips, and a stern confrontational stare.",
        "The person appears angry with a tense jaw and compressed mouth rather than fearful widened eyes.",
        "The visible expression shows anger through brow tension, hard eyes, and a forceful facial set.",
        "The face appears irritated and angry, with tightened lips instead of disgusted nose wrinkling.",
        "The person shows an angry facial expression with knitted brows and strong facial tension.",
        "The expression is angry and stern, not startled like surprise and not scared like fear.",
    ],
    "fearful": [
        "The face looks fearful with widened eyes, raised brows, and a tense scared expression.",
        "The person appears fearful with alarmed eyes and a strained mouth rather than a surprised open-mouth look.",
        "The visible expression suggests fear through anxious wide eyes, facial tension, and apprehension.",
        "The face appears scared and guarded, with tense fear cues instead of anger, calmness, or happiness.",
        "The person shows a fearful facial expression with worried eyes and a tight uneasy mouth.",
        "The expression is fearful and anxious, not simply startled and not smiling like happiness.",
    ],
    "disgust": [
        "The face looks disgusted with a wrinkled nose, raised upper lip, and rejecting expression.",
        "The person appears disgusted with nose tension and lip curl rather than angry tight lips.",
        "The visible expression shows disgust through aversion, nose wrinkling, and an unpleasant grimace.",
        "The face appears disgusted and rejecting, with upper-lip lift instead of fearful widened eyes.",
        "The person shows a disgusted facial expression with nose crease and strong aversive cues.",
        "The expression is disgusted and repulsed, not stern like anger and not startled like surprise.",
    ],
    "surprised": [
        "The face looks surprised with widened eyes, raised brows, and an open startled mouth.",
        "The person appears surprised with a dropped jaw and alert eyes rather than a smiling happy face.",
        "The visible expression shows surprise through an open mouth and sudden startled facial reaction.",
        "The face appears surprised and reactive, more startled than fearful and more open-mouthed than happy.",
        "The person shows a surprised facial expression with lifted brows and a rounded open mouth.",
        "The expression is surprised and startled, not anxious like fear and not cheerful like happiness.",
    ],
}

RAVDESS_STAGE_CUE_PROMPT_GROUPS = {
    "neutral": [
        "The actor keeps a neutral stage expression with still brows, a relaxed mouth, and minimal emphasis.",
        "This looks like a neutral performed expression with controlled features and no strong emotional signal.",
        "The face shows neutral delivery, with steady eyes and little visible tension or lift.",
        "The performer appears neutral, holding a composed mouth and balanced facial muscles.",
        "The visible expression is neutral and restrained, without smile, frown, or alarmed widening.",
        "The actor maintains a neutral facial pose with flat affect and low expressive activation.",
        "The face looks neutral, controlled, and intentionally unaccented for the performance.",
    ],
    "calm": [
        "The actor shows a calm performed expression with soft eyes, light facial relaxation, and gentle control.",
        "This looks calm rather than neutral, with a settled mouth and a composed, easy face.",
        "The face carries calm stage delivery, with smooth brows and a quietly relaxed expression.",
        "The performer appears calm, serene, and intentionally softened in the eyes and mouth.",
        "The visible expression suggests calmness through relaxed facial tension and a peaceful gaze.",
        "The actor maintains a calm facial pose with mild softness instead of flat neutrality.",
        "The face looks calm, composed, and gently expressive without strong emotional intensity.",
    ],
    "happy": [
        "The actor shows a happy performed expression with lifted cheeks, bright eyes, and a clear smile.",
        "This looks like staged happiness, with smiling mouth corners and energized facial movement.",
        "The face carries happy delivery through raised cheeks, warm eyes, and visible positive activation.",
        "The performer appears happy and expressive, with a deliberate smile rather than surprise.",
        "The visible expression suggests happiness through cheerful cheeks, softened eyes, and smiling lips.",
        "The actor maintains a joyful facial pose with upbeat energy and a clear performed smile.",
        "The face looks happy, animated, and intentionally bright for the performance.",
    ],
    "sad": [
        "The actor shows a sad performed expression with lowered energy, drooping eyelids, and downturned lips.",
        "This looks like staged sadness, with a heavy gaze and softly lowered facial features.",
        "The face carries sad delivery through downward mouth tension and withdrawn eye expression.",
        "The performer appears sad, subdued, and intentionally low-energy in the face.",
        "The visible expression suggests sadness through drooped features and a sorrowful, reduced activation pattern.",
        "The actor maintains a sad facial pose with downcast cues rather than angry or fearful tension.",
        "The face looks sad, fatigued, and deliberately pulled downward for the performance.",
    ],
    "angry": [
        "The actor shows an angry performed expression with knitted brows, a firm jaw, and tight lips.",
        "This looks like staged anger, with hard eyes, facial tension, and forceful mouth compression.",
        "The face carries angry delivery through brow lowering, stern focus, and controlled hostility.",
        "The performer appears angry and confrontational, with deliberate tension around the eyes and mouth.",
        "The visible expression suggests anger through compressed lips, brow contraction, and a hard stare.",
        "The actor maintains an angry facial pose with tense features rather than fear or disgust cues.",
        "The face looks angry, forceful, and intentionally tightened for the performance.",
    ],
    "fearful": [
        "The actor shows a fearful performed expression with widened eyes, raised brows, and a tense mouth.",
        "This looks like staged fear, with alarmed eyes and guarded facial tension rather than a smile.",
        "The face carries fearful delivery through anxious widening, lifted brows, and strain around the lips.",
        "The performer appears fearful and defensive, with deliberate tension in the eye and mouth area.",
        "The visible expression suggests fear through worried eyes, stretched lips, and uneasy facial activation.",
        "The actor maintains a fearful facial pose with anxious widening instead of open surprised ease.",
        "The face looks fearful, vigilant, and intentionally strained for the performance.",
    ],
    "disgust": [
        "The actor shows a disgusted performed expression with a wrinkled nose and raised upper lip.",
        "This looks like staged disgust, with aversive facial tension and a rejecting grimace.",
        "The face carries disgust delivery through nose tightening, lip curl, and unpleasant rejection cues.",
        "The performer appears disgusted, with deliberate nose wrinkling and upper-lip tension.",
        "The visible expression suggests disgust through aversion, facial recoil, and a curled mouth shape.",
        "The actor maintains a disgusted facial pose with repelled features rather than angry compression.",
        "The face looks disgusted, rejecting, and intentionally unpleasant for the performance.",
    ],
    "surprised": [
        "The actor shows a surprised performed expression with raised brows, widened eyes, and an open mouth.",
        "This looks like staged surprise, with a rounded jaw drop and reactive facial opening.",
        "The face carries surprise delivery through lifted features, alert eyes, and open-mouth reactivity.",
        "The performer appears surprised and startled, with deliberate facial opening rather than tension-heavy fear.",
        "The visible expression suggests surprise through raised brows, widened eyes, and a dropped jaw.",
        "The actor maintains a surprised facial pose with open reaction instead of smiling happiness.",
        "The face looks surprised, reactive, and intentionally widened for the performance.",
    ],
}

RAVDESS_AUTO_CANDIDATE_PROMPT_GROUPS = {
    "neutral": [
        "The face appears expressionless with level features and little visible emotion.",
        "The facial expression looks neutral, with neither a smile nor a downward mouth pull.",
        "The person shows a flat facial expression with minimal movement in the eyes and lips.",
        "The face remains even and unreadable, without a jaw drop, bright smile, or sad droop.",
        "The visible expression is emotionally plain, with level eyelids and a straight mouth line.",
        "The person has a neutral face with balanced features and no strong upward or downward pull.",
        "The face looks still and unaffected, showing no alarm, no smile, and no sorrowful sagging.",
        "The expression stays blank and controlled, with very little activation around the mouth corners.",
        "The face appears steady and unmarked, not especially peaceful, downcast, or startled.",
        "The visible facial cues are sparse, with no open-mouth surprise and no low-energy sadness.",
        "The person keeps a plain face with steady eyes and a mouth that is neither open nor downturned.",
        "The expression looks restrained and emotionally undecorated, with level brows and stable lips.",
    ],
    "calm": [
        "The face looks relaxed and peaceful, with soft eyes and a gently settled mouth.",
        "The person shows a composed facial expression with light eyelids and smooth brows.",
        "The visible expression appears serene, with facial features resting in quiet ease.",
        "The face carries a gentle, composed look rather than an empty or expressionless one.",
        "The facial cues suggest calm composure, with relaxed muscles and a peaceful gaze.",
        "The person has a softly settled face, with no visible strain or abrupt emotional change.",
        "The expression looks restful and controlled, with smooth features and light facial tension.",
        "The face appears tranquil and steady, more peaceful than flat or emotionally blank.",
        "The visible look is calm and contained, with relaxed lips and quietly attentive eyes.",
        "The person shows a mild, composed expression with gentle facial relaxation.",
        "The face remains peaceful and easy, without cheerful smiling or fearful alarm.",
        "The expression is calm and unhurried, with a settled brow and softly relaxed mouth.",
    ],
    "happy": [
        "The face shows a clear smile with lifted cheeks and bright, lively eyes.",
        "The person appears cheerful, with smiling mouth corners and raised cheek muscles.",
        "The visible expression looks joyful, with a warm smile and energized facial features.",
        "The face carries a pleasant smiling look, with cheeks pushed upward and eyes engaged.",
        "The facial cues suggest happiness through an obvious smile and an open, positive expression.",
        "The person has a cheerful face with lifted cheeks and light in the eye area.",
        "The expression looks upbeat and smiling, not startled, tense, or withdrawn.",
        "The face appears joyful and friendly, with visible smile lines and raised cheeks.",
        "The person shows a happy expression marked by a curved mouth and bright eyes.",
        "The visible look is smiling and positive, with facial features lifted rather than tightened.",
        "The face presents a pleased expression with an easy smile and animated cheeks.",
        "The expression is distinctly cheerful, driven by smiling lips and lifted facial contours.",
    ],
    "sad": [
        "The face shows downturned lips, a lowered gaze, and drooping eyelids.",
        "The person appears downcast, with heavy eyes and a clearly downward mouth line.",
        "The visible expression looks sorrowful, with reduced energy and a pulled-down lower face.",
        "The face carries a tired, saddened look with drooped features instead of level neutral stillness.",
        "The facial cues suggest sadness through weakened mouth corners and a gaze angled downward.",
        "The person has a subdued expression with sagging eyelids and lips that slope downward.",
        "The expression looks withdrawn and low-energy, with no smile and no open startled mouth.",
        "The face appears mournful, with soft downward pull around the eyes and the lip corners.",
        "The visible look is somber and heavy, not blank like neutral and not open like surprise.",
        "The person shows a sad facial expression through drooped eyelids, lowered gaze, and downturned lips.",
        "The face remains low and sorrowful, with a weak mouth line and eyes that do not widen.",
        "The expression is quietly sad, marked by downward facial cues rather than lifted brows or jaw drop.",
    ],
    "angry": [
        "The face shows furrowed brows, tight lips, and a tense jaw.",
        "The person appears stern, with knitted brows and compressed mouth tension.",
        "The visible expression looks forceful and irritated, with hard eyes and a set jaw.",
        "The face carries a severe look, with brow contraction and tightness around the mouth.",
        "The facial cues suggest anger through a rigid jawline and narrowed, confrontational eyes.",
        "The person has an angry expression with firm lips and pronounced brow tension.",
        "The expression looks harsh and tense, not startled, smiling, or peacefully relaxed.",
        "The face appears combative, with compressed lips and a rigid lower face.",
        "The visible look is stern and strained, with muscular tension across the brow and jaw.",
        "The person shows anger through tightened mouth corners and a fixed, hard stare.",
        "The face remains severe and pressurized, with no smile and no open fearful alarm.",
        "The expression is clearly angry, driven by brow furrowing and mouth compression.",
    ],
    "fearful": [
        "The face shows wide eyes, raised brows, and tense fearful facial strain.",
        "The person appears scared, with alarmed eyes and tight tension around the mouth.",
        "The visible expression looks anxious, with widened eyes and lifted brows.",
        "The face carries a fearful look, with strong eye opening and uneasy facial tightness.",
        "The facial cues suggest fear through apprehensive eyes and strained facial muscles.",
        "The person has a tense scared expression, with raised brows and worried eyes.",
        "The expression looks alarmed and guarded, not cheerful, calm, or simply stern.",
        "The face appears distressed, with widened eyes and a mouth shaped by tension rather than anger.",
        "The visible look is fearful and uneasy, with upper-face lift and defensive strain.",
        "The person shows fear through open anxious eyes and tight facial control.",
        "The face remains alert and frightened, with visible tension instead of peaceful relaxation.",
        "The expression is distinctly fearful, marked by brow lift, eye widening, and facial unease.",
    ],
    "disgust": [
        "The face shows a wrinkled nose and a raised upper lip in aversion.",
        "The person appears repulsed, with nose tension and an upper-lip curl.",
        "The visible expression looks disgusted, with a rejecting mouth shape and tightened nose.",
        "The face carries an aversive look, with the nose creased and upper lip lifted.",
        "The facial cues suggest disgust through nose wrinkling and a repelled facial reaction.",
        "The person has a disgusted expression with a curled upper lip and compressed nose area.",
        "The expression looks rejecting and unpleasant, not fearful wide-eyed or angrily stern.",
        "The face appears repelled, with nasal wrinkling and a lifted lip rather than tight compressed lips.",
        "The visible look is aversive and rejecting, with strong tension around the nose bridge.",
        "The person shows disgust through upper-lip raise and an unpleasant grimace.",
        "The face remains repulsed and resistant, with a nose wrinkle instead of a startled jaw drop.",
        "The expression is clearly disgusted, driven by nasal wrinkling and lip lift.",
    ],
    "surprised": [
        "The face shows widened eyes, raised brows, and a clearly dropped jaw.",
        "The person appears startled, with an open rounded mouth and lifted eyebrows.",
        "The visible expression looks surprised, with round eyes and a sudden jaw-open reaction.",
        "The face carries a startled look, with a jaw drop and strong upper-face lift.",
        "The facial cues suggest surprise through a rounded open mouth and brows pushed upward.",
        "The person has a surprised expression with widened eyes and a distinctly open mouth shape.",
        "The expression looks abruptly reactive, with mouth opening rather than the downward pull of sadness.",
        "The face appears startled and open, with expanded eyes and a dropped lower jaw instead of a flat neutral mouth.",
        "The visible look is surprised and reactive, with upward brow movement and a visible mouth opening.",
        "The person shows surprise through a jaw drop, widened eyes, and a rounded lip opening.",
        "The face remains startled and open-faced, with a rounded mouth rather than closed lips or drooping lids.",
        "The expression is distinctly surprised, marked by brow lift, eye widening, and an open jaw posture.",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAVDESS_ROOT = os.environ.get("RAVDESS_ROOT", str(PROJECT_ROOT / "data" / "RAVDESS"))
DEFAULT_BENCHMARK_VIDEO_LIST = str(PROJECT_ROOT.parent / "MMEmotionRecognition-main" / "data" / "ravdess_videos.csv")
DEFAULT_FEATURE_CACHE_DIR = str(PROJECT_ROOT / "cache" / "ravdess_features")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "results" / "ravdess" / "clip_ravdess_emotion_supervised_results.json")
LOG_FILE_HANDLE = None
FEATURE_CACHE_VERSION = "v2"

BENCHMARK_ACTORS_PER_FOLD = {
    0: [2, 5, 14, 15, 16],
    1: [3, 6, 7, 13, 18],
    2: [10, 11, 12, 19, 20],
    3: [8, 17, 21, 23, 24],
    4: [1, 4, 9, 22],
}


def build_sampling_config_payload(
    frame_sampling_mode: str,
    sampling_window_start: float,
    sampling_window_end: float,
    diff_alpha: float,
    diff_beta: float,
    min_gap_ratio: float,
    score_smooth_window: int,
    frame_diff_metric: str,
) -> Dict[str, object]:
    return {
        "frame_sampling_mode": frame_sampling_mode,
        "sampling_window_start": round(float(sampling_window_start), 6),
        "sampling_window_end": round(float(sampling_window_end), 6),
        "diff_alpha": round(float(diff_alpha), 6),
        "diff_beta": round(float(diff_beta), 6),
        "min_gap_ratio": round(float(min_gap_ratio), 6),
        "score_smooth_window": int(score_smooth_window),
        "frame_diff_metric": frame_diff_metric,
    }


def compute_sampling_signature(sampling_config: Dict[str, object]) -> str:
    payload = json.dumps(sampling_config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def apply_experiment_preset(args) -> None:
    preset = EXPERIMENT_PRESETS.get(args.experiment_name)
    if preset is None:
        return
    for key, value in preset.items():
        setattr(args, key, copy.deepcopy(value))


def apply_experiment_output_names(args) -> None:
    if args.experiment_name == "custom":
        return
    default_output_path = Path(DEFAULT_OUTPUT)
    if args.output == DEFAULT_OUTPUT:
        args.output = str(default_output_path.parent / f"{args.experiment_name}.json")
    if args.checkpoint_output is None:
        args.checkpoint_output = str(Path(args.output).with_suffix(".pt"))
    if args.log_file is None:
        args.log_file = str(Path(args.output).with_suffix(".log"))


def normalize_emotion_label(label: str) -> Optional[str]:
    key = str(label).strip().lower()
    return key if key in EMOTION_LABELS else None


def parse_code_set(raw_value: str, valid_codes: Set[str], argument_name: str) -> Set[str]:
    value = str(raw_value or "all").strip().lower()
    if not value or value == "all":
        return set(valid_codes)
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    invalid = sorted(parsed - valid_codes)
    if invalid:
        raise ValueError(f"Invalid {argument_name}: {invalid}. Valid codes: {sorted(valid_codes)}")
    return parsed


def parse_actor_ids(raw_value: str) -> Optional[Set[int]]:
    value = str(raw_value or "").strip().lower()
    if not value or value == "all":
        return None
    actor_ids: Set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        actor_id = int(item)
        if actor_id < 1 or actor_id > 24:
            raise ValueError(f"Invalid actor id: {actor_id}. Valid range is 1-24.")
        actor_ids.add(actor_id)
    return actor_ids or None


def parse_extension_set(raw_value: str) -> Set[str]:
    value = str(raw_value or "").strip().lower()
    if not value or value == "all":
        return set(DEFAULT_VIDEO_EXTENSIONS)
    parsed = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parsed.add(item if item.startswith(".") else f".{item}")
    return parsed or set(DEFAULT_VIDEO_EXTENSIONS)


def index_ravdess_video_files(ravdess_root: str, allowed_extensions: Optional[Set[str]] = None) -> List[Path]:
    root = Path(ravdess_root)
    if not root.exists():
        raise FileNotFoundError(f"RAVDESS root not found: {root}")

    extensions = {ext.lower() for ext in (allowed_extensions or DEFAULT_VIDEO_EXTENSIONS)}
    return [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in extensions]


def load_sequence_allowlist(file_path: str) -> Set[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark video list not found: {path}")

    allowlist = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not allowlist:
        raise RuntimeError(f"Benchmark video list is empty: {path}")
    return allowlist


def resolve_benchmark_val_fold(test_fold: int, val_fold: Optional[int]) -> int:
    if val_fold is None:
        return (test_fold + 1) % len(BENCHMARK_ACTORS_PER_FOLD)
    return val_fold


def actor_to_benchmark_fold(actor_id: int) -> Optional[int]:
    for fold_idx, actor_ids in BENCHMARK_ACTORS_PER_FOLD.items():
        if actor_id in actor_ids:
            return fold_idx
    return None


def parse_ravdess_filename(file_name: str) -> Optional[Dict[str, object]]:
    stem = Path(file_name).stem
    parts = stem.split("-")
    if len(parts) != 7 or any((not part.isdigit()) for part in parts):
        return None

    modality_code, vocal_channel_code, emotion_code, intensity_code, statement_code, repetition_code, actor_code = parts
    label = RAVDESS_EMOTION_CODE_MAP.get(emotion_code)
    modality = RAVDESS_MODALITY_CODE_MAP.get(modality_code)
    vocal_channel = RAVDESS_VOCAL_CHANNEL_CODE_MAP.get(vocal_channel_code)
    intensity = RAVDESS_INTENSITY_CODE_MAP.get(intensity_code)
    if label is None or modality is None or vocal_channel is None or intensity is None:
        return None

    return {
        "sequence_id": stem,
        "modality_code": modality_code,
        "modality": modality,
        "vocal_channel_code": vocal_channel_code,
        "vocal_channel": vocal_channel,
        "emotion_code": emotion_code,
        "label": label,
        "intensity_code": intensity_code,
        "intensity": intensity,
        "statement": int(statement_code),
        "statement_code": statement_code,
        "repetition": int(repetition_code),
        "repetition_code": repetition_code,
        "actor_id": int(actor_code),
    }


def collect_ravdess_samples(
    ravdess_root: str,
    max_sequences: int = 0,
    allowed_modalities: Optional[Set[str]] = None,
    allowed_vocal_channels: Optional[Set[str]] = None,
    allowed_intensities: Optional[Set[str]] = None,
    allowed_extensions: Optional[Set[str]] = None,
    allowed_actor_ids: Optional[Set[int]] = None,
    sequence_allowlist: Optional[Set[str]] = None,
) -> List[Dict]:
    candidate_set = set(EMOTION_LABELS)
    root = Path(ravdess_root)
    if not root.exists():
        raise FileNotFoundError(f"RAVDESS root not found: {root}")

    allowed_modalities = set(allowed_modalities or DEFAULT_VISUAL_MODALITY_CODES)
    allowed_vocal_channels = set(allowed_vocal_channels or DEFAULT_VOCAL_CHANNEL_CODES)
    allowed_intensities = set(allowed_intensities or set(RAVDESS_INTENSITY_CODE_MAP.keys()))
    allowed_extensions = {ext.lower() for ext in (allowed_extensions or DEFAULT_VIDEO_EXTENSIONS)}

    samples: List[Dict] = []
    visual_file_count = 0
    wav_file_count = 0

    # RAVDESS-specific indexing added here: recurse over visual files and parse labels from filenames.
    for path in index_ravdess_video_files(ravdess_root, allowed_extensions):
        suffix = path.suffix.lower()
        if suffix == ".wav":
            wav_file_count += 1
        visual_file_count += 1
        meta = parse_ravdess_filename(path.name)
        if meta is None:
            continue
        if sequence_allowlist is not None and meta["sequence_id"] not in sequence_allowlist:
            continue
        if meta["label"] not in candidate_set:
            continue
        if meta["modality_code"] not in allowed_modalities:
            continue
        if meta["vocal_channel_code"] not in allowed_vocal_channels:
            continue
        if meta["intensity_code"] not in allowed_intensities:
            continue
        if allowed_actor_ids is not None and int(meta["actor_id"]) not in allowed_actor_ids:
            continue

        sample = {
            "sequence_id": meta["sequence_id"],
            "label": meta["label"],
            "video_path": str(path),
            "file_name": path.name,
            "modality_code": meta["modality_code"],
            "modality": meta["modality"],
            "vocal_channel_code": meta["vocal_channel_code"],
            "vocal_channel": meta["vocal_channel"],
            "emotion_code": meta["emotion_code"],
            "intensity_code": meta["intensity_code"],
            "intensity": meta["intensity"],
            "statement": int(meta["statement"]),
            "statement_code": meta["statement_code"],
            "repetition": int(meta["repetition"]),
            "repetition_code": meta["repetition_code"],
            "actor_id": int(meta["actor_id"]),
            "benchmark_fold": actor_to_benchmark_fold(int(meta["actor_id"])),
        }
        samples.append(sample)

    if not samples and visual_file_count == 0 and wav_file_count > 0:
        raise RuntimeError(
            "No visual RAVDESS files were found under the provided root. "
            "The current folder appears to contain audio-only .wav files, while this VLM pipeline requires visual modalities "
            "such as full audiovisual or video-only RAVDESS clips."
        )

    if max_sequences > 0:
        samples = samples[:max_sequences]

    log(
        f"[DATA] ravdess samples={len(samples)} modalities={sorted(allowed_modalities)} "
        f"vocal_channels={sorted(allowed_vocal_channels)} intensities={sorted(allowed_intensities)}"
    )
    return samples


def collect_benchmark_aligned_ravdess_samples(
    ravdess_root: str,
    benchmark_video_list: str,
    max_sequences: int = 0,
    allowed_extensions: Optional[Set[str]] = None,
    allowed_actor_ids: Optional[Set[int]] = None,
) -> List[Dict]:
    sequence_allowlist = load_sequence_allowlist(benchmark_video_list)
    samples = collect_ravdess_samples(
        ravdess_root=ravdess_root,
        max_sequences=max_sequences,
        allowed_modalities={"01"},
        allowed_vocal_channels={"01"},
        allowed_intensities=set(RAVDESS_INTENSITY_CODE_MAP.keys()),
        allowed_extensions=allowed_extensions,
        allowed_actor_ids=allowed_actor_ids,
        sequence_allowlist=sequence_allowlist,
    )
    log(
        f"[DATA] benchmark-aligned whitelist loaded: videos={len(sequence_allowlist)} "
        f"matched_samples={len(samples)} source={benchmark_video_list}"
    )
    return samples


def collect_samples(
    ravdess_root: str,
    max_sequences: int = 0,
    allowed_modalities: Optional[Set[str]] = None,
    allowed_vocal_channels: Optional[Set[str]] = None,
    allowed_intensities: Optional[Set[str]] = None,
    allowed_extensions: Optional[Set[str]] = None,
    allowed_actor_ids: Optional[Set[int]] = None,
) -> List[Dict]:
    return collect_ravdess_samples(
        ravdess_root=ravdess_root,
        max_sequences=max_sequences,
        allowed_modalities=allowed_modalities,
        allowed_vocal_channels=allowed_vocal_channels,
        allowed_intensities=allowed_intensities,
        allowed_extensions=allowed_extensions,
        allowed_actor_ids=allowed_actor_ids,
    )


def compute_split_counts(total_groups: int, ratios: List[float]) -> List[int]:
    raw_counts = [total_groups * ratio for ratio in ratios]
    counts = [int(x) for x in raw_counts]
    remainder = total_groups - sum(counts)
    order = sorted(range(len(ratios)), key=lambda idx: (raw_counts[idx] - counts[idx]), reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1

    positive_ratio_indices = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
    if total_groups >= len(positive_ratio_indices):
        for idx in positive_ratio_indices:
            if counts[idx] == 0:
                donor = max(
                    (j for j in positive_ratio_indices if counts[j] > 1),
                    key=lambda j: counts[j],
                    default=None,
                )
                if donor is not None:
                    counts[donor] -= 1
                    counts[idx] += 1
    return counts


def split_samples_by_actor(samples: List[Dict], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[Dict]]:
    actor_groups: Dict[int, List[Dict]] = {}
    for sample in samples:
        actor_groups.setdefault(int(sample["actor_id"]), []).append(sample)

    rng = random.Random(seed)
    actor_ids = sorted(actor_groups)
    rng.shuffle(actor_ids)

    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    n_train, n_val, n_test = compute_split_counts(len(actor_ids), [train_ratio, val_ratio, test_ratio])

    train_actor_ids = actor_ids[:n_train]
    val_actor_ids = actor_ids[n_train:n_train + n_val]
    test_actor_ids = actor_ids[n_train + n_val:n_train + n_val + n_test]

    train = [sample for actor_id in train_actor_ids for sample in actor_groups[actor_id]]
    val = [sample for actor_id in val_actor_ids for sample in actor_groups[actor_id]]
    test = [sample for actor_id in test_actor_ids for sample in actor_groups[actor_id]]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    if len(actor_ids) < 3:
        log(
            "[WARN] fewer than 3 actors detected. Actor-disjoint train/val/test cannot all be non-empty; "
            "current split will keep subject disjointness but one split may be empty."
        )

    log(
        f"[DATA] actor-wise split -> train_actors={sorted(train_actor_ids)} "
        f"val_actors={sorted(val_actor_ids)} test_actors={sorted(test_actor_ids)}"
    )
    return {"train": train, "val": val, "test": test}


def split_samples_benchmark_folds(samples: List[Dict], test_fold: int, val_fold: Optional[int]) -> Dict[str, List[Dict]]:
    if test_fold not in BENCHMARK_ACTORS_PER_FOLD:
        raise ValueError(f"Invalid benchmark test_fold: {test_fold}. Choices: {sorted(BENCHMARK_ACTORS_PER_FOLD)}")

    val_fold = resolve_benchmark_val_fold(test_fold, val_fold)
    if val_fold not in BENCHMARK_ACTORS_PER_FOLD:
        raise ValueError(f"Invalid benchmark val_fold: {val_fold}. Choices: {sorted(BENCHMARK_ACTORS_PER_FOLD)}")
    if val_fold == test_fold:
        raise ValueError("benchmark val_fold must be different from benchmark test_fold")

    train_folds = [fold for fold in sorted(BENCHMARK_ACTORS_PER_FOLD) if fold not in {test_fold, val_fold}]
    test_actor_ids = set(BENCHMARK_ACTORS_PER_FOLD[test_fold])
    val_actor_ids = set(BENCHMARK_ACTORS_PER_FOLD[val_fold])
    train_actor_ids = set(actor_id for fold in train_folds for actor_id in BENCHMARK_ACTORS_PER_FOLD[fold])

    train = [sample for sample in samples if int(sample["actor_id"]) in train_actor_ids]
    val = [sample for sample in samples if int(sample["actor_id"]) in val_actor_ids]
    test = [sample for sample in samples if int(sample["actor_id"]) in test_actor_ids]

    observed_actor_ids = sorted({int(sample["actor_id"]) for sample in samples})
    log(
        f"[DATA] benchmark 5-fold split -> test_fold={test_fold} val_fold={val_fold} train_folds={train_folds} "
        f"observed_actors={observed_actor_ids}"
    )
    log(
        f"[DATA] benchmark actor split -> train_actors={sorted({int(s['actor_id']) for s in train})} "
        f"val_actors={sorted({int(s['actor_id']) for s in val})} test_actors={sorted({int(s['actor_id']) for s in test})}"
    )
    return {"train": train, "val": val, "test": test}


def split_samples(samples: List[Dict], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[Dict]]:
    return split_samples_by_actor(samples, train_ratio, val_ratio, seed)


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def weighted_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    if not y_true:
        return 0.0
    support = Counter(y_true)
    total = len(y_true)
    weighted = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        weighted += (support.get(label, 0) / total) * f1
    return weighted


def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    mat = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in mat and p in mat[t]:
            mat[t][p] += 1
    return mat


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def log(message: str) -> None:
    global LOG_FILE_HANDLE

    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.write(line + "\n")
        LOG_FILE_HANDLE.flush()


def init_log_file(log_file: Optional[str]) -> Optional[str]:
    global LOG_FILE_HANDLE

    if not log_file:
        return None
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE_HANDLE = log_path.open("a", encoding="utf-8")
    return str(log_path)


def close_log_file() -> None:
    global LOG_FILE_HANDLE

    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.close()
        LOG_FILE_HANDLE = None


def default_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(".ckpt.pt")


def evaluate_split(y_true: List[str], y_pred: List[str]) -> Dict:
    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "weighted_f1": round(weighted_f1(y_true, y_pred, EMOTION_LABELS), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, EMOTION_LABELS),
    }


def summarize_predictions(y_true: List[str], y_pred: List[str]) -> Dict:
    metrics = evaluate_split(y_true, y_pred)
    metrics["prediction_distribution"] = prediction_distribution(y_pred, EMOTION_LABELS)
    return metrics


def prediction_distribution(y_pred: List[str], labels: List[str]) -> Dict[str, int]:
    counter = Counter(y_pred)
    return {label: int(counter.get(label, 0)) for label in labels}


def round_float_dict_matrix(matrix: List[List[float]], labels: List[str], digits: int = 6) -> Dict[str, Dict[str, float]]:
    return {
        row_label: {col_label: round(float(matrix[row_idx][col_idx]), digits) for col_idx, col_label in enumerate(labels)}
        for row_idx, row_label in enumerate(labels)
    }


def log_matrix(title: str, matrix_dict: Dict[str, Dict[str, float]]) -> None:
    labels = list(matrix_dict.keys())
    log(title)
    header = "label".ljust(12) + " ".join(label[:10].rjust(10) for label in labels)
    log(header)
    for row_label in labels:
        row = row_label[:12].ljust(12) + " ".join(f"{matrix_dict[row_label][col_label]:10.4f}" for col_label in labels)
        log(row)


def build_prompt_templates(prompt_template: str, prompt_set: str) -> List[str]:
    if prompt_set == "single":
        return [prompt_template]
    if prompt_set == "default_5":
        return [
            "The person looks <LABEL>.",
            "The facial expression is <LABEL>.",
            "Emotion state: <LABEL>.",
            "The speaker appears <LABEL>.",
            "This person feels <LABEL>.",
        ]
    if prompt_set == "ravdess_8":
        return [
            "The person looks <LABEL>.",
            "The facial expression is <LABEL>.",
            "The visible emotion is <LABEL>.",
            "This video shows a <LABEL> person.",
            "The speaker appears <LABEL>.",
            "Emotion label for the person: <LABEL>.",
            "Current emotional state: <LABEL>.",
        ]
    custom = [x.strip() for x in prompt_set.split("||") if x.strip()]
    return custom if custom else [prompt_template]


def build_ravdess_prompt_groups() -> List[List[str]]:
    return [list(RAVDESS_PROMPT_GROUPS[label]) for label in EMOTION_LABELS]


def build_ravdess_facial_cue_prompt_groups() -> List[List[str]]:
    return [list(RAVDESS_FACIAL_CUE_PROMPT_GROUPS[label]) for label in EMOTION_LABELS]


def build_ravdess_pairwise_cue_prompt_groups() -> List[List[str]]:
    return [list(RAVDESS_PAIRWISE_CUE_PROMPT_GROUPS[label]) for label in EMOTION_LABELS]


def build_ravdess_stage_cue_prompt_groups() -> List[List[str]]:
    return [list(RAVDESS_STAGE_CUE_PROMPT_GROUPS[label]) for label in EMOTION_LABELS]


def build_ravdess_auto_candidate_prompt_groups() -> List[List[str]]:
    return [list(RAVDESS_AUTO_CANDIDATE_PROMPT_GROUPS[label]) for label in EMOTION_LABELS]


def build_ravdess_hybrid_candidate_prompt_groups() -> List[List[str]]:
    hybrid_groups: List[List[str]] = []
    for label in EMOTION_LABELS:
        merged = list(RAVDESS_FACIAL_CUE_PROMPT_GROUPS[label]) + list(RAVDESS_AUTO_CANDIDATE_PROMPT_GROUPS[label])
        deduped = list(dict.fromkeys(merged))
        hybrid_groups.append(deduped)
    return hybrid_groups


def extract_variable_text_features(prompt_groups: List[List[str]], processor, model, device: str) -> List:
    import torch

    class_prompt_features = []
    total_prompts = sum(len(group) for group in prompt_groups)
    log(
        f"[TEXT] start variable text feature extraction: classes={len(prompt_groups)}, "
        f"total_prompts={total_prompts}"
    )
    for prompts in prompt_groups:
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        class_prompt_features.append(text_features.float().detach())
    log("[TEXT] done variable text feature extraction")
    return class_prompt_features


def compute_similarity_stats(similarity_matrix, labels: List[str], top_pairs: int = 6) -> Dict[str, object]:
    import torch

    sim = similarity_matrix.detach().cpu()
    if sim.numel() == 0:
        return {
            "mean_off_diagonal_similarity": 0.0,
            "max_off_diagonal_similarity": 0.0,
            "most_confusing_pairs": [],
        }

    mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
    off_diag = sim[mask]
    mean_off = float(off_diag.mean().item()) if off_diag.numel() > 0 else 0.0
    max_off = float(off_diag.max().item()) if off_diag.numel() > 0 else 0.0

    pair_scores = []
    for row_idx, row_label in enumerate(labels):
        for col_idx in range(row_idx + 1, len(labels)):
            pair_scores.append(
                {
                    "pair": [row_label, labels[col_idx]],
                    "similarity": round(float(sim[row_idx, col_idx].item()), 6),
                }
            )
    pair_scores.sort(key=lambda item: item["similarity"], reverse=True)
    return {
        "mean_off_diagonal_similarity": round(mean_off, 6),
        "max_off_diagonal_similarity": round(max_off, 6),
        "most_confusing_pairs": pair_scores[:top_pairs],
    }


def compute_mean_within_class_similarity(feature_groups: List, selected_indices: List[Tuple[int, ...]]) -> float:
    import torch

    per_class_scores: List[float] = []
    for class_idx, indices in enumerate(selected_indices):
        feats = feature_groups[class_idx][list(indices)]
        if feats.shape[0] <= 1:
            per_class_scores.append(1.0)
            continue
        sim = feats @ feats.transpose(0, 1)
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        per_class_scores.append(float(sim[mask].mean().item()))
    if not per_class_scores:
        return 0.0
    return float(sum(per_class_scores) / len(per_class_scores))


def build_selected_text_feature_tensor(feature_groups: List, selected_indices: List[Tuple[int, ...]]):
    import torch

    selected = []
    for class_idx, indices in enumerate(selected_indices):
        class_feats = feature_groups[class_idx][list(indices)]
        class_feats = class_feats / class_feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        selected.append(class_feats)
    return torch.stack(selected, dim=0).float()


def build_selected_prototypes(feature_groups: List, selected_indices: List[Tuple[int, ...]]):
    import torch

    prototypes = []
    for class_idx, indices in enumerate(selected_indices):
        class_feats = feature_groups[class_idx][list(indices)]
        class_feats = class_feats / class_feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        class_proto = class_feats.mean(dim=0)
        class_proto = class_proto / class_proto.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        prototypes.append(class_proto)
    return torch.stack(prototypes, dim=0).float()


def evaluate_prompt_selection(
    feature_groups: List,
    selected_indices: List[Tuple[int, ...]],
    top_pairs: int = 6,
) -> Dict[str, object]:
    selected_text_features = None
    if len({len(indices) for indices in selected_indices}) == 1:
        selected_text_features = build_selected_text_feature_tensor(feature_groups, selected_indices)
        prototypes = selected_text_features.mean(dim=1)
        prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    else:
        prototypes = build_selected_prototypes(feature_groups, selected_indices)
    similarity = prototypes @ prototypes.transpose(0, 1)
    similarity_dict = round_float_dict_matrix(similarity.detach().cpu().tolist(), EMOTION_LABELS)
    stats = compute_similarity_stats(similarity, EMOTION_LABELS, top_pairs=top_pairs)
    mean_intra = compute_mean_within_class_similarity(feature_groups, selected_indices)
    score = (
        float(stats["max_off_diagonal_similarity"]),
        float(stats["mean_off_diagonal_similarity"]),
        -float(mean_intra),
    )
    return {
        "score": score,
        "selected_text_features": selected_text_features,
        "prototype_similarity": similarity_dict,
        "mean_off_diagonal_similarity": stats["mean_off_diagonal_similarity"],
        "max_off_diagonal_similarity": stats["max_off_diagonal_similarity"],
        "most_confusing_pairs": stats["most_confusing_pairs"],
        "mean_within_class_similarity": round(float(mean_intra), 6),
    }


def initialize_prompt_selection(feature_groups: List, prompts_per_class: int, top_pairs: int) -> List[Tuple[int, ...]]:
    import torch

    selected_indices: List[Tuple[int, ...]] = []
    for class_feats in feature_groups:
        class_mean = class_feats.mean(dim=0, keepdim=True)
        class_mean = class_mean / class_mean.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sims = torch.mv(class_feats, class_mean.squeeze(0))
        best_idx = int(torch.argmax(sims).item())
        selected_indices.append((best_idx,))

    while len(selected_indices[0]) < prompts_per_class:
        for class_idx, class_feats in enumerate(feature_groups):
            remaining = [idx for idx in range(class_feats.shape[0]) if idx not in selected_indices[class_idx]]
            best_choice = None
            best_eval = None
            for candidate_idx in remaining:
                trial = list(selected_indices)
                trial[class_idx] = tuple(sorted(selected_indices[class_idx] + (candidate_idx,)))
                trial_eval = evaluate_prompt_selection(feature_groups, trial, top_pairs=top_pairs)
                if best_eval is None or trial_eval["score"] < best_eval["score"]:
                    best_eval = trial_eval
                    best_choice = candidate_idx
            selected_indices[class_idx] = tuple(sorted(selected_indices[class_idx] + (int(best_choice),)))
    return selected_indices


def select_ravdess_auto_prompt_groups(
    processor,
    model,
    device: str,
    prompts_per_class: int = 4,
    refine_passes: int = 2,
    top_pairs: int = 6,
    candidate_prompt_groups: Optional[List[List[str]]] = None,
    selection_name: str = "ravdess_8_auto_selected",
):
    candidate_prompt_groups = candidate_prompt_groups or build_ravdess_auto_candidate_prompt_groups()
    min_pool_size = min(len(group) for group in candidate_prompt_groups)
    if prompts_per_class <= 0:
        raise ValueError("auto prompt selection requires prompts_per_class > 0")
    if prompts_per_class > min_pool_size:
        raise ValueError(
            f"auto prompt selection requires prompts_per_class <= {min_pool_size}, got {prompts_per_class}"
        )

    feature_groups = extract_variable_text_features(candidate_prompt_groups, processor, model, device)
    selected_indices = initialize_prompt_selection(feature_groups, prompts_per_class=prompts_per_class, top_pairs=top_pairs)
    current_eval = evaluate_prompt_selection(feature_groups, selected_indices, top_pairs=top_pairs)

    all_combinations = [
        list(itertools.combinations(range(len(candidate_prompt_groups[class_idx])), prompts_per_class))
        for class_idx in range(len(candidate_prompt_groups))
    ]
    for _ in range(max(0, refine_passes)):
        improved = False
        for class_idx, class_combos in enumerate(all_combinations):
            best_indices = selected_indices[class_idx]
            best_eval = current_eval
            for combo in class_combos:
                if combo == selected_indices[class_idx]:
                    continue
                trial = list(selected_indices)
                trial[class_idx] = tuple(combo)
                trial_eval = evaluate_prompt_selection(feature_groups, trial, top_pairs=top_pairs)
                if trial_eval["score"] < best_eval["score"]:
                    best_indices = tuple(combo)
                    best_eval = trial_eval
            if best_indices != selected_indices[class_idx]:
                selected_indices[class_idx] = best_indices
                current_eval = best_eval
                improved = True
        if not improved:
            break

    selected_prompt_groups = [
        [candidate_prompt_groups[class_idx][prompt_idx] for prompt_idx in selected_indices[class_idx]]
        for class_idx in range(len(candidate_prompt_groups))
    ]
    selected_text_features = current_eval["selected_text_features"]
    diagnostics = {
        "selection_name": selection_name,
        "selection_method": "deterministic_greedy_coordinate_descent",
        "candidate_pool": {label: list(candidate_prompt_groups[class_idx]) for class_idx, label in enumerate(EMOTION_LABELS)},
        "candidate_pool_size_per_class": {label: len(candidate_prompt_groups[class_idx]) for class_idx, label in enumerate(EMOTION_LABELS)},
        "selected_prompt_indices": {
            label: [int(idx) for idx in selected_indices[class_idx]] for class_idx, label in enumerate(EMOTION_LABELS)
        },
        "selected_prompt_groups": {label: selected_prompt_groups[class_idx] for class_idx, label in enumerate(EMOTION_LABELS)},
        "prototype_similarity": current_eval["prototype_similarity"],
        "mean_off_diagonal_similarity": current_eval["mean_off_diagonal_similarity"],
        "max_off_diagonal_similarity": current_eval["max_off_diagonal_similarity"],
        "mean_within_class_similarity": current_eval["mean_within_class_similarity"],
        "most_confusing_pairs": current_eval["most_confusing_pairs"],
        "prompts_per_class": prompts_per_class,
        "refine_passes": refine_passes,
    }
    return selected_prompt_groups, selected_text_features, diagnostics


def build_class_prompts(prompt_template: str, prompt_set: str) -> List[List[str]]:
    if prompt_set == "ravdess_8":
        return build_ravdess_prompt_groups()
    if prompt_set == "ravdess_8_facial_cues":
        return build_ravdess_facial_cue_prompt_groups()
    if prompt_set == "ravdess_8_pairwise_cues":
        return build_ravdess_pairwise_cue_prompt_groups()
    if prompt_set == "ravdess_8_stage_cues":
        return build_ravdess_stage_cue_prompt_groups()
    if prompt_set == "ravdess_8_auto_selected":
        raise ValueError("ravdess_8_auto_selected requires CLIP-backed prompt selection and must be built inside main()")
    if prompt_set == "ravdess_8_auto_selected_hybrid":
        raise ValueError("ravdess_8_auto_selected_hybrid requires CLIP-backed prompt selection and must be built inside main()")
    templates = build_prompt_templates(prompt_template, prompt_set)
    return [[tpl.replace("<LABEL>", EMOTION_DISPLAY_MAP[label]) for tpl in templates] for label in EMOTION_LABELS]


def build_prompt_group_indices(num_prompts: int, group_size: int) -> List[List[int]]:
    if group_size <= 0 or group_size >= num_prompts:
        return [list(range(num_prompts))]
    groups = []
    for start in range(0, num_prompts, group_size):
        groups.append(list(range(start, min(start + group_size, num_prompts))))
    return groups


def count_parameters(parameters) -> int:
    return int(sum(param.numel() for param in parameters if getattr(param, "requires_grad", False)))


def format_tensor_list(tensor, precision: int = 4) -> List[float]:
    if tensor is None:
        return []
    values = tensor.detach().cpu().view(-1).tolist()
    return [round(float(value), precision) for value in values]


def clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def resolve_sampling_window(total_frames: int, start_ratio: float, end_ratio: float) -> Tuple[int, int]:
    if total_frames <= 0:
        return 0, 0
    start_ratio = clamp_ratio(start_ratio)
    end_ratio = clamp_ratio(end_ratio)
    if end_ratio < start_ratio:
        start_ratio, end_ratio = end_ratio, start_ratio
    start_idx = int(round((total_frames - 1) * start_ratio))
    end_idx = int(round((total_frames - 1) * end_ratio))
    start_idx = max(0, min(start_idx, total_frames - 1))
    end_idx = max(start_idx, min(end_idx, total_frames - 1))
    return start_idx, end_idx


def uniform_indices_in_window(total_frames: int, num_frames: int, start_ratio: float, end_ratio: float) -> List[int]:
    if total_frames <= 0:
        return [0]
    start_idx, end_idx = resolve_sampling_window(total_frames, start_ratio, end_ratio)
    if num_frames <= 1:
        return [int(round((start_idx + end_idx) / 2.0))]
    if end_idx <= start_idx:
        return [start_idx for _ in range(num_frames)]
    return [round(start_idx + index * (end_idx - start_idx) / (num_frames - 1)) for index in range(num_frames)]


def smooth_score_curve(scores: List[float], window_size: int) -> List[float]:
    if window_size <= 1 or len(scores) <= 2:
        return [float(score) for score in scores]
    radius = max(0, window_size // 2)
    smoothed = []
    for idx in range(len(scores)):
        start = max(0, idx - radius)
        end = min(len(scores), idx + radius + 1)
        smoothed.append(float(sum(scores[start:end]) / max(1, end - start)))
    return smoothed


def compute_frame_difference_score(frame_a: np.ndarray, frame_b: np.ndarray, metric: str) -> float:
    if metric != "gray_l1":
        raise ValueError(f"Unsupported frame_diff_metric: {metric}")
    return float(np.mean(np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32))))


def require_cv2() -> None:
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV (cv2) is required for RAVDESS video decoding but is not installed in the current environment.")


def read_gray_frame(capture, frame_index: int) -> Optional[np.ndarray]:
    require_cv2()
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def sample_diff_guided_frame_indices(
    capture,
    total_frames: int,
    num_frames: int,
    sampling_window_start: float,
    sampling_window_end: float,
    diff_alpha: float,
    diff_beta: float,
    min_gap_ratio: float,
    score_smooth_window: int,
    frame_diff_metric: str,
) -> List[int]:
    if total_frames <= 0:
        return [0]
    if num_frames <= 1:
        return uniform_indices_in_window(total_frames, 1, sampling_window_start, sampling_window_end)

    candidate_start, candidate_end = resolve_sampling_window(total_frames, sampling_window_start, sampling_window_end)
    candidate_indices = list(range(candidate_start, candidate_end + 1))
    if len(candidate_indices) <= num_frames:
        return uniform_indices_in_window(total_frames, num_frames, sampling_window_start, sampling_window_end)

    reference_gray = read_gray_frame(capture, 0)
    if reference_gray is None:
        return uniform_indices_in_window(total_frames, num_frames, sampling_window_start, sampling_window_end)

    valid_indices: List[int] = []
    scores: List[float] = []
    previous_gray: Optional[np.ndarray] = None
    for frame_index in candidate_indices:
        current_gray = read_gray_frame(capture, frame_index)
        if current_gray is None:
            continue
        local_diff = 0.0 if previous_gray is None else compute_frame_difference_score(current_gray, previous_gray, frame_diff_metric)
        ref_diff = compute_frame_difference_score(current_gray, reference_gray, frame_diff_metric)
        scores.append(float(diff_alpha) * local_diff + float(diff_beta) * ref_diff)
        valid_indices.append(frame_index)
        previous_gray = current_gray

    if len(valid_indices) < num_frames:
        return uniform_indices_in_window(total_frames, num_frames, sampling_window_start, sampling_window_end)

    smoothed_scores = smooth_score_curve(scores, score_smooth_window)
    min_gap_frames = max(1, int(round(max(1, total_frames - 1) * max(0.0, float(min_gap_ratio)))))

    ranked_positions = sorted(range(len(valid_indices)), key=lambda idx: (-smoothed_scores[idx], valid_indices[idx]))
    selected_indices: List[int] = []
    for position in ranked_positions:
        frame_index = valid_indices[position]
        if all(abs(frame_index - existing) >= min_gap_frames for existing in selected_indices):
            selected_indices.append(frame_index)
        if len(selected_indices) == num_frames:
            break

    if len(selected_indices) < num_frames:
        fallback_candidates = uniform_indices_in_window(total_frames, num_frames, sampling_window_start, sampling_window_end)
        for frame_index in fallback_candidates:
            if frame_index not in selected_indices:
                selected_indices.append(frame_index)
            if len(selected_indices) == num_frames:
                break

    if len(selected_indices) < num_frames:
        return uniform_indices_in_window(total_frames, num_frames, sampling_window_start, sampling_window_end)

    return sorted(selected_indices[:num_frames])


def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    frame_sampling_mode: str = "uniform",
    sampling_window_start: float = DEFAULT_SAMPLING_WINDOW_START,
    sampling_window_end: float = DEFAULT_SAMPLING_WINDOW_END,
) -> List[int]:
    if total_frames <= 0:
        return [0]
    if num_frames <= 1:
        if frame_sampling_mode == "middle_late":
            return uniform_indices_in_window(total_frames, 1, sampling_window_start, sampling_window_end)
        return [max(0, total_frames // 2)]
    if total_frames <= num_frames:
        return list(range(total_frames))
    if frame_sampling_mode == "middle_late":
        return uniform_indices_in_window(total_frames, num_frames, sampling_window_start, sampling_window_end)
    return [round(index * (total_frames - 1) / (num_frames - 1)) for index in range(num_frames)]


def read_sampled_frames(
    video_path: str,
    num_frames: int,
    frame_sampling_mode: str = "uniform",
    sampling_window_start: float = DEFAULT_SAMPLING_WINDOW_START,
    sampling_window_end: float = DEFAULT_SAMPLING_WINDOW_END,
    diff_alpha: float = DEFAULT_DIFF_ALPHA,
    diff_beta: float = DEFAULT_DIFF_BETA,
    min_gap_ratio: float = DEFAULT_MIN_GAP_RATIO,
    score_smooth_window: int = DEFAULT_SCORE_SMOOTH_WINDOW,
    frame_diff_metric: str = DEFAULT_FRAME_DIFF_METRIC,
) -> Tuple[List[Image.Image], List[int], int]:
    require_cv2()
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_sampling_mode == "diff_guided":
        target_indices = sample_diff_guided_frame_indices(
            capture=capture,
            total_frames=total_frames,
            num_frames=num_frames,
            sampling_window_start=sampling_window_start,
            sampling_window_end=sampling_window_end,
            diff_alpha=diff_alpha,
            diff_beta=diff_beta,
            min_gap_ratio=min_gap_ratio,
            score_smooth_window=score_smooth_window,
            frame_diff_metric=frame_diff_metric,
        )
    else:
        target_indices = sample_frame_indices(
            total_frames,
            num_frames,
            frame_sampling_mode=frame_sampling_mode,
            sampling_window_start=sampling_window_start,
            sampling_window_end=sampling_window_end,
        )
    images: List[Image.Image] = []

    for frame_index in target_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))

    if not images:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))

    capture.release()

    if not images:
        raise RuntimeError(f"Failed to decode frames from video: {video_path}")
    if num_frames > 0 and len(images) < num_frames:
        last = images[-1]
        while len(images) < num_frames:
            images.append(last.copy())
    if num_frames > 0 and len(target_indices) < num_frames:
        last_index = target_indices[-1] if target_indices else 0
        while len(target_indices) < num_frames:
            target_indices.append(last_index)
    return images, target_indices, total_frames


def _frame_path_sort_key(image_path: str) -> Tuple[int, str]:
    stem = Path(image_path).stem
    tail = stem.rsplit("-", 1)[-1]
    number_token = tail.split("_")[0]
    try:
        return int(number_token), stem
    except ValueError:
        return 0, stem


def read_sampled_image_sequence(
    image_paths: List[str],
    num_frames: int,
    frame_sampling_mode: str = "uniform",
    sampling_window_start: float = DEFAULT_SAMPLING_WINDOW_START,
    sampling_window_end: float = DEFAULT_SAMPLING_WINDOW_END,
) -> Tuple[List[Image.Image], List[int], int]:
    ordered_paths = sorted([str(path) for path in image_paths], key=_frame_path_sort_key)
    total_frames = len(ordered_paths)
    if total_frames <= 0:
        raise RuntimeError("No image frames available in image sequence")

    target_indices = sample_frame_indices(
        total_frames,
        num_frames,
        frame_sampling_mode="middle_late" if frame_sampling_mode == "middle_late" else "uniform",
        sampling_window_start=sampling_window_start,
        sampling_window_end=sampling_window_end,
    )

    images: List[Image.Image] = []
    selected_indices: List[int] = []
    for frame_index in target_indices:
        frame_index = max(0, min(frame_index, total_frames - 1))
        with Image.open(ordered_paths[frame_index]) as image:
            images.append(image.convert("RGB"))
        selected_indices.append(frame_index)

    if num_frames > 0 and len(images) < num_frames:
        last = images[-1]
        while len(images) < num_frames:
            images.append(last.copy())
    if num_frames > 0 and len(selected_indices) < num_frames:
        last_index = selected_indices[-1] if selected_indices else 0
        while len(selected_indices) < num_frames:
            selected_indices.append(last_index)
    return images, selected_indices, total_frames


def read_sampled_media(
    sample: Dict,
    num_frames: int,
    frame_sampling_mode: str = "uniform",
    sampling_window_start: float = DEFAULT_SAMPLING_WINDOW_START,
    sampling_window_end: float = DEFAULT_SAMPLING_WINDOW_END,
    diff_alpha: float = DEFAULT_DIFF_ALPHA,
    diff_beta: float = DEFAULT_DIFF_BETA,
    min_gap_ratio: float = DEFAULT_MIN_GAP_RATIO,
    score_smooth_window: int = DEFAULT_SCORE_SMOOTH_WINDOW,
    frame_diff_metric: str = DEFAULT_FRAME_DIFF_METRIC,
) -> Tuple[List[Image.Image], List[int], int]:
    frame_paths = sample.get("frame_paths")
    if frame_paths:
        return read_sampled_image_sequence(
            image_paths=list(frame_paths),
            num_frames=num_frames,
            frame_sampling_mode=frame_sampling_mode,
            sampling_window_start=sampling_window_start,
            sampling_window_end=sampling_window_end,
        )
    return read_sampled_frames(
        sample["video_path"],
        num_frames,
        frame_sampling_mode=frame_sampling_mode,
        sampling_window_start=sampling_window_start,
        sampling_window_end=sampling_window_end,
        diff_alpha=diff_alpha,
        diff_beta=diff_beta,
        min_gap_ratio=min_gap_ratio,
        score_smooth_window=score_smooth_window,
        frame_diff_metric=frame_diff_metric,
    )


def read_middle_frame(video_path: str, frame_sampling_mode: str = "uniform") -> Image.Image:
    frames, _, _ = read_sampled_frames(video_path, 1, frame_sampling_mode=frame_sampling_mode)
    return frames[0]


def sanitize_model_id(model_id: str) -> str:
    safe = str(model_id).replace("/", "_").replace(":", "_").replace(" ", "_")
    return safe[:80]


def build_split_samples_with_index(samples: List[Dict], split_name: str) -> List[Dict]:
    indexed_samples: List[Dict] = []
    for sample_index, sample in enumerate(samples):
        enriched = dict(sample)
        enriched["split_name"] = split_name
        enriched["sample_index"] = sample_index
        indexed_samples.append(enriched)
    return indexed_samples


def compute_samples_fingerprint(samples: List[Dict]) -> str:
    payload = [
        {
            "sample_index": int(sample["sample_index"]),
            "sequence_id": sample["sequence_id"],
            "video_path": sample.get("video_path"),
            "frame_dir": sample.get("frame_dir"),
            "frame_paths": list(sample.get("frame_paths", [])),
        }
        for sample in samples
    ]
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def build_extraction_cache_config(
    dataset_name: str,
    split_name: str,
    samples: List[Dict],
    model_id: str,
    num_frames: int,
    frame_sampling_mode: str,
    video_extensions: List[str],
    feature_layout: str,
    sampling_window_start: float,
    sampling_window_end: float,
    diff_alpha: float,
    diff_beta: float,
    min_gap_ratio: float,
    score_smooth_window: int,
    frame_diff_metric: str,
) -> Dict[str, object]:
    sampling_config = build_sampling_config_payload(
        frame_sampling_mode=frame_sampling_mode,
        sampling_window_start=sampling_window_start,
        sampling_window_end=sampling_window_end,
        diff_alpha=diff_alpha,
        diff_beta=diff_beta,
        min_gap_ratio=min_gap_ratio,
        score_smooth_window=score_smooth_window,
        frame_diff_metric=frame_diff_metric,
    )
    return {
        "cache_version": FEATURE_CACHE_VERSION,
        "dataset": dataset_name,
        "split_name": split_name,
        "model_id": model_id,
        "num_frames": int(num_frames),
        "frame_sampling_mode": frame_sampling_mode,
        "feature_layout": feature_layout,
        "sampling_window_start": sampling_config["sampling_window_start"],
        "sampling_window_end": sampling_config["sampling_window_end"],
        "diff_alpha": sampling_config["diff_alpha"],
        "diff_beta": sampling_config["diff_beta"],
        "min_gap_ratio": sampling_config["min_gap_ratio"],
        "score_smooth_window": sampling_config["score_smooth_window"],
        "frame_diff_metric": sampling_config["frame_diff_metric"],
        "sampling_signature": compute_sampling_signature(sampling_config),
        "video_extensions": list(video_extensions),
        "sample_count": len(samples),
        "samples_fingerprint": compute_samples_fingerprint(samples),
    }


def build_feature_cache_name(
    dataset_name: str,
    split_name: str,
    model_id: str,
    num_frames: int,
    frame_sampling_mode: str,
    feature_layout: str,
    sample_count: int,
    samples_fingerprint: str,
    sampling_signature: str,
    shard_index: Optional[int] = None,
    total_shards: Optional[int] = None,
) -> str:
    base = (
        f"{dataset_name.lower()}_{split_name}_{sanitize_model_id(model_id)}_"
        f"f{num_frames}_{frame_sampling_mode}_{sampling_signature}_n{sample_count}_{samples_fingerprint}"
    )
    if feature_layout != "pooled":
        base = f"{base}_{feature_layout}"
    if shard_index is not None and total_shards is not None:
        return f"{base}_shard{shard_index}of{total_shards}.pt"
    return f"{base}.pt"


def build_feature_cache_path(
    feature_cache_dir: str,
    dataset_name: str,
    split_name: str,
    model_id: str,
    num_frames: int,
    frame_sampling_mode: str,
    feature_layout: str,
    sample_count: int,
    samples_fingerprint: str,
    sampling_signature: str,
    shard_index: Optional[int] = None,
    total_shards: Optional[int] = None,
) -> Path:
    cache_root = Path(feature_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / build_feature_cache_name(
        dataset_name=dataset_name,
        split_name=split_name,
        model_id=model_id,
        num_frames=num_frames,
        frame_sampling_mode=frame_sampling_mode,
        feature_layout=feature_layout,
        sample_count=sample_count,
        samples_fingerprint=samples_fingerprint,
        sampling_signature=sampling_signature,
        shard_index=shard_index,
        total_shards=total_shards,
    )


def build_failure_log_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".failures.json")


def count_existing_shards(cache_plan: Dict[str, object]) -> int:
    return sum(1 for path in cache_plan["shard_paths"] if path.exists())


def list_missing_shards(cache_plan: Dict[str, object]) -> List[Path]:
    return [path for path in cache_plan["shard_paths"] if not path.exists()]


def validate_feature_cache_payload(cache_payload: Dict, expected_config: Dict[str, object], cache_path: Path) -> None:
    cache_config = dict(cache_payload.get("config", {}))
    if "feature_layout" not in cache_config:
        cache_config["feature_layout"] = "pooled"
    expected_subset = {
        "cache_version": expected_config["cache_version"],
        "dataset": expected_config["dataset"],
        "split_name": expected_config["split_name"],
        "model_id": expected_config["model_id"],
        "num_frames": expected_config["num_frames"],
        "frame_sampling_mode": expected_config["frame_sampling_mode"],
        "feature_layout": expected_config.get("feature_layout", "pooled"),
        "sampling_window_start": expected_config.get("sampling_window_start"),
        "sampling_window_end": expected_config.get("sampling_window_end"),
        "diff_alpha": expected_config.get("diff_alpha"),
        "diff_beta": expected_config.get("diff_beta"),
        "min_gap_ratio": expected_config.get("min_gap_ratio"),
        "score_smooth_window": expected_config.get("score_smooth_window"),
        "frame_diff_metric": expected_config.get("frame_diff_metric"),
        "sampling_signature": expected_config.get("sampling_signature"),
        "video_extensions": list(expected_config["video_extensions"]),
        "sample_count": expected_config["sample_count"],
        "samples_fingerprint": expected_config["samples_fingerprint"],
    }
    mismatches = {}
    for key, expected_value in expected_subset.items():
        observed_value = cache_config.get(key)
        if observed_value != expected_value:
            mismatches[key] = {"expected": expected_value, "observed": observed_value}
    if mismatches:
        raise RuntimeError(
            f"Feature cache config mismatch for {cache_path}: {json.dumps(mismatches, ensure_ascii=False)}"
        )


def resolve_existing_split_cache(cache_plan: Dict[str, object], split_name: str) -> Optional[Path]:
    final_path = cache_plan["final_path"]
    if final_path.exists():
        payload = load_split_feature_cache(final_path)
        validate_feature_cache_payload(payload, cache_plan["config"], final_path)
        log(f"[CACHE] feature cache hit for {split_name}: {final_path}")
        return final_path
    return None


def decode_sample_frames(
    sample: Dict,
    num_frames: int,
    frame_sampling_mode: str,
    split_name: str,
    sampling_window_start: float,
    sampling_window_end: float,
    diff_alpha: float,
    diff_beta: float,
    min_gap_ratio: float,
    score_smooth_window: int,
    frame_diff_metric: str,
) -> Dict[str, object]:
    try:
        frames, selected_indices, total_frames = read_sampled_frames(
            sample["video_path"],
            num_frames,
            frame_sampling_mode=frame_sampling_mode,
            sampling_window_start=sampling_window_start,
            sampling_window_end=sampling_window_end,
            diff_alpha=diff_alpha,
            diff_beta=diff_beta,
            min_gap_ratio=min_gap_ratio,
            score_smooth_window=score_smooth_window,
            frame_diff_metric=frame_diff_metric,
        )
        if not frames:
            raise RuntimeError("No decoded frames")
        return {
            "sample": sample,
            "frames": frames,
            "selected_indices": selected_indices,
            "total_frames": total_frames,
            "error": None,
        }
    except Exception as exc:
        return {
            "sample": sample,
            "frames": None,
            "selected_indices": [],
            "total_frames": 0,
            "error": {
                "sequence_id": sample.get("sequence_id"),
                "video_path": sample.get("video_path"),
                "label": sample.get("label"),
                "actor_id": sample.get("actor_id"),
                "split_name": split_name,
                "sample_index": sample.get("sample_index"),
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def prepare_image_inputs(processor, images, device: str, pin_memory: bool):
    import torch

    normalized_images = list(images)
    while normalized_images and all(isinstance(item, (list, tuple)) for item in normalized_images):
        flattened = []
        for item in normalized_images:
            flattened.extend(item)
        normalized_images = flattened
    if not normalized_images:
        raise ValueError("No images available for CLIP preprocessing")
    if any(isinstance(item, (list, tuple)) for item in normalized_images):
        raise ValueError(f"Unexpected nested image batch structure: sample_type={type(normalized_images[0])}")

    inputs = processor(images=normalized_images, return_tensors="pt", padding=True)
    tensor_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if pin_memory and str(device).startswith("cuda") and hasattr(value, "pin_memory"):
                value = value.pin_memory()
            tensor_inputs[key] = value.to(device, non_blocking=pin_memory)
        else:
            tensor_inputs[key] = value
    return tensor_inputs


def build_split_cache_plan(feature_cache_dir: str, dataset_name: str, split_name: str, samples: List[Dict], args) -> Dict[str, object]:
    config = build_extraction_cache_config(
        dataset_name=dataset_name,
        split_name=split_name,
        samples=samples,
        model_id=args.model_id,
        num_frames=args.num_frames,
        frame_sampling_mode=args.frame_sampling_mode,
        video_extensions=sorted(parse_extension_set(args.video_extensions)),
        feature_layout=args.feature_layout,
        sampling_window_start=args.sampling_window_start,
        sampling_window_end=args.sampling_window_end,
        diff_alpha=args.diff_alpha,
        diff_beta=args.diff_beta,
        min_gap_ratio=args.min_gap_ratio,
        score_smooth_window=args.score_smooth_window,
        frame_diff_metric=args.frame_diff_metric,
    )
    final_path = build_feature_cache_path(
        feature_cache_dir=feature_cache_dir,
        dataset_name=dataset_name,
        split_name=split_name,
        model_id=args.model_id,
        num_frames=args.num_frames,
        frame_sampling_mode=args.frame_sampling_mode,
        feature_layout=args.feature_layout,
        sample_count=len(samples),
        samples_fingerprint=str(config["samples_fingerprint"]),
        sampling_signature=str(config["sampling_signature"]),
    )
    shard_paths = [
        build_feature_cache_path(
            feature_cache_dir=feature_cache_dir,
            dataset_name=dataset_name,
            split_name=split_name,
            model_id=args.model_id,
            num_frames=args.num_frames,
            frame_sampling_mode=args.frame_sampling_mode,
            feature_layout=args.feature_layout,
            sample_count=len(samples),
            samples_fingerprint=str(config["samples_fingerprint"]),
            sampling_signature=str(config["sampling_signature"]),
            shard_index=shard_index,
            total_shards=args.total_shards,
        )
        for shard_index in range(args.total_shards)
    ]
    return {
        "config": config,
        "final_path": final_path,
        "shard_paths": shard_paths,
    }


def shard_samples(samples: List[Dict], shard_index: int, total_shards: int) -> List[Dict]:
    return samples[shard_index::total_shards]


def extract_image_features_with_metadata(
    samples: List[Dict],
    processor,
    model,
    device: str,
    batch_size: int,
    num_frames: int,
    split_name: str,
    frame_sampling_mode: str,
    feature_layout: str,
    num_workers: int,
    pin_memory: bool,
    sampling_window_start: float,
    sampling_window_end: float,
    diff_alpha: float,
    diff_beta: float,
    min_gap_ratio: float,
    score_smooth_window: int,
    frame_diff_metric: str,
):
    import torch

    feature_dim = int(getattr(model.config, "projection_dim", 512))
    if not samples:
        empty_shape = (0, feature_dim) if feature_layout == "pooled" else (0, num_frames, feature_dim)
        return {
            "features": torch.empty(empty_shape, dtype=torch.float32),
            "samples": [],
            "failed_samples": [],
            "sampling_debug": [],
        }

    feats = []
    kept_samples: List[Dict] = []
    failed_samples: List[Dict] = []
    sampling_debug: List[Dict[str, object]] = []
    total_batches = max(1, math.ceil(len(samples) / batch_size))
    start_time = time.time()
    log(
        f"[FEATURES] start {split_name}: samples={len(samples)}, batches={total_batches}, "
        f"num_frames={num_frames}, frame_sampling_mode={frame_sampling_mode}, feature_layout={feature_layout}"
    )
    for batch_idx, start in enumerate(range(0, len(samples), batch_size), start=1):
        batch = samples[start:start + batch_size]
        valid_samples: List[Dict] = []
        frame_groups: List[List] = []
        if num_workers and num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                decoded_items = list(
                    executor.map(
                        lambda sample: decode_sample_frames(
                            sample,
                            num_frames,
                            frame_sampling_mode,
                            split_name,
                            sampling_window_start,
                            sampling_window_end,
                            diff_alpha,
                            diff_beta,
                            min_gap_ratio,
                            score_smooth_window,
                            frame_diff_metric,
                        ),
                        batch,
                    )
                )
        else:
            decoded_items = [
                decode_sample_frames(
                    sample,
                    num_frames,
                    frame_sampling_mode,
                    split_name,
                    sampling_window_start,
                    sampling_window_end,
                    diff_alpha,
                    diff_beta,
                    min_gap_ratio,
                    score_smooth_window,
                    frame_diff_metric,
                )
                for sample in batch
            ]

        for decoded in decoded_items:
            if decoded["error"] is not None:
                failed_samples.append(decoded["error"])
                continue
            valid_samples.append(decoded["sample"])
            frame_groups.append(decoded["frames"])
            if len(sampling_debug) < DEFAULT_SAMPLING_DEBUG_SAMPLES:
                sampling_debug.append(
                    {
                        "split_name": split_name,
                        "sample_index": decoded["sample"].get("sample_index"),
                        "sequence_id": decoded["sample"].get("sequence_id"),
                        "video_path": decoded["sample"].get("video_path"),
                        "total_frames": decoded.get("total_frames", 0),
                        "selected_indices": list(decoded.get("selected_indices", [])),
                    }
                )

        if valid_samples:
            flat_images = [image for images in frame_groups for image in images]
            frame_count = len(frame_groups[0])
            inputs = prepare_image_inputs(processor, flat_images, device=device, pin_memory=pin_memory)
            with torch.no_grad():
                autocast_enabled = str(device).startswith("cuda")
                with torch.cuda.amp.autocast(enabled=autocast_enabled):
                    image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            image_features = image_features.view(len(valid_samples), frame_count, -1)
            if feature_layout == "sequence":
                feats.append(image_features.float().cpu())
            else:
                pooled = image_features.mean(dim=1)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                feats.append(pooled.float().cpu())
            kept_samples.extend(valid_samples)

        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            elapsed = time.time() - start_time
            eta = elapsed / batch_idx * (total_batches - batch_idx) if batch_idx < total_batches else 0.0
            log(
                f"[FEATURES] {split_name}: batch {batch_idx}/{total_batches}, kept={len(kept_samples)}, "
                f"failed={len(failed_samples)}, elapsed={format_duration(elapsed)}, eta={format_duration(eta)}"
            )

    if feats:
        feature_tensor = torch.cat(feats, dim=0)
    else:
        empty_shape = (0, feature_dim) if feature_layout == "pooled" else (0, num_frames, feature_dim)
        feature_tensor = torch.empty(empty_shape, dtype=torch.float32)
    return {
        "features": feature_tensor,
        "samples": kept_samples,
        "failed_samples": failed_samples,
        "sampling_debug": sampling_debug,
    }


def save_failure_log(failure_log_path: Path, failed_samples: List[Dict]) -> None:
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_log_path.open("w", encoding="utf-8") as f:
        json.dump(failed_samples, f, ensure_ascii=False, indent=2)


def save_split_feature_cache(cache_path: Path, payload: Dict) -> None:
    import torch

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def load_split_feature_cache(cache_path: Path) -> Dict:
    import torch

    return torch.load(cache_path, map_location="cpu")


def extract_feature_shard(samples: List[Dict], processor, model, args, split_name: str, cache_plan: Dict[str, object]) -> Path:
    shard_path = cache_plan["shard_paths"][args.shard_index]
    final_path = cache_plan["final_path"]
    if final_path.exists():
        payload = load_split_feature_cache(final_path)
        validate_feature_cache_payload(payload, cache_plan["config"], final_path)
        log(f"[CACHE] final merged cache already exists for {split_name}, skip shard extraction: {final_path}")
        return final_path
    if shard_path.exists():
        payload = load_split_feature_cache(shard_path)
        validate_feature_cache_payload(payload, cache_plan["config"], shard_path)
        log(f"[CACHE] shard cache hit for {split_name} shard {args.shard_index}/{args.total_shards}: {shard_path}")
        return shard_path

    existing_shards = count_existing_shards(cache_plan)
    log(
        f"[CACHE] shard extraction state for {split_name}: existing_shards={existing_shards}/{args.total_shards}, "
        f"target_shard={args.shard_index}"
    )
    shard_sample_list = shard_samples(samples, shard_index=args.shard_index, total_shards=args.total_shards)
    log(
        f"[SHARD] extracting split={split_name} shard={args.shard_index}/{args.total_shards} "
        f"samples={len(shard_sample_list)} device={args.device} output={shard_path}"
    )
    extracted = extract_image_features_with_metadata(
        shard_sample_list,
        processor=processor,
        model=model,
        device=args.device,
        batch_size=args.extract_batch_size,
        num_frames=args.num_frames,
        split_name=f"{split_name}_shard{args.shard_index}",
        frame_sampling_mode=args.frame_sampling_mode,
        feature_layout=args.feature_layout,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        sampling_window_start=args.sampling_window_start,
        sampling_window_end=args.sampling_window_end,
        diff_alpha=args.diff_alpha,
        diff_beta=args.diff_beta,
        min_gap_ratio=args.min_gap_ratio,
        score_smooth_window=args.score_smooth_window,
        frame_diff_metric=args.frame_diff_metric,
    )
    failure_log_path = build_failure_log_path(shard_path)
    save_failure_log(failure_log_path, extracted["failed_samples"])
    payload = {
        "cache_type": "feature_shard_cache",
        "dataset": "RAVDESS",
        "split_name": split_name,
        "config": cache_plan["config"],
        "shard_index": args.shard_index,
        "total_shards": args.total_shards,
        "sample_count": len(extracted["samples"]),
        "failed_count": len(extracted["failed_samples"]),
        "features": extracted["features"],
        "samples": extracted["samples"],
        "failed_samples": extracted["failed_samples"],
        "sampling_debug": extracted.get("sampling_debug", []),
        "failure_log_path": str(failure_log_path),
    }
    save_split_feature_cache(shard_path, payload)
    log(f"[CACHE] saved shard feature cache: {shard_path}")
    return shard_path


def merge_feature_shards(cache_plan: Dict[str, object], delete_shards_after_merge: bool = False) -> Path:
    final_path = cache_plan["final_path"]
    expected_shards = cache_plan["shard_paths"]
    missing = [str(path) for path in list_missing_shards(cache_plan)]
    if missing:
        raise FileNotFoundError(f"Missing shard caches for merge: {missing}")
    if final_path.exists():
        payload = load_split_feature_cache(final_path)
        validate_feature_cache_payload(payload, cache_plan["config"], final_path)
        log(f"[CACHE] merged split cache already exists: {final_path}")
        return final_path

    all_entries: List[Tuple[int, Dict, object]] = []
    failed_samples: List[Dict] = []
    sampling_debug: List[Dict[str, object]] = []
    split_name = None
    for shard_path in expected_shards:
        shard_payload = load_split_feature_cache(shard_path)
        validate_feature_cache_payload(shard_payload, cache_plan["config"], shard_path)
        split_name = split_name or shard_payload.get("split_name")
        features = shard_payload["features"]
        shard_samples = shard_payload["samples"]
        if int(features.shape[0]) != len(shard_samples):
            raise RuntimeError(f"Feature/sample count mismatch in shard: {shard_path}")
        failed_samples.extend(list(shard_payload.get("failed_samples", [])))
        sampling_debug.extend(list(shard_payload.get("sampling_debug", [])))
        for row_idx, sample in enumerate(shard_samples):
            all_entries.append((int(sample["sample_index"]), sample, features[row_idx]))

    all_entries.sort(key=lambda item: item[0])
    duplicate_indices = [idx for idx, count in Counter(item[0] for item in all_entries).items() if count > 1]
    if duplicate_indices:
        raise RuntimeError(f"Duplicate sample_index values found while merging shards: {duplicate_indices[:10]}")

    import torch

    merged_samples = [item[1] for item in all_entries]
    if all_entries:
        merged_features = torch.stack([item[2] for item in all_entries], dim=0).cpu()
    else:
        merged_features = torch.empty((0, 0), dtype=torch.float32)
    payload = {
        "cache_type": "feature_split_cache",
        "dataset": "RAVDESS",
        "split_name": split_name,
        "config": cache_plan["config"],
        "sample_count": len(merged_samples),
        "failed_count": len(failed_samples),
        "features": merged_features,
        "samples": merged_samples,
        "failed_samples": failed_samples,
        "sampling_debug": sampling_debug[:DEFAULT_SAMPLING_DEBUG_SAMPLES],
        "source_shards": [str(path) for path in expected_shards],
    }
    save_split_feature_cache(final_path, payload)
    save_failure_log(build_failure_log_path(final_path), failed_samples)
    log(f"[CACHE] merged {len(expected_shards)} shards into final split cache: {final_path}")

    if delete_shards_after_merge:
        for shard_path in expected_shards:
            shard_path.unlink(missing_ok=True)
            build_failure_log_path(shard_path).unlink(missing_ok=True)
        log(f"[CACHE] deleted shard caches after merge for {split_name}")
    return final_path


def extract_split_feature_cache(samples: List[Dict], processor, model, args, split_name: str, cache_plan: Dict[str, object]) -> Path:
    final_path = cache_plan["final_path"]
    if final_path.exists():
        payload = load_split_feature_cache(final_path)
        validate_feature_cache_payload(payload, cache_plan["config"], final_path)
        log(f"[CACHE] feature cache hit for {split_name}: {final_path}")
        return final_path

    log(f"[CACHE] feature cache miss for {split_name}: {final_path}")
    extracted = extract_image_features_with_metadata(
        samples,
        processor=processor,
        model=model,
        device=args.device,
        batch_size=args.extract_batch_size,
        num_frames=args.num_frames,
        split_name=split_name,
        frame_sampling_mode=args.frame_sampling_mode,
        feature_layout=args.feature_layout,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        sampling_window_start=args.sampling_window_start,
        sampling_window_end=args.sampling_window_end,
        diff_alpha=args.diff_alpha,
        diff_beta=args.diff_beta,
        min_gap_ratio=args.min_gap_ratio,
        score_smooth_window=args.score_smooth_window,
        frame_diff_metric=args.frame_diff_metric,
    )
    payload = {
        "cache_type": "feature_split_cache",
        "dataset": "RAVDESS",
        "split_name": split_name,
        "config": cache_plan["config"],
        "sample_count": len(extracted["samples"]),
        "failed_count": len(extracted["failed_samples"]),
        "features": extracted["features"],
        "samples": extracted["samples"],
        "failed_samples": extracted["failed_samples"],
        "sampling_debug": extracted.get("sampling_debug", []),
        "source_shards": [],
    }
    save_split_feature_cache(final_path, payload)
    save_failure_log(build_failure_log_path(final_path), extracted["failed_samples"])
    log(f"[CACHE] saved final split cache for {split_name}: {final_path}")
    return final_path


def load_features_and_labels_from_split_cache(cache_path: Path, label2idx: Dict[str, int]):
    import torch

    payload = load_split_feature_cache(cache_path)
    samples = payload.get("samples", [])
    labels = torch.tensor([label2idx[sample["label"]] for sample in samples], dtype=torch.long)
    return payload["features"], labels, samples, payload


def load_legacy_strict_feature_cache(cache_path: Path) -> Dict:
    import torch

    payload = torch.load(cache_path, map_location="cpu")
    required_keys = ["train_x", "train_y", "val_x", "val_y", "test_x", "test_y", "text_features"]
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise RuntimeError(f"Legacy strict cache missing keys at {cache_path}: {missing_keys}")
    return payload


def validate_legacy_cache_split_alignment(
    legacy_payload: Dict,
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    label2idx: Dict[str, int],
) -> None:
    import torch

    split_specs = [
        ("train", train_samples, "train_y"),
        ("val", val_samples, "val_y"),
        ("test", test_samples, "test_y"),
    ]
    for split_name, split_samples, label_key in split_specs:
        expected = torch.tensor([label2idx[sample["label"]] for sample in split_samples], dtype=torch.long)
        observed = legacy_payload[label_key].detach().cpu().long()
        if expected.shape[0] != observed.shape[0]:
            raise RuntimeError(
                f"Legacy strict cache sample count mismatch for {split_name}: expected={expected.shape[0]} observed={observed.shape[0]}"
            )
        if not torch.equal(expected, observed):
            mismatch_idx = int((expected != observed).nonzero(as_tuple=False).view(-1)[0].item())
            raise RuntimeError(
                f"Legacy strict cache label order mismatch for {split_name} at index {mismatch_idx}: "
                f"expected={int(expected[mismatch_idx].item())} observed={int(observed[mismatch_idx].item())}"
            )


def ensure_training_split_cache(
    split_name: str,
    split_samples: List[Dict],
    cache_plan: Dict[str, object],
    processor,
    model,
    args,
) -> Path:
    existing_final = resolve_existing_split_cache(cache_plan, split_name)
    if existing_final is not None:
        return existing_final

    existing_shards = count_existing_shards(cache_plan)
    if args.total_shards > 1:
        if existing_shards == args.total_shards:
            return merge_feature_shards(cache_plan, delete_shards_after_merge=args.delete_shards_after_merge)
        if existing_shards > 0:
            missing = [str(path.name) for path in list_missing_shards(cache_plan)]
            raise RuntimeError(
                f"Partial shard cache state detected for {split_name}: "
                f"{existing_shards}/{args.total_shards} shard files present, missing={missing}. "
                f"Finish missing shards before training or run with total_shards=1."
            )

    return extract_split_feature_cache(
        split_samples,
        processor=processor,
        model=model,
        args=args,
        split_name=split_name,
        cache_plan=cache_plan,
    )


def extract_text_features(prompt_groups: List[List[str]], processor, model, device: str):
    import torch

    class_prompt_features = []
    log(
        f"[TEXT] start text feature extraction: classes={len(prompt_groups)}, "
        f"prompts_per_class={len(prompt_groups[0]) if prompt_groups else 0}"
    )
    for prompts in prompt_groups:
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        class_prompt_features.append(text_features)
    log("[TEXT] done text feature extraction")
    return torch.stack(class_prompt_features, dim=0).float().detach()


def compute_text_prototype_similarity(text_features) -> Dict[str, Dict[str, float]]:
    import torch

    if int(text_features.shape[0]) == 0:
        return round_float_dict_matrix([], [])
    prototypes = text_features.mean(dim=1)
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    similarity = prototypes @ prototypes.transpose(0, 1)
    return round_float_dict_matrix(similarity.detach().cpu().tolist(), EMOTION_LABELS)


class CAPCHead:
    def __init__(
        self,
        device: str,
        num_classes: int,
        num_prompts: int,
        use_global_logit_scale: bool = False,
        use_prompt_weight: bool = True,
        use_class_temperature: bool = True,
        use_class_bias: bool = True,
    ):
        import torch
        import torch.nn as nn

        self.device = device
        self.logit_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.prompt_weight_logits = nn.Parameter(torch.zeros(num_classes, num_prompts, device=device))
        self.class_logit_scale = nn.Parameter(torch.zeros(num_classes, device=device))
        self.class_bias = nn.Parameter(torch.zeros(num_classes, device=device))
        self.use_global_logit_scale = use_global_logit_scale
        self.use_prompt_weight = use_prompt_weight
        self.use_class_temperature = use_class_temperature
        self.use_class_bias = use_class_bias

    def parameters(self):
        params = []
        if self.use_global_logit_scale:
            params.append(self.logit_scale)
        if self.use_prompt_weight:
            params.append(self.prompt_weight_logits)
        if self.use_class_temperature:
            params.append(self.class_logit_scale)
        if self.use_class_bias:
            params.append(self.class_bias)
        return params

    def state_dict(self):
        return {
            "logit_scale": self.logit_scale.detach().cpu().clone(),
            "prompt_weight_logits": self.prompt_weight_logits.detach().cpu().clone(),
            "class_logit_scale": self.class_logit_scale.detach().cpu().clone(),
            "class_bias": self.class_bias.detach().cpu().clone(),
            "use_global_logit_scale": self.use_global_logit_scale,
            "use_prompt_weight": self.use_prompt_weight,
            "use_class_temperature": self.use_class_temperature,
            "use_class_bias": self.use_class_bias,
        }

    def load_state_dict(self, state):
        self.logit_scale.data.copy_(state["logit_scale"].to(self.device))
        self.prompt_weight_logits.data.copy_(state["prompt_weight_logits"].to(self.device))
        self.class_logit_scale.data.copy_(state["class_logit_scale"].to(self.device))
        self.class_bias.data.copy_(state["class_bias"].to(self.device))
        self.use_global_logit_scale = state.get("use_global_logit_scale", False)
        self.use_prompt_weight = state.get("use_prompt_weight", True)
        self.use_class_temperature = state.get("use_class_temperature", True)
        self.use_class_bias = state.get("use_class_bias", True)

    def logits(self, adapted_features, text_x):
        import torch
        import torch.nn.functional as F

        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bd,cpd->bcp", adapted_features, txt)

        if self.use_prompt_weight:
            prompt_w = F.softmax(self.prompt_weight_logits, dim=-1).unsqueeze(0)
            class_sim = (sim * prompt_w).sum(dim=-1)
        else:
            class_sim = sim.mean(dim=-1)

        if self.use_global_logit_scale:
            global_scale = self.logit_scale.exp().clamp(max=100.0)
        else:
            global_scale = 1.0
        if self.use_class_temperature:
            class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).unsqueeze(0)
        else:
            class_scale = 1.0
        if self.use_class_bias:
            class_bias = self.class_bias.unsqueeze(0)
        else:
            class_bias = 0.0
        return global_scale * class_sim * class_scale + class_bias

    def grouped_logits(self, adapted_features, text_x, group_indices: List[List[int]]):
        import torch

        txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bd,cpd->bcp", adapted_features, txt)

        group_scores = []
        for gidx in group_indices:
            group_scores.append(sim[:, :, gidx].mean(dim=-1))
        scores = torch.stack(group_scores, dim=-1)

        if self.use_global_logit_scale:
            global_scale = self.logit_scale.exp().clamp(max=100.0)
        else:
            global_scale = 1.0
        if self.use_class_temperature:
            class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).view(1, -1, 1)
        else:
            class_scale = 1.0
        if self.use_class_bias:
            class_bias = self.class_bias.view(1, -1, 1)
        else:
            class_bias = 0.0
        return global_scale * scores * class_scale + class_bias


class ClipImageAdapter:
    def __init__(
        self,
        dim: int,
        device: str,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        num_prompts: int,
        use_qcpa: bool = True,
        qcpa_num_heads: int = 4,
        use_global_logit_scale: bool = False,
        use_prompt_weight: bool = True,
        use_class_temperature: bool = True,
        use_class_bias: bool = True,
        attention_scope: str = "local",
        dual_stage: bool = False,
        use_residual_gate: bool = True,
        use_mahalanobis_temp: bool = True,
        use_lowrank_bias: bool = True,
        bias_rank: int = 16,
    ):
        import torch
        import torch.nn as nn

        self.device = device
        self.input_proj = nn.Linear(dim, hidden_dim).to(device)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        self.out_proj = nn.Linear(hidden_dim, dim).to(device)
        self.use_qcpa = bool(use_qcpa)
        if self.use_qcpa:
            self.head = QCPAHead(
                feat_dim=dim,
                num_classes=num_classes,
                num_prompts=num_prompts,
                num_heads=qcpa_num_heads,
                attention_scope=attention_scope,
                dual_stage=dual_stage,
                use_residual_gate=use_residual_gate,
                use_mahalanobis_temp=use_mahalanobis_temp,
                use_lowrank_bias=use_lowrank_bias,
                bias_rank=bias_rank,
            ).to(device)
            self.legacy_head = None
        else:
            self.head = None
            self.legacy_head = CAPCHead(
                device=device,
                num_classes=num_classes,
                num_prompts=num_prompts,
                use_global_logit_scale=use_global_logit_scale,
                use_prompt_weight=use_prompt_weight,
                use_class_temperature=use_class_temperature,
                use_class_bias=use_class_bias,
            )

    def parameters(self):
        return self.adapter_parameters() + self.head_parameters()

    def adapter_parameters(self):
        return list(self.input_proj.parameters()) + list(self.net.parameters()) + list(self.out_proj.parameters())

    def taga_parameters(self):
        return []

    def head_parameters(self):
        return self.qcpa_parameters() if self.use_qcpa else self.capc_parameters()

    def qcpa_parameters(self):
        return list(self.head.parameters()) if self.use_qcpa and self.head is not None else []

    def capc_parameters(self):
        return self.legacy_head.parameters() if self.legacy_head is not None else []

    def state_dict(self):
        state = {
            "input_proj": self.input_proj.state_dict(),
            "net": self.net.state_dict(),
            "out_proj": self.out_proj.state_dict(),
            "use_qcpa": self.use_qcpa,
        }
        if self.use_qcpa and self.head is not None:
            state["head"] = self.head.state_dict()
        elif self.legacy_head is not None:
            state["legacy_head"] = self.legacy_head.state_dict()
        return state

    def load_state_dict(self, state):
        self.input_proj.load_state_dict(state["input_proj"])
        self.net.load_state_dict(state["net"])
        self.out_proj.load_state_dict(state["out_proj"])
        if self.use_qcpa:
            self.head.load_state_dict(state["head"])
        else:
            legacy_state = state.get("legacy_head")
            if legacy_state is None:
                raise KeyError("legacy_head missing from checkpoint state")
            self.legacy_head.load_state_dict(legacy_state)

    def train(self):
        self.input_proj.train()
        self.net.train()
        self.out_proj.train()
        if self.use_qcpa and self.head is not None:
            self.head.train()

    def eval(self):
        self.input_proj.eval()
        self.net.eval()
        self.out_proj.eval()
        if self.use_qcpa and self.head is not None:
            self.head.eval()

    def _adapt_image(self, image_x):
        if image_x.ndim == 3:
            image_x = image_x.mean(dim=1)
        base = self.input_proj(image_x)
        delta = self.net(base)
        fused = base + delta
        img = self.out_proj(fused)
        return img / img.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def compute_logits_from_adapted(self, adapted_features, text_x, return_aux: bool = False):
        if self.use_qcpa:
            txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return self.head(adapted_features, txt, return_aux=return_aux)
        return self.legacy_head.logits(adapted_features, text_x)

    def logits(self, image_x, text_x, return_aux: bool = False):
        img = self._adapt_image(image_x)
        return self.compute_logits_from_adapted(img, text_x, return_aux=return_aux)

    def grouped_logits(self, image_x, text_x, group_indices: List[List[int]]):
        img = self._adapt_image(image_x)
        if self.use_qcpa:
            import torch

            txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            sim = torch.einsum("bd,cpd->bcp", img, txt)

            group_scores = []
            for gidx in group_indices:
                group_scores.append(sim[:, :, gidx].mean(dim=-1))
            scores = torch.stack(group_scores, dim=-1)
            global_scale = self.head.log_scale.exp().clamp(max=100.0).view(1, 1, 1)
            class_scale = self.head.log_tau.exp().view(1, -1, 1)
            class_bias = self.head.bias_static.view(1, -1, 1) if getattr(self.head, "bias_static", None) is not None else 0.0
            return global_scale * scores * class_scale + class_bias
        return self.legacy_head.grouped_logits(img, text_x, group_indices)


class TemporalTransformerClipImageAdapter:
    def __init__(
        self,
        dim: int,
        device: str,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        num_prompts: int,
        num_frames: int,
        temporal_num_heads: int,
        temporal_num_layers: int,
        temporal_pooling: str = "cls",
        use_qcpa: bool = True,
        qcpa_num_heads: int = 4,
        use_global_logit_scale: bool = False,
        use_prompt_weight: bool = True,
        use_class_temperature: bool = True,
        use_class_bias: bool = True,
        attention_scope: str = "local",
        dual_stage: bool = False,
        use_residual_gate: bool = True,
        use_mahalanobis_temp: bool = True,
        use_lowrank_bias: bool = True,
        bias_rank: int = 16,
    ):
        import torch
        import torch.nn as nn

        self.device = device
        self.num_frames = int(num_frames)
        self.temporal_num_heads = int(temporal_num_heads)
        self.temporal_num_layers = int(temporal_num_layers)
        self.temporal_pooling = temporal_pooling
        self.frame_proj = nn.Linear(dim, hidden_dim).to(device)
        self.frame_pos = nn.Parameter(torch.zeros(1, self.num_frames + 1, hidden_dim, device=device))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim, device=device))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.temporal_num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        ).to(device)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.temporal_num_layers).to(device)
        self.temporal_norm = nn.LayerNorm(hidden_dim).to(device)
        self.temporal_attn_pool = nn.Linear(hidden_dim, 1).to(device)
        self.temporal_fusion_gate = nn.Linear(hidden_dim * 2, 1).to(device)
        self.temporal_out = nn.Linear(hidden_dim, dim).to(device)
        self.input_proj = nn.Linear(dim, hidden_dim).to(device)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        self.out_proj = nn.Linear(hidden_dim, dim).to(device)
        self.use_qcpa = bool(use_qcpa)
        if self.use_qcpa:
            self.head = QCPAHead(
                feat_dim=dim,
                num_classes=num_classes,
                num_prompts=num_prompts,
                num_heads=qcpa_num_heads,
                attention_scope=attention_scope,
                dual_stage=dual_stage,
                use_residual_gate=use_residual_gate,
                use_mahalanobis_temp=use_mahalanobis_temp,
                use_lowrank_bias=use_lowrank_bias,
                bias_rank=bias_rank,
            ).to(device)
            self.legacy_head = None
        else:
            self.head = None
            self.legacy_head = CAPCHead(
                device=device,
                num_classes=num_classes,
                num_prompts=num_prompts,
                use_global_logit_scale=use_global_logit_scale,
                use_prompt_weight=use_prompt_weight,
                use_class_temperature=use_class_temperature,
                use_class_bias=use_class_bias,
            )

    def parameters(self):
        return self.taga_parameters() + self.adapter_parameters() + self.head_parameters()

    def taga_parameters(self):
        return (
            list(self.frame_proj.parameters())
            + [self.frame_pos, self.cls_token]
            + list(self.temporal_encoder.parameters())
            + list(self.temporal_norm.parameters())
            + list(self.temporal_attn_pool.parameters())
            + list(self.temporal_fusion_gate.parameters())
            + list(self.temporal_out.parameters())
        )

    def adapter_parameters(self):
        return list(self.input_proj.parameters()) + list(self.net.parameters()) + list(self.out_proj.parameters())

    def head_parameters(self):
        return self.qcpa_parameters() if self.use_qcpa else self.capc_parameters()

    def qcpa_parameters(self):
        return list(self.head.parameters()) if self.use_qcpa and self.head is not None else []

    def capc_parameters(self):
        return self.legacy_head.parameters() if self.legacy_head is not None else []

    def state_dict(self):
        state = {
            "frame_proj": self.frame_proj.state_dict(),
            "frame_pos": self.frame_pos.detach().cpu().clone(),
            "cls_token": self.cls_token.detach().cpu().clone(),
            "temporal_encoder": self.temporal_encoder.state_dict(),
            "temporal_norm": self.temporal_norm.state_dict(),
            "temporal_attn_pool": self.temporal_attn_pool.state_dict(),
            "temporal_fusion_gate": self.temporal_fusion_gate.state_dict(),
            "temporal_out": self.temporal_out.state_dict(),
            "input_proj": self.input_proj.state_dict(),
            "net": self.net.state_dict(),
            "out_proj": self.out_proj.state_dict(),
            "use_qcpa": self.use_qcpa,
            "num_frames": self.num_frames,
            "temporal_num_heads": self.temporal_num_heads,
            "temporal_num_layers": self.temporal_num_layers,
            "temporal_pooling": self.temporal_pooling,
        }
        if self.use_qcpa and self.head is not None:
            state["head"] = self.head.state_dict()
        elif self.legacy_head is not None:
            state["legacy_head"] = self.legacy_head.state_dict()
        return state

    def load_state_dict(self, state):
        self.frame_proj.load_state_dict(state["frame_proj"])
        self.temporal_encoder.load_state_dict(state["temporal_encoder"])
        self.temporal_norm.load_state_dict(state["temporal_norm"])
        if "temporal_attn_pool" in state:
            self.temporal_attn_pool.load_state_dict(state["temporal_attn_pool"])
        if "temporal_fusion_gate" in state:
            self.temporal_fusion_gate.load_state_dict(state["temporal_fusion_gate"])
        self.temporal_out.load_state_dict(state["temporal_out"])
        self.input_proj.load_state_dict(state["input_proj"])
        self.net.load_state_dict(state["net"])
        self.out_proj.load_state_dict(state["out_proj"])
        self.frame_pos.data.copy_(state["frame_pos"].to(self.device))
        self.cls_token.data.copy_(state["cls_token"].to(self.device))
        if self.use_qcpa:
            self.head.load_state_dict(state["head"])
        else:
            legacy_state = state.get("legacy_head")
            if legacy_state is None:
                raise KeyError("legacy_head missing from checkpoint state")
            self.legacy_head.load_state_dict(legacy_state)
        self.num_frames = int(state.get("num_frames", self.num_frames))
        self.temporal_num_heads = int(state.get("temporal_num_heads", self.temporal_num_heads))
        self.temporal_num_layers = int(state.get("temporal_num_layers", self.temporal_num_layers))
        self.temporal_pooling = state.get("temporal_pooling", self.temporal_pooling)

    def train(self):
        self.frame_proj.train()
        self.temporal_encoder.train()
        self.temporal_norm.train()
        self.temporal_attn_pool.train()
        self.temporal_fusion_gate.train()
        self.temporal_out.train()
        self.input_proj.train()
        self.net.train()
        self.out_proj.train()
        if self.use_qcpa and self.head is not None:
            self.head.train()

    def eval(self):
        self.frame_proj.eval()
        self.temporal_encoder.eval()
        self.temporal_norm.eval()
        self.temporal_attn_pool.eval()
        self.temporal_fusion_gate.eval()
        self.temporal_out.eval()
        self.input_proj.eval()
        self.net.eval()
        self.out_proj.eval()
        if self.use_qcpa and self.head is not None:
            self.head.eval()

    def _pool_frames(self, image_x):
        import torch

        if image_x.ndim != 3:
            raise ValueError(
                f"TemporalTransformerClipImageAdapter expects [batch, frames, dim] inputs, got shape={tuple(image_x.shape)}"
            )
        frame_count = int(image_x.shape[1])
        if frame_count > self.num_frames:
            raise ValueError(
                f"TemporalTransformerClipImageAdapter received {frame_count} frames but was initialized for {self.num_frames}"
            )
        hidden = self.frame_proj(image_x)
        cls = self.cls_token.expand(hidden.shape[0], -1, -1)
        tokens = torch.cat([cls, hidden], dim=1)
        tokens = tokens + self.frame_pos[:, : frame_count + 1, :]
        encoded = self.temporal_encoder(tokens)
        cls_hidden = encoded[:, 0, :]
        frame_hidden = encoded[:, 1: frame_count + 1, :]
        if self.temporal_pooling == "cls":
            pooled_hidden = cls_hidden
        elif self.temporal_pooling == "mean":
            pooled_hidden = frame_hidden.mean(dim=1)
        elif self.temporal_pooling == "cls_mean_avg":
            pooled_hidden = 0.5 * (cls_hidden + frame_hidden.mean(dim=1))
        elif self.temporal_pooling == "cls_mean_gate":
            mean_hidden = frame_hidden.mean(dim=1)
            fusion_input = torch.cat([cls_hidden, mean_hidden], dim=-1)
            fusion_gate = torch.sigmoid(self.temporal_fusion_gate(fusion_input))
            pooled_hidden = fusion_gate * cls_hidden + (1.0 - fusion_gate) * mean_hidden
        else:
            attn_scores = self.temporal_attn_pool(frame_hidden).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            attn_hidden = torch.einsum("bf,bfh->bh", attn_weights, frame_hidden)
            if self.temporal_pooling == "attn":
                pooled_hidden = attn_hidden
            elif self.temporal_pooling == "cls_attn_fusion":
                fusion_input = torch.cat([cls_hidden, attn_hidden], dim=-1)
                fusion_gate = torch.sigmoid(self.temporal_fusion_gate(fusion_input))
                pooled_hidden = fusion_gate * cls_hidden + (1.0 - fusion_gate) * attn_hidden
            else:
                raise ValueError(f"Unsupported temporal_pooling: {self.temporal_pooling}")
        pooled_hidden = self.temporal_norm(pooled_hidden)
        pooled = self.temporal_out(pooled_hidden)
        return pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def _adapt_image(self, image_x):
        pooled = self._pool_frames(image_x)
        base = self.input_proj(pooled)
        delta = self.net(base)
        fused = base + delta
        img = self.out_proj(fused)
        return img / img.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def compute_logits_from_adapted(self, adapted_features, text_x, return_aux: bool = False):
        if self.use_qcpa:
            txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return self.head(adapted_features, txt, return_aux=return_aux)
        return self.legacy_head.logits(adapted_features, text_x)

    def logits(self, image_x, text_x, return_aux: bool = False):
        img = self._adapt_image(image_x)
        return self.compute_logits_from_adapted(img, text_x, return_aux=return_aux)

    def grouped_logits(self, image_x, text_x, group_indices: List[List[int]]):
        img = self._adapt_image(image_x)
        if self.use_qcpa:
            import torch

            txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            sim = torch.einsum("bd,cpd->bcp", img, txt)

            group_scores = []
            for gidx in group_indices:
                group_scores.append(sim[:, :, gidx].mean(dim=-1))
            scores = torch.stack(group_scores, dim=-1)
            global_scale = self.head.log_scale.exp().clamp(max=100.0).view(1, 1, 1)
            class_scale = self.head.log_tau.exp().view(1, -1, 1)
            class_bias = self.head.bias_static.view(1, -1, 1) if getattr(self.head, "bias_static", None) is not None else 0.0
            return global_scale * scores * class_scale + class_bias
        return self.legacy_head.grouped_logits(img, text_x, group_indices)


class FocalCrossEntropyLoss:
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as F

        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        pt = probs.gather(dim=-1, index=targets.view(-1, 1)).squeeze(-1).clamp(min=1e-12, max=1.0)
        focal = ((1 - pt).clamp(min=0.0) ** self.gamma) * ce
        return focal.mean()


def zeroshot_logits(image_x, text_x):
    import torch

    if image_x.ndim == 3:
        image_x = image_x.mean(dim=1)
    img = image_x / image_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    sim = torch.einsum("bd,cpd->bcp", img, txt)
    return sim.mean(dim=-1)


def zeroshot_grouped_logits(image_x, text_x, group_indices: List[List[int]]):
    import torch

    if image_x.ndim == 3:
        image_x = image_x.mean(dim=1)
    img = image_x / image_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    txt = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    sim = torch.einsum("bd,cpd->bcp", img, txt)

    group_scores = []
    for gidx in group_indices:
        group_scores.append(sim[:, :, gidx].mean(dim=-1))
    return torch.stack(group_scores, dim=-1)


def predict_zeroshot_from_features(
    image_features,
    text_features,
    idx2label: Dict[int, str],
    batch_size: int,
    use_test_ensemble: bool,
    ensemble_group_size: int,
) -> List[str]:
    import torch

    preds = []
    target_device = text_features.device
    text_features = text_features.to(target_device)
    group_indices = build_prompt_group_indices(int(text_features.shape[1]), ensemble_group_size)

    for start in range(0, image_features.shape[0], batch_size):
        batch_x = image_features[start:start + batch_size].to(target_device)
        with torch.no_grad():
            if use_test_ensemble and len(group_indices) > 1:
                g_logits = zeroshot_grouped_logits(batch_x, text_features, group_indices)
                group_pred = g_logits.argmax(dim=1)
                total_scores = g_logits.sum(dim=-1)
                idxs = []
                for i in range(group_pred.shape[0]):
                    votes = torch.bincount(group_pred[i], minlength=len(idx2label))
                    top = votes.max()
                    cands = (votes == top).nonzero(as_tuple=False).view(-1)
                    if cands.numel() == 1:
                        idxs.append(int(cands[0].item()))
                    else:
                        cs = total_scores[i, cands]
                        idxs.append(int(cands[cs.argmax()].item()))
            else:
                logits = zeroshot_logits(batch_x, text_features)
                idxs = logits.argmax(dim=-1).detach().cpu().tolist()
        preds.extend([idx2label[i] for i in idxs])
    return preds


def predict_zeroshot_from_samples(
    samples: List[Dict],
    processor,
    model,
    text_features,
    idx2label: Dict[int, str],
    batch_size: int,
    use_test_ensemble: bool,
    ensemble_group_size: int,
    args,
    split_name: str,
):
    import torch

    preds: List[str] = []
    kept_samples: List[Dict] = []
    failed_samples: List[Dict] = []
    sampling_debug: List[Dict[str, object]] = []
    group_indices = build_prompt_group_indices(int(text_features.shape[1]), ensemble_group_size)
    target_device = next(model.parameters()).device
    text_features_device = text_features.to(target_device)

    model.eval()
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        extracted = extract_batch_image_features_from_samples(
            batch_samples=batch,
            processor=processor,
            model=model,
            device=str(target_device),
            num_frames=args.num_frames,
            frame_sampling_mode=args.frame_sampling_mode,
            feature_layout=args.feature_layout,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            sampling_window_start=args.sampling_window_start,
            sampling_window_end=args.sampling_window_end,
            diff_alpha=args.diff_alpha,
            diff_beta=args.diff_beta,
            min_gap_ratio=args.min_gap_ratio,
            score_smooth_window=args.score_smooth_window,
            frame_diff_metric=args.frame_diff_metric,
            split_name=split_name,
            enable_grad=False,
        )
        failed_samples.extend(extracted["failed_samples"])
        remaining_debug = DEFAULT_SAMPLING_DEBUG_SAMPLES - len(sampling_debug)
        if remaining_debug > 0:
            sampling_debug.extend(extracted["sampling_debug"][:remaining_debug])
        if int(extracted["features"].shape[0]) == 0:
            continue
        kept_samples.extend(extracted["samples"])
        with torch.no_grad():
            if use_test_ensemble and len(group_indices) > 1:
                g_logits = zeroshot_grouped_logits(extracted["features"], text_features_device, group_indices)
                group_pred = g_logits.argmax(dim=1)
                total_scores = g_logits.sum(dim=-1)
                idxs = []
                for i in range(group_pred.shape[0]):
                    votes = torch.bincount(group_pred[i], minlength=len(idx2label))
                    top = votes.max()
                    cands = (votes == top).nonzero(as_tuple=False).view(-1)
                    if cands.numel() == 1:
                        idxs.append(int(cands[0].item()))
                    else:
                        cs = total_scores[i, cands]
                        idxs.append(int(cands[cs.argmax()].item()))
            else:
                logits = zeroshot_logits(extracted["features"], text_features_device)
                idxs = logits.argmax(dim=-1).detach().cpu().tolist()
        preds.extend([idx2label[i] for i in idxs])

    return {
        "preds": preds,
        "samples": kept_samples,
        "failed_samples": failed_samples,
        "sampling_debug": sampling_debug,
    }


def predict_emotion_from_features(
    image_features,
    text_features,
    adapter,
    idx2label: Dict[int, str],
    batch_size: int,
    use_test_ensemble: bool,
    ensemble_group_size: int,
    dump_attn_path: Optional[str] = None,
) -> List[str]:
    import torch

    preds = []
    group_indices = build_prompt_group_indices(int(text_features.shape[1]), ensemble_group_size)
    attn_batches = []

    for start in range(0, image_features.shape[0], batch_size):
        batch_x = image_features[start:start + batch_size]
        with torch.no_grad():
            if dump_attn_path:
                _, aux = adapter.logits(batch_x.to(adapter.device), text_features.to(adapter.device), return_aux=True)
                if aux is not None and aux.get("attn") is not None:
                    attn_batches.append(aux["attn"].detach().cpu().numpy())
            if use_test_ensemble and len(group_indices) > 1:
                g_logits = adapter.grouped_logits(batch_x.to(adapter.device), text_features.to(adapter.device), group_indices)
                group_pred = g_logits.argmax(dim=1)
                total_scores = g_logits.sum(dim=-1)
                idxs = []
                for i in range(group_pred.shape[0]):
                    votes = torch.bincount(group_pred[i], minlength=len(idx2label))
                    top = votes.max()
                    cands = (votes == top).nonzero(as_tuple=False).view(-1)
                    if cands.numel() == 1:
                        idxs.append(int(cands[0].item()))
                    else:
                        cs = total_scores[i, cands]
                        idxs.append(int(cands[cs.argmax()].item()))
            else:
                logits = adapter.logits(batch_x.to(adapter.device), text_features.to(adapter.device))
                idxs = logits.argmax(dim=-1).detach().cpu().tolist()
        preds.extend([idx2label[i] for i in idxs])
    if dump_attn_path and attn_batches:
        dump_path = Path(dump_attn_path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(dump_path), np.concatenate(attn_batches, axis=0))
        log(f"[QCPA] attention dump saved to: {dump_path}")
    return preds


def extract_batch_image_features_from_samples(
    batch_samples: List[Dict],
    processor,
    model,
    device: str,
    num_frames: int,
    frame_sampling_mode: str,
    feature_layout: str,
    num_workers: int,
    pin_memory: bool,
    sampling_window_start: float,
    sampling_window_end: float,
    diff_alpha: float,
    diff_beta: float,
    min_gap_ratio: float,
    score_smooth_window: int,
    frame_diff_metric: str,
    split_name: str,
    enable_grad: bool,
):
    import torch

    feature_dim = int(getattr(model.config, "projection_dim", 512))
    if not batch_samples:
        empty_shape = (0, feature_dim) if feature_layout == "pooled" else (0, num_frames, feature_dim)
        return {
            "features": torch.empty(empty_shape, device=device),
            "samples": [],
            "failed_samples": [],
            "sampling_debug": [],
        }

    valid_samples: List[Dict] = []
    frame_groups: List[List] = []
    failed_samples: List[Dict] = []
    sampling_debug: List[Dict[str, object]] = []
    if num_workers and num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            decoded_items = list(
                executor.map(
                    lambda sample: decode_sample_frames(
                        sample,
                        num_frames,
                        frame_sampling_mode,
                        split_name,
                        sampling_window_start,
                        sampling_window_end,
                        diff_alpha,
                        diff_beta,
                        min_gap_ratio,
                        score_smooth_window,
                        frame_diff_metric,
                    ),
                    batch_samples,
                )
            )
    else:
        decoded_items = [
            decode_sample_frames(
                sample,
                num_frames,
                frame_sampling_mode,
                split_name,
                sampling_window_start,
                sampling_window_end,
                diff_alpha,
                diff_beta,
                min_gap_ratio,
                score_smooth_window,
                frame_diff_metric,
            )
            for sample in batch_samples
        ]

    for decoded in decoded_items:
        if decoded["error"] is not None:
            failed_samples.append(decoded["error"])
            continue
        valid_samples.append(decoded["sample"])
        frame_groups.append(decoded["frames"])
        if len(sampling_debug) < DEFAULT_SAMPLING_DEBUG_SAMPLES:
            sampling_debug.append(
                {
                    "split_name": split_name,
                    "sample_index": decoded["sample"].get("sample_index"),
                    "sequence_id": decoded["sample"].get("sequence_id"),
                    "video_path": decoded["sample"].get("video_path"),
                    "total_frames": decoded.get("total_frames", 0),
                    "selected_indices": list(decoded.get("selected_indices", [])),
                }
            )

    if not valid_samples:
        empty_shape = (0, feature_dim) if feature_layout == "pooled" else (0, num_frames, feature_dim)
        return {
            "features": torch.empty(empty_shape, device=device),
            "samples": [],
            "failed_samples": failed_samples,
            "sampling_debug": sampling_debug,
        }

    flat_images = [image for images in frame_groups for image in images]
    frame_count = len(frame_groups[0])
    inputs = prepare_image_inputs(processor, flat_images, device=device, pin_memory=pin_memory)
    autocast_enabled = str(device).startswith("cuda")
    if enable_grad:
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    else:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    image_features = image_features.view(len(valid_samples), frame_count, -1)
    if feature_layout == "sequence":
        features = image_features
    else:
        features = image_features.mean(dim=1)
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return {
        "features": features,
        "samples": valid_samples,
        "failed_samples": failed_samples,
        "sampling_debug": sampling_debug,
    }


def enable_last_visual_block_training(model):
    last_block = model.vision_model.encoder.layers[-1]
    for param in last_block.parameters():
        param.requires_grad = True
    return last_block


def predict_emotion_from_samples_with_adapter(
    samples: List[Dict],
    processor,
    model,
    text_features,
    adapter,
    idx2label: Dict[int, str],
    batch_size: int,
    use_test_ensemble: bool,
    ensemble_group_size: int,
    args,
    split_name: str,
):
    import torch

    preds: List[str] = []
    kept_samples: List[Dict] = []
    failed_samples: List[Dict] = []
    sampling_debug: List[Dict[str, object]] = []
    group_indices = build_prompt_group_indices(int(text_features.shape[1]), ensemble_group_size)
    text_features_device = text_features.to(adapter.device)

    model.eval()
    adapter.eval()
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        extracted = extract_batch_image_features_from_samples(
            batch_samples=batch,
            processor=processor,
            model=model,
            device=adapter.device,
            num_frames=args.num_frames,
            frame_sampling_mode=args.frame_sampling_mode,
            feature_layout=args.feature_layout,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            sampling_window_start=args.sampling_window_start,
            sampling_window_end=args.sampling_window_end,
            diff_alpha=args.diff_alpha,
            diff_beta=args.diff_beta,
            min_gap_ratio=args.min_gap_ratio,
            score_smooth_window=args.score_smooth_window,
            frame_diff_metric=args.frame_diff_metric,
            split_name=split_name,
            enable_grad=False,
        )
        failed_samples.extend(extracted["failed_samples"])
        remaining_debug = DEFAULT_SAMPLING_DEBUG_SAMPLES - len(sampling_debug)
        if remaining_debug > 0:
            sampling_debug.extend(extracted["sampling_debug"][:remaining_debug])
        if int(extracted["features"].shape[0]) == 0:
            continue
        kept_samples.extend(extracted["samples"])
        with torch.no_grad():
            if use_test_ensemble and len(group_indices) > 1:
                g_logits = adapter.grouped_logits(extracted["features"], text_features_device, group_indices)
                group_pred = g_logits.argmax(dim=1)
                total_scores = g_logits.sum(dim=-1)
                idxs = []
                for i in range(group_pred.shape[0]):
                    votes = torch.bincount(group_pred[i], minlength=len(idx2label))
                    top = votes.max()
                    cands = (votes == top).nonzero(as_tuple=False).view(-1)
                    if cands.numel() == 1:
                        idxs.append(int(cands[0].item()))
                    else:
                        cs = total_scores[i, cands]
                        idxs.append(int(cands[cs.argmax()].item()))
            else:
                logits = adapter.logits(extracted["features"], text_features_device)
                idxs = logits.argmax(dim=-1).detach().cpu().tolist()
        preds.extend([idx2label[i] for i in idxs])

    return {
        "preds": preds,
        "samples": kept_samples,
        "failed_samples": failed_samples,
        "sampling_debug": sampling_debug,
    }


def train_unfreeze_last_visual_block_clip(
    train_samples: List[Dict],
    val_samples: List[Dict],
    processor,
    model,
    text_features,
    adapter,
    epochs: int,
    batch_size: int,
    lr: float,
    visual_block_lr_scale: float,
    weight_decay: float,
    max_grad_norm: float,
    use_class_weight: bool,
    label_smoothing: float,
    loss_type: str,
    focal_gamma: float,
    select_metric: str,
    use_test_ensemble: bool,
    ensemble_group_size: int,
    use_amp: bool,
    lr_scheduler: str,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    args,
):
    import torch
    import torch.nn as nn

    class_weights = None
    if use_class_weight:
        train_targets = torch.tensor([EMOTION_LABELS.index(sample["label"]) for sample in train_samples], dtype=torch.long)
        class_counts = torch.bincount(train_targets, minlength=len(EMOTION_LABELS)).float()
        class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).to(adapter.device)
        class_weights = class_weights / class_weights.mean().clamp(min=1e-12)

    if loss_type == "focal":
        criterion = FocalCrossEntropyLoss(gamma=focal_gamma, weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    last_visual_block = enable_last_visual_block_training(model)
    model.eval()
    last_visual_block.train()
    visual_params = [param for param in last_visual_block.parameters() if param.requires_grad]
    adapter_params = list(adapter.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": lr},
            {"params": visual_params, "lr": lr * visual_block_lr_scale},
        ],
        weight_decay=weight_decay,
    )
    scheduler = None
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(2, early_stopping_patience // 2) if early_stopping_patience > 0 else 2,
            threshold=early_stopping_min_delta,
            min_lr=1e-6,
        )
    amp_enabled = bool(use_amp and str(adapter.device).startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    text_features_device = text_features.to(adapter.device)

    best_adapter_state = None
    best_visual_state = None
    best_val_metric = -1.0
    best_epoch_idx = -1
    no_improve_epochs = 0
    idx2label = {idx: label for idx, label in enumerate(EMOTION_LABELS)}
    overall_start = time.time()

    for epoch_idx in range(epochs):
        adapter.train()
        model.eval()
        last_visual_block.train()
        epoch_start = time.time()
        running_loss = 0.0
        num_batches = 0
        shuffled_samples = list(train_samples)
        random.shuffle(shuffled_samples)

        for start in range(0, len(shuffled_samples), batch_size):
            batch = shuffled_samples[start:start + batch_size]
            extracted = extract_batch_image_features_from_samples(
                batch_samples=batch,
                processor=processor,
                model=model,
                device=adapter.device,
                num_frames=args.num_frames,
                frame_sampling_mode=args.frame_sampling_mode,
                feature_layout=args.feature_layout,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                sampling_window_start=args.sampling_window_start,
                sampling_window_end=args.sampling_window_end,
                diff_alpha=args.diff_alpha,
                diff_beta=args.diff_beta,
                min_gap_ratio=args.min_gap_ratio,
                score_smooth_window=args.score_smooth_window,
                frame_diff_metric=args.frame_diff_metric,
                split_name="train_online",
                enable_grad=True,
            )
            if extracted["failed_samples"]:
                log(f"[WARN] skipped {len(extracted['failed_samples'])} failed train decodes in online last-block mode")
            if int(extracted["features"].shape[0]) == 0:
                continue

            targets = torch.tensor(
                [EMOTION_LABELS.index(sample["label"]) for sample in extracted["samples"]],
                dtype=torch.long,
                device=adapter.device,
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = adapter.logits(extracted["features"], text_features_device)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter_params + visual_params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            num_batches += 1

        val_outputs = predict_emotion_from_samples_with_adapter(
            samples=val_samples,
            processor=processor,
            model=model,
            text_features=text_features,
            adapter=adapter,
            idx2label=idx2label,
            batch_size=batch_size,
            use_test_ensemble=use_test_ensemble,
            ensemble_group_size=ensemble_group_size,
            args=args,
            split_name="val_online",
        )
        val_true = [sample["label"] for sample in val_outputs["samples"]]
        val_pred = val_outputs["preds"]
        val_acc = accuracy(val_true, val_pred)
        val_wf1 = weighted_f1(val_true, val_pred, EMOTION_LABELS)
        metric = val_wf1 if select_metric == "weighted_f1" else val_acc

        if scheduler is not None:
            scheduler.step(metric)

        if metric > best_val_metric + early_stopping_min_delta:
            best_val_metric = metric
            best_epoch_idx = epoch_idx
            no_improve_epochs = 0
            best_adapter_state = copy.deepcopy(adapter.state_dict())
            best_visual_state = copy.deepcopy(last_visual_block.state_dict())
        else:
            no_improve_epochs += 1

        elapsed = time.time() - overall_start
        eta = elapsed / (epoch_idx + 1) * (epochs - epoch_idx - 1)
        adapter_lr = optimizer.param_groups[0]["lr"]
        visual_lr = optimizer.param_groups[1]["lr"]
        log(
            f"[TRAIN] epoch {epoch_idx + 1}/{epochs} | loss={running_loss / max(1, num_batches):.6f} | "
            f"val_acc={val_acc:.6f} | val_wf1={val_wf1:.6f} | best_metric={best_val_metric:.6f} | "
            f"best_epoch={best_epoch_idx + 1 if best_epoch_idx >= 0 else 0} | no_improve={no_improve_epochs} | "
            f"adapter_lr={adapter_lr:.6e} | visual_lr={visual_lr:.6e} | "
            f"epoch_time={format_duration(time.time() - epoch_start)} | elapsed={format_duration(elapsed)} | eta={format_duration(eta)}"
        )

        if early_stopping_patience > 0 and no_improve_epochs >= early_stopping_patience:
            log(
                f"[TRAIN] early stopping triggered at epoch {epoch_idx + 1} "
                f"(best_epoch={best_epoch_idx + 1 if best_epoch_idx >= 0 else 0}, best_metric={best_val_metric:.6f})"
            )
            break

    if best_adapter_state is not None:
        adapter.load_state_dict(best_adapter_state)
    if best_visual_state is not None:
        last_visual_block.load_state_dict(best_visual_state)
    adapter.eval()
    model.eval()
    return adapter


def predict_emotion(samples: List[Dict], processor, model, prompts: List[str], device: str, batch_size: int) -> List[str]:
    import torch

    preds = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [read_middle_frame(s["video_path"]).convert("RGB") for s in batch]
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            idxs = logits.argmax(dim=-1).detach().cpu().tolist()
        preds.extend([EMOTION_LABELS[i] for i in idxs])
    return preds


def train_clip_supervised(
    model,
    processor,
    train_samples: List[Dict],
    val_samples: List[Dict],
    prompts: List[str],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
):
    import torch
    import torch.nn as nn

    label2idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    overall_start = time.time()

    for epoch_idx in range(epochs):
        model.train()
        random.shuffle(train_samples)
        epoch_start = time.time()
        running_loss = 0.0
        num_batches = 0

        for start in range(0, len(train_samples), batch_size):
            batch = train_samples[start:start + batch_size]
            images = [read_middle_frame(s["video_path"]).convert("RGB") for s in batch]
            targets = torch.tensor([label2idx[s["label"]] for s in batch], dtype=torch.long, device=device)
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)

            outputs = model(**inputs)
            logits = outputs.logits_per_image
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += float(loss.item())
            num_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = predict_emotion(val_samples, processor, model, prompts, device, batch_size)
            val_true = [s["label"] for s in val_samples]
            val_acc = accuracy(val_true, val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - overall_start
        eta = elapsed / (epoch_idx + 1) * (epochs - epoch_idx - 1)
        log(
            f"[TRAIN] epoch {epoch_idx + 1}/{epochs} | loss={running_loss / max(1, num_batches):.6f} | "
            f"val_acc={val_acc:.6f} | best_val_acc={best_val_acc:.6f} | "
            f"epoch_time={format_duration(time.time() - epoch_start)} | elapsed={format_duration(elapsed)} | eta={format_duration(eta)}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_strict_frozen_clip(
    train_x,
    train_y,
    val_x,
    val_y,
    text_features,
    adapter: ClipImageAdapter,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    use_class_weight: bool,
    label_smoothing: float,
    loss_type: str,
    focal_gamma: float,
    select_metric: str,
    use_test_ensemble: bool,
    ensemble_group_size: int,
    use_amp: bool,
    lr_scheduler: str,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
):
    import torch
    import torch.nn as nn

    class_weights = None
    if use_class_weight:
        class_counts = torch.bincount(train_y, minlength=len(EMOTION_LABELS)).float()
        class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).to(adapter.device)
        class_weights = class_weights / class_weights.mean().clamp(min=1e-12)

    if loss_type == "focal":
        criterion = FocalCrossEntropyLoss(gamma=focal_gamma, weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(2, early_stopping_patience // 2) if early_stopping_patience > 0 else 2,
            threshold=early_stopping_min_delta,
            min_lr=1e-6,
        )
    amp_enabled = bool(use_amp and str(adapter.device).startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    text_features_device = text_features.to(adapter.device)

    best_state = None
    best_val_metric = -1.0
    best_val_loss = float("inf")
    best_epoch_idx = -1
    no_improve_epochs = 0
    idx2label = {idx: label for idx, label in enumerate(EMOTION_LABELS)}
    overall_start = time.time()
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    val_wf1_history = []

    for epoch_idx in range(epochs):
        adapter.train()
        epoch_start = time.time()
        running_loss = 0.0
        num_batches = 0
        perm = torch.randperm(train_x.shape[0])
        train_x = train_x[perm]
        train_y = train_y[perm]

        for start in range(0, train_x.shape[0], batch_size):
            bx = train_x[start:start + batch_size].to(adapter.device)
            by = train_y[start:start + batch_size].to(adapter.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = adapter.logits(bx, text_features_device)
                loss = criterion(logits, by)

            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            num_batches += 1

        adapter.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            val_num_batches = 0
            for start in range(0, val_x.shape[0], batch_size):
                bx = val_x[start:start + batch_size].to(adapter.device)
                by = val_y[start:start + batch_size].to(adapter.device)
                logits = adapter.logits(bx, text_features_device)
                batch_loss = criterion(logits, by)
                val_running_loss += float(batch_loss.item())
                val_num_batches += 1
            val_loss_mean = val_running_loss / max(1, val_num_batches)
            val_pred = predict_emotion_from_features(
                val_x,
                text_features,
                adapter,
                idx2label,
                batch_size,
                use_test_ensemble=use_test_ensemble,
                ensemble_group_size=ensemble_group_size,
            )
            val_true = [EMOTION_LABELS[int(x.item())] for x in val_y]
            val_acc = accuracy(val_true, val_pred)
            val_wf1 = weighted_f1(val_true, val_pred, EMOTION_LABELS)
            metric = val_wf1 if select_metric == "weighted_f1" else val_acc

        train_loss_mean = running_loss / max(1, num_batches)
        train_loss_history.append(round(float(train_loss_mean), 6))
        val_loss_history.append(round(float(val_loss_mean), 6))
        val_acc_history.append(round(float(val_acc), 6))
        val_wf1_history.append(round(float(val_wf1), 6))
        best_val_loss = min(best_val_loss, float(val_loss_mean))

        if scheduler is not None:
            scheduler.step(metric)

        if metric > best_val_metric + early_stopping_min_delta:
            best_val_metric = metric
            best_epoch_idx = epoch_idx
            no_improve_epochs = 0
            best_state = copy.deepcopy(adapter.state_dict())
        else:
            no_improve_epochs += 1

        elapsed = time.time() - overall_start
        eta = elapsed / (epoch_idx + 1) * (epochs - epoch_idx - 1)
        current_lr = optimizer.param_groups[0]["lr"]
        qcpa_log = ""
        if getattr(adapter, "use_qcpa", False) and getattr(adapter, "head", None) is not None:
            head = adapter.head
            qcpa_log = (
                f" | gate={format_tensor_list(getattr(head, 'gate', None), precision=4)}"
                f" | tau={format_tensor_list(getattr(head, 'log_tau', None).exp(), precision=4) if getattr(head, 'log_tau', None) is not None else []}"
                f" | scale={round(float(getattr(head, 'log_scale').exp().detach().cpu().item()), 4)}"
            )
        log(
            f"[TRAIN] epoch {epoch_idx + 1}/{epochs} | loss={train_loss_mean:.6f} | val_loss={val_loss_mean:.6f} | "
            f"val_acc={val_acc:.6f} | val_wf1={val_wf1:.6f} | best_metric={best_val_metric:.6f} | "
            f"best_epoch={best_epoch_idx + 1 if best_epoch_idx >= 0 else 0} | no_improve={no_improve_epochs} | lr={current_lr:.6e} | "
            f"epoch_time={format_duration(time.time() - epoch_start)} | elapsed={format_duration(elapsed)} | eta={format_duration(eta)}{qcpa_log}"
        )

        if early_stopping_patience > 0 and no_improve_epochs >= early_stopping_patience:
            log(
                f"[TRAIN] early stopping triggered at epoch {epoch_idx + 1}; "
                f"best_epoch={best_epoch_idx + 1 if best_epoch_idx >= 0 else 0}, best_metric={best_val_metric:.6f}"
            )
            break

    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.training_diagnostics = {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "val_accuracy_history": val_acc_history,
        "val_weighted_f1_history": val_wf1_history,
        "best_val_loss": round(float(best_val_loss), 6) if best_val_loss < float("inf") else None,
        "best_epoch": best_epoch_idx + 1 if best_epoch_idx >= 0 else None,
        "best_val_weighted_f1": round(max(val_wf1_history), 6) if val_wf1_history else None,
    }
    adapter.eval()
    return adapter


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone RAVDESS CLIP emotion training with strict frozen CLIP adapter")
    parser.add_argument(
        "--experiment_name",
        choices=["custom", "exp_baseline_locked", "exp_baseline_locked_capc", "exp_baseline_locked_qcpa", "exp_middle_late", "exp_diff_guided"],
        default="custom",
    )
    parser.add_argument("--ravdess_root", default=DEFAULT_RAVDESS_ROOT)
    parser.add_argument("--train_ratio", type=float, default=0.65)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_mode", choices=["offline_only", "auto"], default="auto")
    parser.add_argument("--model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt_template", default="The person looks <LABEL>.")
    parser.add_argument(
        "--prompt_set",
        default="ravdess_8",
        help="single | default_5 | ravdess_8 | ravdess_8_facial_cues | ravdess_8_pairwise_cues | ravdess_8_stage_cues | ravdess_8_auto_selected | ravdess_8_auto_selected_hybrid | custom templates joined by ||",
    )
    parser.add_argument("--auto_prompt_k", type=int, default=4, help="Selected prompts per class for ravdess_8_auto_selected")
    parser.add_argument("--auto_prompt_refine_passes", type=int, default=2, help="Coordinate refinement passes for ravdess_8_auto_selected")
    parser.add_argument("--auto_prompt_top_pairs", type=int, default=6, help="Number of top confusing text prototype pairs to report")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--extract_batch_size", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=None, help="Deprecated alias for --train_batch_size")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_sampling_mode", choices=["uniform", "middle_late", "diff_guided"], default="uniform")
    parser.add_argument("--sampling_window_start", type=float, default=DEFAULT_SAMPLING_WINDOW_START)
    parser.add_argument("--sampling_window_end", type=float, default=DEFAULT_SAMPLING_WINDOW_END)
    parser.add_argument("--diff_alpha", type=float, default=DEFAULT_DIFF_ALPHA)
    parser.add_argument("--diff_beta", type=float, default=DEFAULT_DIFF_BETA)
    parser.add_argument("--min_gap_ratio", type=float, default=DEFAULT_MIN_GAP_RATIO)
    parser.add_argument("--score_smooth_window", type=int, default=DEFAULT_SCORE_SMOOTH_WINDOW)
    parser.add_argument("--frame_diff_metric", choices=["gray_l1"], default=DEFAULT_FRAME_DIFF_METRIC)
    parser.add_argument("--feature_layout", choices=["pooled", "sequence"], default="pooled")
    parser.add_argument("--adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--temporal_head", choices=["none", "transformer"], default="none")
    parser.add_argument("--temporal_num_heads", type=int, default=4)
    parser.add_argument("--temporal_num_layers", type=int, default=2)
    parser.add_argument("--temporal_pooling", choices=["cls", "mean", "cls_mean_avg", "cls_mean_gate", "attn", "cls_attn_fusion"], default="cls")
    parser.add_argument("--num_workers", type=int, default=4, help="CPU workers for per-batch video decoding inside one shard process")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--disable_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--use_class_weight", dest="use_class_weight", action="store_true")
    parser.add_argument("--disable_class_weight", dest="use_class_weight", action="store_false")
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--loss_type", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--select_metric", choices=["accuracy", "weighted_f1"], default="weighted_f1")
    parser.add_argument("--use_test_ensemble", dest="use_test_ensemble", action="store_true")
    parser.add_argument("--disable_test_ensemble", dest="use_test_ensemble", action="store_false")
    parser.add_argument("--ensemble_group_size", type=int, default=2)
    parser.add_argument("--strict_frozen_clip", dest="strict_frozen_clip", action="store_true", help="Freeze CLIP and train only custom adapter module")
    parser.add_argument("--disable_strict_frozen_clip", dest="strict_frozen_clip", action="store_false")
    parser.add_argument("--unfreeze_last_visual_block", action="store_true", help="Train adapter plus the last CLIP visual encoder block with a smaller LR")
    parser.add_argument("--visual_block_lr_scale", type=float, default=0.1)
    parser.add_argument("--use_prompt_weight", dest="use_prompt_weight", action="store_true")
    parser.add_argument("--disable_prompt_weight", dest="use_prompt_weight", action="store_false")
    parser.add_argument("--use_global_logit_scale", dest="use_global_logit_scale", action="store_true")
    parser.add_argument("--disable_global_logit_scale", dest="use_global_logit_scale", action="store_false")
    parser.add_argument("--use_class_temperature", dest="use_class_temperature", action="store_true")
    parser.add_argument("--disable_class_temperature", dest="use_class_temperature", action="store_false")
    parser.add_argument("--use_class_bias", dest="use_class_bias", action="store_true")
    parser.add_argument("--disable_class_bias", dest="use_class_bias", action="store_false")
    parser.add_argument("--use_qcpa", dest="use_qcpa", action="store_true")
    parser.add_argument("--disable_qcpa", dest="use_qcpa", action="store_false")
    parser.add_argument("--qcpa_num_heads", type=int, default=4)
    parser.add_argument("--attention_scope", choices=["local", "global"], default="local")
    parser.add_argument("--dual_stage", action="store_true")
    parser.add_argument("--no_residual_gate", action="store_true")
    parser.add_argument("--no_mahalanobis", action="store_true")
    parser.add_argument("--no_lowrank_bias", action="store_true")
    parser.add_argument("--bias_rank", type=int, default=16)
    parser.add_argument("--dump_attn", action="store_true")
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--disable_amp", dest="use_amp", action="store_false")
    parser.add_argument("--strict_seed_control", action="store_true")
    parser.add_argument("--lr_scheduler", choices=["plateau", "none"], default="plateau")
    parser.add_argument("--disable_scheduler", action="store_true", default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--feature_cache_dir", default=DEFAULT_FEATURE_CACHE_DIR)
    parser.add_argument("--legacy_feature_cache_path", default=None)
    parser.add_argument("--split_name", choices=["train", "val", "test"], default=None)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--extract_only", action="store_true", help="Only extract image features for one split and exit")
    mode_group.add_argument("--merge_shards", action="store_true", help="Merge cached feature shards for one split and exit")
    parser.add_argument("--delete_shards_after_merge", action="store_true")
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--split_mode", choices=["benchmark_5fold", "actor_random"], default="benchmark_5fold")
    parser.add_argument("--benchmark_video_list", default=DEFAULT_BENCHMARK_VIDEO_LIST)
    parser.add_argument("--benchmark_use_video_list", action="store_true")
    parser.add_argument("--benchmark_test_fold", type=int, default=0)
    parser.add_argument("--benchmark_val_fold", type=int, default=None)
    parser.add_argument("--allowed_modalities", default="02", help="Comma-separated modality codes. Use all for no filtering.")
    parser.add_argument("--allowed_vocal_channels", default="01", help="Comma-separated vocal channel codes. Use all for no filtering.")
    parser.add_argument("--allowed_intensities", default="01,02", help="Comma-separated intensity codes. Use all for no filtering.")
    parser.add_argument("--actor_ids", default="all", help="Comma-separated actor ids in [1, 24]. Use all for no filtering.")
    parser.add_argument("--video_extensions", default=".mp4")
    parser.add_argument("--checkpoint_output", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--run_zero_shot_eval", action="store_true")
    parser.add_argument("--zero_shot_only", action="store_true")
    parser.add_argument("--report_train_metrics", action="store_true")
    parser.set_defaults(
        use_class_weight=True,
        use_test_ensemble=True,
        strict_frozen_clip=True,
        use_global_logit_scale=False,
        use_prompt_weight=False,
        use_class_temperature=False,
        use_class_bias=False,
        use_qcpa=True,
        pin_memory=True,
        use_amp=True,
    )
    args = parser.parse_args()
    apply_experiment_preset(args)
    if args.disable_scheduler:
        args.lr_scheduler = "none"
    apply_experiment_output_names(args)

    if (args.extract_only or args.merge_shards) and not args.split_name:
        parser.error("--split_name must be provided when using --extract_only or --merge_shards")
    if args.gpu_id is not None:
        args.device = f"cuda:{args.gpu_id}"
    if args.total_shards < 1:
        parser.error("--total_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.total_shards:
        parser.error(
            f"--shard_index must satisfy 0 <= shard_index < total_shards; got shard_index={args.shard_index}, total_shards={args.total_shards}"
        )
    if args.bias_rank < 1:
        parser.error("--bias_rank must be >= 1")
    if args.early_stopping_patience < 0:
        parser.error("--early_stopping_patience must be >= 0")
    if args.early_stopping_min_delta < 0:
        parser.error("--early_stopping_min_delta must be >= 0")
    if not 0.0 <= args.sampling_window_start <= 1.0:
        parser.error("--sampling_window_start must be in [0, 1]")
    if not 0.0 <= args.sampling_window_end <= 1.0:
        parser.error("--sampling_window_end must be in [0, 1]")
    if args.score_smooth_window < 1:
        parser.error("--score_smooth_window must be >= 1")
    if args.min_gap_ratio < 0.0:
        parser.error("--min_gap_ratio must be >= 0")
    if args.batch_size is not None:
        args.train_batch_size = args.batch_size
    if args.temporal_head != "none":
        args.feature_layout = "sequence"
    if args.unfreeze_last_visual_block and args.feature_layout != "sequence":
        parser.error("--unfreeze_last_visual_block currently requires feature_layout=sequence")
    if args.visual_block_lr_scale <= 0:
        parser.error("--visual_block_lr_scale must be > 0")
    if args.temporal_num_heads < 1:
        parser.error("--temporal_num_heads must be >= 1")
    if args.temporal_num_layers < 1:
        parser.error("--temporal_num_layers must be >= 1")
    if args.adapter_hidden_dim % args.temporal_num_heads != 0:
        parser.error("--adapter_hidden_dim must be divisible by --temporal_num_heads")
    if args.qcpa_num_heads < 1:
        parser.error("--qcpa_num_heads must be >= 1")
    args.batch_size = args.train_batch_size
    return args


def configure_random_seeds(seed: int, strict_seed_control: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if strict_seed_control:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def main():
    args = parse_args()

    if args.gpu_id is not None and str(args.device).startswith("cuda"):
        args.device = f"cuda:{args.gpu_id}"
    if not args.strict_frozen_clip:
        raise ValueError("This RAVDESS script is configured to preserve the strict_frozen_clip adapter pipeline. Do not disable it.")
    if args.temporal_head != "none" and args.feature_layout != "sequence":
        raise ValueError("temporal_head requires feature_layout=sequence")
    if args.unfreeze_last_visual_block and args.zero_shot_only:
        raise ValueError("--unfreeze_last_visual_block does not support --zero_shot_only")

    if not args.log_file:
        args.log_file = str(Path(args.output).with_suffix(".log"))
    resolved_log_file = init_log_file(args.log_file)
    atexit.register(close_log_file)
    if resolved_log_file:
        log(f"[LOG] writing log file to: {resolved_log_file}")

    allowed_modalities = parse_code_set(args.allowed_modalities, set(RAVDESS_MODALITY_CODE_MAP.keys()), "allowed_modalities")
    allowed_vocal_channels = parse_code_set(
        args.allowed_vocal_channels, set(RAVDESS_VOCAL_CHANNEL_CODE_MAP.keys()), "allowed_vocal_channels"
    )
    allowed_intensities = parse_code_set(args.allowed_intensities, set(RAVDESS_INTENSITY_CODE_MAP.keys()), "allowed_intensities")
    allowed_actor_ids = parse_actor_ids(args.actor_ids)
    allowed_extensions = parse_extension_set(args.video_extensions)
    effective_benchmark_val_fold = resolve_benchmark_val_fold(args.benchmark_test_fold, args.benchmark_val_fold)

    if args.split_mode == "benchmark_5fold":
        if args.benchmark_use_video_list:
            samples = collect_benchmark_aligned_ravdess_samples(
                ravdess_root=args.ravdess_root,
                benchmark_video_list=args.benchmark_video_list,
                max_sequences=args.max_sequences,
                allowed_extensions=allowed_extensions,
                allowed_actor_ids=allowed_actor_ids,
            )
            allowed_modalities = {"01"}
            allowed_vocal_channels = {"01"}
            allowed_intensities = set(RAVDESS_INTENSITY_CODE_MAP.keys())
        else:
            samples = collect_ravdess_samples(
                args.ravdess_root,
                args.max_sequences,
                allowed_modalities=allowed_modalities,
                allowed_vocal_channels=allowed_vocal_channels,
                allowed_intensities=allowed_intensities,
                allowed_extensions=allowed_extensions,
                allowed_actor_ids=allowed_actor_ids,
            )
    else:
        samples = collect_ravdess_samples(
            args.ravdess_root,
            args.max_sequences,
            allowed_modalities=allowed_modalities,
            allowed_vocal_channels=allowed_vocal_channels,
            allowed_intensities=allowed_intensities,
            allowed_extensions=allowed_extensions,
            allowed_actor_ids=allowed_actor_ids,
        )
    if len(samples) < 10:
        raise RuntimeError(f"Too few valid samples: {len(samples)}")
    log(f"[INFO] valid samples: {len(samples)}")

    if args.split_mode == "benchmark_5fold":
        splits = split_samples_benchmark_folds(samples, args.benchmark_test_fold, effective_benchmark_val_fold)
    else:
        splits = split_samples_by_actor(samples, args.train_ratio, args.val_ratio, args.seed)
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]
    log(f"[INFO] split sizes -> train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    import torch
    from transformers import CLIPModel, CLIPProcessor

    configure_random_seeds(args.seed, args.strict_seed_control)

    if args.clip_mode == "auto":
        try:
            processor = CLIPProcessor.from_pretrained(args.model_id)
            model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False)
        except Exception:
            processor = CLIPProcessor.from_pretrained(args.model_id, local_files_only=True)
            model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False, local_files_only=True)
    else:
        processor = CLIPProcessor.from_pretrained(args.model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(args.model_id, use_safetensors=False, local_files_only=True)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = model.to(device=args.device, dtype=dtype)
    log(f"[INFO] model loaded: {args.model_id} on {args.device}")

    label2idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
    idx2label = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    prompt_selection_info = None
    selected_text_features = None
    if args.prompt_set in {"ravdess_8_auto_selected", "ravdess_8_auto_selected_hybrid"}:
        candidate_prompt_groups = (
            build_ravdess_hybrid_candidate_prompt_groups()
            if args.prompt_set == "ravdess_8_auto_selected_hybrid"
            else build_ravdess_auto_candidate_prompt_groups()
        )
        prompt_groups, selected_text_features, prompt_selection_info = select_ravdess_auto_prompt_groups(
            processor=processor,
            model=model,
            device=args.device,
            prompts_per_class=args.auto_prompt_k,
            refine_passes=args.auto_prompt_refine_passes,
            top_pairs=args.auto_prompt_top_pairs,
            candidate_prompt_groups=candidate_prompt_groups,
            selection_name=args.prompt_set,
        )
        log(
            f"[PROMPT-SELECT] mean_off_diag={prompt_selection_info['mean_off_diagonal_similarity']:.6f} "
            f"max_off_diag={prompt_selection_info['max_off_diagonal_similarity']:.6f} "
            f"mean_within_class={prompt_selection_info['mean_within_class_similarity']:.6f}"
        )
        for label in EMOTION_LABELS:
            log(f"[PROMPT-SELECT] selected prompts for {label}:")
            for prompt in prompt_selection_info["selected_prompt_groups"][label]:
                log(f"[PROMPT-SELECT]   - {prompt}")
        if prompt_selection_info["most_confusing_pairs"]:
            log("[PROMPT-SELECT] most confusing text prototype pairs")
            for item in prompt_selection_info["most_confusing_pairs"]:
                pair = " vs ".join(item["pair"])
                log(f"[PROMPT-SELECT]   {pair}: {item['similarity']:.6f}")
    else:
        prompt_groups = build_class_prompts(args.prompt_template, args.prompt_set)

    prompt_groups_hash = hashlib.sha1(
        json.dumps(prompt_groups, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]

    train_samples = build_split_samples_with_index(train_samples, split_name="train")
    val_samples = build_split_samples_with_index(val_samples, split_name="val")
    test_samples = build_split_samples_with_index(test_samples, split_name="test")
    split_samples_map = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }
    log(f"[MODE] feature_layout={args.feature_layout} temporal_head={args.temporal_head}")
    resolved_split_cache_paths: Dict[str, str] = {}
    resolved_legacy_cache_path: Optional[str] = None
    split_cache_plans = {
        split_name: build_split_cache_plan(args.feature_cache_dir, "RAVDESS", split_name, split_samples_map[split_name], args)
        for split_name in ["train", "val", "test"]
    }
    for split_name, cache_plan in split_cache_plans.items():
        log(
            f"[CACHE] plan for {split_name}: final={cache_plan['final_path']} existing_shards={count_existing_shards(cache_plan)}/{args.total_shards}"
        )

    if args.merge_shards:
        merged_path = merge_feature_shards(
            split_cache_plans[args.split_name],
            delete_shards_after_merge=args.delete_shards_after_merge,
        )
        log(f"[DONE] merge-only mode finished: {merged_path}")
        return

    if args.extract_only:
        target_samples = split_samples_map[args.split_name]
        if args.total_shards == 1:
            extract_split_feature_cache(
                target_samples,
                processor=processor,
                model=model,
                args=args,
                split_name=args.split_name,
                cache_plan=split_cache_plans[args.split_name],
            )
        else:
            extract_feature_shard(
                target_samples,
                processor=processor,
                model=model,
                args=args,
                split_name=args.split_name,
                cache_plan=split_cache_plans[args.split_name],
            )
        log("[DONE] extraction-only mode finished")
        return

    if args.unfreeze_last_visual_block:
        train_cache_payload = {"failed_count": 0, "sampling_debug": []}
        val_cache_payload = {"failed_count": 0, "sampling_debug": []}
        test_cache_payload = {"failed_count": 0, "sampling_debug": []}
        if selected_text_features is not None:
            text_features = selected_text_features.cpu()
        else:
            text_features = extract_text_features(prompt_groups, processor, model, args.device)
        train_x = train_y = val_x = val_y = test_x = test_y = None
    elif args.legacy_feature_cache_path:
        legacy_cache_path = Path(args.legacy_feature_cache_path)
        if not legacy_cache_path.exists():
            raise RuntimeError(f"Legacy strict cache not found: {legacy_cache_path}")
        legacy_payload = load_legacy_strict_feature_cache(legacy_cache_path)
        validate_legacy_cache_split_alignment(
            legacy_payload,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            label2idx=label2idx,
        )
        resolved_legacy_cache_path = str(legacy_cache_path)
        resolved_split_cache_paths = {
            "train": resolved_legacy_cache_path,
            "val": resolved_legacy_cache_path,
            "test": resolved_legacy_cache_path,
        }
        train_x = legacy_payload["train_x"]
        train_y = legacy_payload["train_y"].long()
        val_x = legacy_payload["val_x"]
        val_y = legacy_payload["val_y"].long()
        test_x = legacy_payload["test_x"]
        test_y = legacy_payload["test_y"].long()
        train_cache_payload = {"failed_count": 0, "sampling_debug": []}
        val_cache_payload = {"failed_count": 0, "sampling_debug": []}
        test_cache_payload = {"failed_count": 0, "sampling_debug": []}
        if selected_text_features is not None:
            text_features = selected_text_features.cpu()
        else:
            text_features = legacy_payload["text_features"].float().cpu()
        log(f"[CACHE] using legacy strict cache for training/eval: {legacy_cache_path}")
    else:
        for split_name in ["train", "val", "test"]:
            resolved_path = ensure_training_split_cache(
                split_name=split_name,
                split_samples=split_samples_map[split_name],
                cache_plan=split_cache_plans[split_name],
                processor=processor,
                model=model,
                args=args,
            )
            resolved_split_cache_paths[split_name] = str(resolved_path)

        train_x, train_y, train_samples, train_cache_payload = load_features_and_labels_from_split_cache(
            split_cache_plans["train"]["final_path"],
            label2idx,
        )
        val_x, val_y, val_samples, val_cache_payload = load_features_and_labels_from_split_cache(
            split_cache_plans["val"]["final_path"],
            label2idx,
        )
        test_x, test_y, test_samples, test_cache_payload = load_features_and_labels_from_split_cache(
            split_cache_plans["test"]["final_path"],
            label2idx,
        )
        if selected_text_features is not None:
            text_features = selected_text_features.cpu()
        else:
            text_features = extract_text_features(prompt_groups, processor, model, args.device)

    text_prototype_similarity = compute_text_prototype_similarity(text_features.cpu())
    log_matrix("[TEXT] class prototype similarity matrix", text_prototype_similarity)
    text_similarity_stats = compute_similarity_stats(
        torch.tensor([[text_prototype_similarity[row][col] for col in EMOTION_LABELS] for row in EMOTION_LABELS], dtype=torch.float32),
        EMOTION_LABELS,
        top_pairs=args.auto_prompt_top_pairs,
    )
    log(
        f"[TEXT] mean_off_diag={text_similarity_stats['mean_off_diagonal_similarity']:.6f} "
        f"max_off_diag={text_similarity_stats['max_off_diagonal_similarity']:.6f}"
    )
    if text_similarity_stats["most_confusing_pairs"]:
        log("[TEXT] most confusing class pairs by prototype similarity")
        for item in text_similarity_stats["most_confusing_pairs"]:
            pair = " vs ".join(item["pair"])
            log(f"[TEXT]   {pair}: {item['similarity']:.6f}")

    if args.unfreeze_last_visual_block:
        val_true = [sample["label"] for sample in val_samples]
        test_true = [sample["label"] for sample in test_samples]
        train_true = [sample["label"] for sample in train_samples]
    else:
        val_true = [EMOTION_LABELS[int(i.item())] for i in val_y]
        test_true = [EMOTION_LABELS[int(i.item())] for i in test_y]
        train_true = [EMOTION_LABELS[int(i.item())] for i in train_y]

    zero_shot_val = None
    zero_shot_test = None
    zero_shot_train = None
    if args.run_zero_shot_eval or args.zero_shot_only:
        if args.unfreeze_last_visual_block:
            zero_shot_val_outputs = predict_zeroshot_from_samples(
                samples=val_samples,
                processor=processor,
                model=model,
                text_features=text_features,
                idx2label=idx2label,
                batch_size=args.batch_size,
                use_test_ensemble=args.use_test_ensemble,
                ensemble_group_size=args.ensemble_group_size,
                args=args,
                split_name="val_zero_shot_online",
            )
            zero_shot_test_outputs = predict_zeroshot_from_samples(
                samples=test_samples,
                processor=processor,
                model=model,
                text_features=text_features,
                idx2label=idx2label,
                batch_size=args.batch_size,
                use_test_ensemble=args.use_test_ensemble,
                ensemble_group_size=args.ensemble_group_size,
                args=args,
                split_name="test_zero_shot_online",
            )
            zero_shot_val = summarize_predictions(
                [sample["label"] for sample in zero_shot_val_outputs["samples"]],
                zero_shot_val_outputs["preds"],
            )
            zero_shot_test = summarize_predictions(
                [sample["label"] for sample in zero_shot_test_outputs["samples"]],
                zero_shot_test_outputs["preds"],
            )
        else:
            zero_shot_val_pred = predict_zeroshot_from_features(
                val_x,
                text_features,
                idx2label,
                args.batch_size,
                use_test_ensemble=args.use_test_ensemble,
                ensemble_group_size=args.ensemble_group_size,
            )
            zero_shot_test_pred = predict_zeroshot_from_features(
                test_x,
                text_features,
                idx2label,
                args.batch_size,
                use_test_ensemble=args.use_test_ensemble,
                ensemble_group_size=args.ensemble_group_size,
            )
            zero_shot_val = summarize_predictions(val_true, zero_shot_val_pred)
            zero_shot_test = summarize_predictions(test_true, zero_shot_test_pred)
        log(f"[ZEROSHOT] val_acc={zero_shot_val['accuracy']:.6f} val_wf1={zero_shot_val['weighted_f1']:.6f}")
        log(f"[ZEROSHOT] test_acc={zero_shot_test['accuracy']:.6f} test_wf1={zero_shot_test['weighted_f1']:.6f}")
        if args.report_train_metrics:
            if args.unfreeze_last_visual_block:
                zero_shot_train_outputs = predict_zeroshot_from_samples(
                    samples=train_samples,
                    processor=processor,
                    model=model,
                    text_features=text_features,
                    idx2label=idx2label,
                    batch_size=args.batch_size,
                    use_test_ensemble=args.use_test_ensemble,
                    ensemble_group_size=args.ensemble_group_size,
                    args=args,
                    split_name="train_zero_shot_online",
                )
                zero_shot_train = summarize_predictions(
                    [sample["label"] for sample in zero_shot_train_outputs["samples"]],
                    zero_shot_train_outputs["preds"],
                )
            else:
                zero_shot_train_pred = predict_zeroshot_from_features(
                    train_x,
                    text_features,
                    idx2label,
                    args.batch_size,
                    use_test_ensemble=args.use_test_ensemble,
                    ensemble_group_size=args.ensemble_group_size,
                )
                zero_shot_train = summarize_predictions(train_true, zero_shot_train_pred)

    adapter = None
    train_summary = None
    if args.zero_shot_only:
        if zero_shot_val is None or zero_shot_test is None:
            raise RuntimeError("zero-shot metrics were not computed")
        val_summary = zero_shot_val
        test_summary = zero_shot_test
        if args.report_train_metrics and zero_shot_train is not None:
            train_summary = zero_shot_train
    else:
        feature_dim = int(getattr(model.config, "projection_dim", 512)) if args.unfreeze_last_visual_block else int(train_x.shape[-1])
        if args.temporal_head == "transformer":
            adapter = TemporalTransformerClipImageAdapter(
                dim=feature_dim,
                device=args.device,
                hidden_dim=args.adapter_hidden_dim,
                dropout=args.adapter_dropout,
                num_classes=len(EMOTION_LABELS),
                num_prompts=int(text_features.shape[1]),
                num_frames=args.num_frames,
                temporal_num_heads=args.temporal_num_heads,
                temporal_num_layers=args.temporal_num_layers,
                temporal_pooling=args.temporal_pooling,
                use_qcpa=args.use_qcpa,
                qcpa_num_heads=args.qcpa_num_heads,
                use_global_logit_scale=args.use_global_logit_scale,
                use_prompt_weight=args.use_prompt_weight,
                use_class_temperature=args.use_class_temperature,
                use_class_bias=args.use_class_bias,
                attention_scope=args.attention_scope,
                dual_stage=args.dual_stage,
                use_residual_gate=not args.no_residual_gate,
                use_mahalanobis_temp=not args.no_mahalanobis,
                use_lowrank_bias=not args.no_lowrank_bias,
                bias_rank=args.bias_rank,
            )
        else:
            adapter = ClipImageAdapter(
                dim=feature_dim,
                device=args.device,
                hidden_dim=args.adapter_hidden_dim,
                dropout=args.adapter_dropout,
                num_classes=len(EMOTION_LABELS),
                num_prompts=int(text_features.shape[1]),
                use_qcpa=args.use_qcpa,
                qcpa_num_heads=args.qcpa_num_heads,
                use_global_logit_scale=args.use_global_logit_scale,
                use_prompt_weight=args.use_prompt_weight,
                use_class_temperature=args.use_class_temperature,
                use_class_bias=args.use_class_bias,
                attention_scope=args.attention_scope,
                dual_stage=args.dual_stage,
                use_residual_gate=not args.no_residual_gate,
                use_mahalanobis_temp=not args.no_mahalanobis,
                use_lowrank_bias=not args.no_lowrank_bias,
                bias_rank=args.bias_rank,
            )
        if args.strict_frozen_clip and not args.unfreeze_last_visual_block and args.use_qcpa and hasattr(adapter, "qcpa_parameters"):
            qcpa_param_count = count_parameters(adapter.qcpa_parameters())
            taga_param_count = count_parameters(adapter.taga_parameters()) if hasattr(adapter, "taga_parameters") else 0
            adapter_param_count = count_parameters(adapter.adapter_parameters()) if hasattr(adapter, "adapter_parameters") else 0
            trainable_param_count = count_parameters(adapter.parameters())
            frozen_backbone_param_count = int(sum(param.numel() for param in model.parameters() if not param.requires_grad))
            trainable_ratio = trainable_param_count / max(1, frozen_backbone_param_count)
            log(
                f"[QCPA] params | head={qcpa_param_count} | taga={taga_param_count} | adapter={adapter_param_count} | "
                f"trainable={trainable_param_count} | frozen_backbone={frozen_backbone_param_count} | ratio={trainable_ratio:.6f}"
            )
        if args.unfreeze_last_visual_block:
            adapter = train_unfreeze_last_visual_block_clip(
                train_samples=train_samples,
                val_samples=val_samples,
                processor=processor,
                model=model,
                text_features=text_features,
                adapter=adapter,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                visual_block_lr_scale=args.visual_block_lr_scale,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                use_class_weight=args.use_class_weight,
                label_smoothing=args.label_smoothing,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                select_metric=args.select_metric,
                use_test_ensemble=args.use_test_ensemble,
                ensemble_group_size=args.ensemble_group_size,
                use_amp=args.use_amp,
                lr_scheduler=args.lr_scheduler,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                args=args,
            )
            val_outputs = predict_emotion_from_samples_with_adapter(
                samples=val_samples,
                processor=processor,
                model=model,
                text_features=text_features,
                adapter=adapter,
                idx2label=idx2label,
                batch_size=args.batch_size,
                use_test_ensemble=args.use_test_ensemble,
                ensemble_group_size=args.ensemble_group_size,
                args=args,
                split_name="val_online_final",
            )
            test_outputs = predict_emotion_from_samples_with_adapter(
                samples=test_samples,
                processor=processor,
                model=model,
                text_features=text_features,
                adapter=adapter,
                idx2label=idx2label,
                batch_size=args.batch_size,
                use_test_ensemble=args.use_test_ensemble,
                ensemble_group_size=args.ensemble_group_size,
                args=args,
                split_name="test_online_final",
            )
            val_cache_payload = {
                "failed_count": len(val_outputs["failed_samples"]),
                "sampling_debug": val_outputs["sampling_debug"],
            }
            test_cache_payload = {
                "failed_count": len(test_outputs["failed_samples"]),
                "sampling_debug": test_outputs["sampling_debug"],
            }
            val_true = [sample["label"] for sample in val_outputs["samples"]]
            test_true = [sample["label"] for sample in test_outputs["samples"]]
            val_summary = summarize_predictions(val_true, val_outputs["preds"])
            test_summary = summarize_predictions(test_true, test_outputs["preds"])
            if args.report_train_metrics:
                train_outputs = predict_emotion_from_samples_with_adapter(
                    samples=train_samples,
                    processor=processor,
                    model=model,
                    text_features=text_features,
                    adapter=adapter,
                    idx2label=idx2label,
                    batch_size=args.batch_size,
                    use_test_ensemble=args.use_test_ensemble,
                    ensemble_group_size=args.ensemble_group_size,
                    args=args,
                    split_name="train_online_final",
                )
                train_cache_payload = {
                    "failed_count": len(train_outputs["failed_samples"]),
                    "sampling_debug": train_outputs["sampling_debug"],
                }
                train_true = [sample["label"] for sample in train_outputs["samples"]]
                train_summary = summarize_predictions(train_true, train_outputs["preds"])
        else:
            adapter = train_strict_frozen_clip(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                text_features=text_features,
                adapter=adapter,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                use_class_weight=args.use_class_weight,
                label_smoothing=args.label_smoothing,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                select_metric=args.select_metric,
                use_test_ensemble=args.use_test_ensemble,
                ensemble_group_size=args.ensemble_group_size,
                use_amp=args.use_amp,
                lr_scheduler=args.lr_scheduler,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
            )

            val_pred = predict_emotion_from_features(
                val_x,
                text_features,
                adapter,
                idx2label,
                args.batch_size,
                use_test_ensemble=args.use_test_ensemble,
                ensemble_group_size=args.ensemble_group_size,
                dump_attn_path=str(Path(args.output).with_name(f"{Path(args.output).stem}_val_attn.npy")) if args.dump_attn else None,
            )
            test_pred = predict_emotion_from_features(
                test_x,
                text_features,
                adapter,
                idx2label,
                args.batch_size,
                use_test_ensemble=args.use_test_ensemble,
                ensemble_group_size=args.ensemble_group_size,
                dump_attn_path=str(Path(args.output).with_name(f"{Path(args.output).stem}_test_attn.npy")) if args.dump_attn else None,
            )
            val_summary = summarize_predictions(val_true, val_pred)
            test_summary = summarize_predictions(test_true, test_pred)
            if args.report_train_metrics:
                train_pred = predict_emotion_from_features(
                    train_x,
                    text_features,
                    adapter,
                    idx2label,
                    args.batch_size,
                    use_test_ensemble=args.use_test_ensemble,
                    ensemble_group_size=args.ensemble_group_size,
                )
                train_summary = summarize_predictions(train_true, train_pred)

    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else default_checkpoint_path(Path(args.output))

    result = {
        "config": {
            "method": "clip_supervised_text_image_emotion",
            "experiment_name": args.experiment_name,
            "execution_mode": "zero_shot_only" if args.zero_shot_only else "strict_frozen_clip_adapter",
            "dataset": "RAVDESS",
            "task": "emotion",
            "split": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": round(1 - args.train_ratio - args.val_ratio, 6),
            },
            "split_strategy": "benchmark_5fold" if args.split_mode == "benchmark_5fold" else "actor_disjoint",
            "split_mode": args.split_mode,
            "benchmark_video_list": args.benchmark_video_list if args.split_mode == "benchmark_5fold" else None,
            "benchmark_use_video_list": args.benchmark_use_video_list if args.split_mode == "benchmark_5fold" else False,
            "benchmark_test_fold": args.benchmark_test_fold if args.split_mode == "benchmark_5fold" else None,
            "benchmark_val_fold": effective_benchmark_val_fold if args.split_mode == "benchmark_5fold" else None,
            "ravdess_root": str(Path(args.ravdess_root).resolve()),
            "allowed_modalities": sorted(allowed_modalities),
            "allowed_vocal_channels": sorted(allowed_vocal_channels),
            "allowed_intensities": sorted(allowed_intensities),
            "actor_ids": sorted(allowed_actor_ids) if allowed_actor_ids is not None else None,
            "video_extensions": sorted(allowed_extensions),
            "model_id": args.model_id,
            "prompt_template": args.prompt_template,
            "prompt_set": args.prompt_set,
            "auto_prompt_k": args.auto_prompt_k,
            "auto_prompt_refine_passes": args.auto_prompt_refine_passes,
            "auto_prompt_top_pairs": args.auto_prompt_top_pairs,
            "epochs": args.epochs,
            "extract_batch_size": args.extract_batch_size,
            "train_batch_size": args.train_batch_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "num_frames": args.num_frames,
            "frame_sampling_mode": args.frame_sampling_mode,
            "sampling_window_start": args.sampling_window_start,
            "sampling_window_end": args.sampling_window_end,
            "diff_alpha": args.diff_alpha,
            "diff_beta": args.diff_beta,
            "min_gap_ratio": args.min_gap_ratio,
            "score_smooth_window": args.score_smooth_window,
            "frame_diff_metric": args.frame_diff_metric,
            "feature_layout": args.feature_layout,
            "adapter_hidden_dim": args.adapter_hidden_dim,
            "adapter_dropout": args.adapter_dropout,
            "temporal_head": args.temporal_head,
            "temporal_num_heads": args.temporal_num_heads,
            "temporal_num_layers": args.temporal_num_layers,
            "temporal_pooling": args.temporal_pooling,
            "use_class_weight": args.use_class_weight,
            "label_smoothing": args.label_smoothing,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "select_metric": args.select_metric,
            "use_test_ensemble": args.use_test_ensemble,
            "ensemble_group_size": args.ensemble_group_size,
            "strict_frozen_clip": args.strict_frozen_clip,
            "unfreeze_last_visual_block": args.unfreeze_last_visual_block,
            "visual_block_lr_scale": args.visual_block_lr_scale,
            "use_global_logit_scale": args.use_global_logit_scale,
            "use_prompt_weight": args.use_prompt_weight,
            "use_class_temperature": args.use_class_temperature,
            "use_class_bias": args.use_class_bias,
            "use_qcpa": args.use_qcpa,
            "head_type": "qcpa" if args.use_qcpa else "capc",
            "qcpa_num_heads": args.qcpa_num_heads,
            "attention_scope": args.attention_scope,
            "dual_stage": args.dual_stage,
            "use_residual_gate": not args.no_residual_gate,
            "use_mahalanobis_temp": not args.no_mahalanobis,
            "use_lowrank_bias": not args.no_lowrank_bias,
            "bias_rank": args.bias_rank,
            "dump_attn": args.dump_attn,
            "use_amp": args.use_amp,
            "strict_seed_control": args.strict_seed_control,
            "lr_scheduler": args.lr_scheduler,
            "disable_scheduler": args.disable_scheduler,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "run_zero_shot_eval": args.run_zero_shot_eval,
            "zero_shot_only": args.zero_shot_only,
            "report_train_metrics": args.report_train_metrics,
            "feature_cache_dir": args.feature_cache_dir,
            "legacy_feature_cache_path": resolved_legacy_cache_path,
            "resolved_feature_cache_path": resolved_legacy_cache_path,
            "resolved_feature_cache_paths": resolved_split_cache_paths,
            "checkpoint_output": str(checkpoint_path),
            "log_file": resolved_log_file,
            "prompts_per_class": len(prompt_groups[0]) if prompt_groups else 0,
            "total_text_prompts": sum(len(group) for group in prompt_groups),
            "seed": args.seed,
            "max_sequences": args.max_sequences,
        },
        "dataset": {
            "total": len(samples),
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "num_actors_total": len(sorted({int(sample["actor_id"]) for sample in samples})),
            "actors_train": sorted({int(sample["actor_id"]) for sample in train_samples}),
            "actors_val": sorted({int(sample["actor_id"]) for sample in val_samples}),
            "actors_test": sorted({int(sample["actor_id"]) for sample in test_samples}),
            "benchmark_folds_present": sorted({int(sample["benchmark_fold"]) for sample in samples if sample.get("benchmark_fold") is not None}),
            "label_distribution_train": dict(Counter([s["label"] for s in train_samples])),
            "vocal_channel_distribution_train": dict(Counter([s["vocal_channel"] for s in train_samples])),
            "modality_distribution_train": dict(Counter([s["modality"] for s in train_samples])),
            "failed_samples_train": train_cache_payload.get("failed_count", 0),
            "failed_samples_val": val_cache_payload.get("failed_count", 0),
            "failed_samples_test": test_cache_payload.get("failed_count", 0),
        },
        "label_map": RAVDESS_LABEL_MAP,
        "prompt_groups": prompt_groups,
        "text_prototype_similarity": text_prototype_similarity,
        "text_similarity_stats": text_similarity_stats,
        "val": val_summary,
        "test": test_summary,
    }

    if prompt_selection_info is not None:
        result["prompt_selection"] = prompt_selection_info

    result["sampling_debug_examples"] = {
        "train": list(train_cache_payload.get("sampling_debug", [])),
        "val": list(val_cache_payload.get("sampling_debug", [])),
        "test": list(test_cache_payload.get("sampling_debug", [])),
    }

    if args.report_train_metrics and train_summary is not None:
        result["train"] = train_summary
        result["train_confusion_matrix"] = train_summary["confusion_matrix"]
        result["train_prediction_distribution"] = train_summary["prediction_distribution"]

    if adapter is not None:
        if getattr(adapter, "use_qcpa", False) and getattr(adapter, "head", None) is not None:
            result["learned_global_logit_scale"] = round(float(adapter.head.log_scale.exp().detach().cpu().item()), 6)
            result["learned_qcpa_tau"] = format_tensor_list(adapter.head.log_tau.exp(), precision=6)
            if getattr(adapter.head, "gate", None) is not None:
                result["learned_qcpa_gate"] = format_tensor_list(adapter.head.gate, precision=6)
        elif getattr(adapter, "legacy_head", None) is not None and adapter.legacy_head.use_global_logit_scale:
            result["learned_global_logit_scale"] = round(float(adapter.legacy_head.logit_scale.exp().detach().cpu().item()), 6)
        training_diagnostics = getattr(adapter, "training_diagnostics", None)
        if isinstance(training_diagnostics, dict):
            result.update(training_diagnostics)

    result["val_prediction_distribution"] = val_summary["prediction_distribution"]
    result["test_prediction_distribution"] = test_summary["prediction_distribution"]

    if zero_shot_val is not None:
        result["zero_shot_val_metrics"] = zero_shot_val
    if zero_shot_test is not None:
        result["zero_shot_test_metrics"] = zero_shot_test
    if zero_shot_train is not None:
        result["zero_shot_train_metrics"] = zero_shot_train

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "config": result["config"],
        "dataset": result["dataset"],
        "label_map": result["label_map"],
        "metrics": {
            "val": result["val"],
            "test": result["test"],
        },
        "text_prototype_similarity": result["text_prototype_similarity"],
        "text_similarity_stats": result["text_similarity_stats"],
        "prompt_groups": prompt_groups,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "output_path": str(output_path),
    }
    if prompt_selection_info is not None:
        checkpoint_payload["prompt_selection"] = prompt_selection_info
    if adapter is not None:
        checkpoint_payload.update(
            {
                "checkpoint_type": "strict_frozen_clip_adapter_unfreeze_last_visual_block" if args.unfreeze_last_visual_block else "strict_frozen_clip_adapter",
                "adapter_state_dict": adapter.state_dict(),
                "text_features": text_features.cpu(),
            }
        )
        if args.unfreeze_last_visual_block:
            checkpoint_payload["clip_last_visual_block_state_dict"] = model.vision_model.encoder.layers[-1].state_dict()
    else:
        checkpoint_payload.update(
            {
                "checkpoint_type": "zero_shot_text_prototypes",
                "text_features": text_features.cpu(),
            }
        )
    torch.save(checkpoint_payload, checkpoint_path)

    log(f"[DONE] saved supervised CLIP emotion report to: {output_path}")
    log(f"[DONE] saved checkpoint to: {checkpoint_path}")
    log(f"[DONE] final test metrics: {json.dumps(result['test'], ensure_ascii=False)}")
    print(json.dumps({"test": result["test"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
