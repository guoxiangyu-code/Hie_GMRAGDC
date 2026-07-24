import json
import os
import re

repo_root = '/home/guoxiangyu/generalized-moment-retrieval'
pdf_text_path = '/tmp/pdf_text.txt'
doc_path = os.path.join(repo_root, 'GMR_Progress_Report_Updated_20260723.md')

print("========================================================================")
print("GOAL AUDIT: PDF Registration Coverage & Log/Method Traceability Check")
print("========================================================================")

# 1. Load PDF text and check key figures & experiments
with open(pdf_text_path, encoding='utf-8') as f:
    pdf_text = f.read()

# PDF Table entries
pdf_entries = [
    ("Moment-DETR Strict GMR (Base)", "8.93", "30.78", "70.95", "61.22"),
    ("HieA2M-DGQC + Independent Zero", "9.16", "39.77", "72.62", "69.28"),
    ("EaTR Strict GMR (Base)", "8.02", "16.82", "71.67", "39.35"),
    ("EaTR + Quality", "8.24", "19.13", "71.98", "44.31"),
    ("Flash-VTG GMR Release Anchor", "26.01", "33.93", "73.95", "62.53"),
    ("Flash-VTG GMR + Quality", "26.67", "34.03", "73.95", "62.53"),
    ("EaTR + Dual Grounding", "8.06", "21.10", "72.05", "48.12"),
    ("Learned Dedup Top-3", "6.96", "5.06", "69.55", "7.66"),
    ("Learned Dedup + Soft Count", "6.72", "7.25", "69.55", "7.66"),
    ("QD Continued Control", "6.91", "35.23", "72.02", "65.96"),
    ("QD + Quality + Dual", "6.27", "42.04", "72.74", "70.26"),
]

with open(doc_path, encoding='utf-8') as f:
    doc_text = f.read()

print("\n--- [Audit 1: PDF Entry Registration Check] ---")
pdf_audit_passed = True
for name, map_v, g3_v, auroc_v, rej_v in pdf_entries:
    found_map = map_v in doc_text
    found_g3 = g3_v in doc_text
    status = "✅ PASS" if (found_map and found_g3) else "❌ MISSING"
    print(f"{status} | {name:32s} | mAP={map_v:6s} | G@3={g3_v:6s}")
    if not (found_map and found_g3):
        pdf_audit_passed = False

print("\n--- [Audit 2: Method, Code & Log Traceability Check] ---")

all_json_logs_valid = True
# Extract all markdown links and JSON/Log paths from doc_text
json_matches = re.findall(r'`(artifacts/[^`]+\.json)`', doc_text)
log_matches = re.findall(r'`(artifacts/[^`]+\.(?:log|txt|jsonl))`', doc_text)

print(f"Found {len(json_matches)} JSON artifact references and {len(log_matches)} Log file references.")

missing_jsons = []
for jpath in json_matches:
    full_j = os.path.join(repo_root, jpath)
    if not os.path.exists(full_j):
        missing_jsons.append(jpath)
        all_json_logs_valid = False

missing_logs = []
for lpath in log_matches:
    full_l = os.path.join(repo_root, lpath)
    if not os.path.exists(full_l):
        missing_logs.append(lpath)
        all_json_logs_valid = False

print(f"Missing JSON files: {len(missing_jsons)}")
print(f"Missing Log files: {len(missing_logs)}")

if missing_jsons:
    for m in missing_jsons:
        print(f"  ❌ Missing JSON: {m}")

if missing_logs:
    for m in missing_logs:
        print(f"  ❌ Missing Log: {m}")

print("========================================================================")
if pdf_audit_passed and all_json_logs_valid:
    print("🎉 ALL AUDIT CHECKS PASSED 100%! The document is fully compliant.")
else:
    print("⚠️ Audit identified items to fix. Updating document...")
