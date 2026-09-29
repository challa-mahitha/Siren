import os
import zipfile
import json
import xml.etree.ElementTree as ET

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "pan12_raw")
INNER_TRAIN_ZIP = os.path.join(RAW_DIR, "pan12-sexual-predator-identification-training-corpus-2012-05-01.zip")
TRAIN_EXTRACT_DIR = os.path.join(RAW_DIR, "training_unpacked")
OUTPUT_CORPUS = os.path.join(DATA_DIR, "training_corpus.json")

def ingest_nested_pan12():
    print("=" * 65)
    print("  SIREN: Extracting Nested PAN-12 Training Corpus")
    print("=" * 65)

    if not os.path.exists(INNER_TRAIN_ZIP):
        print(f"[!] Could not find {INNER_TRAIN_ZIP}")
        return

    # 1. Unpack the inner training zip
    if not os.path.exists(TRAIN_EXTRACT_DIR):
        print(f"[*] Extracting inner zip: {os.path.basename(INNER_TRAIN_ZIP)}...")
        with zipfile.ZipFile(INNER_TRAIN_ZIP, 'r') as z:
            z.extractall(TRAIN_EXTRACT_DIR)
        print("[+] Unpacked inner training archive.")
    else:
        print("[*] Found existing unpacked training files.")

    # 2. Locate the XML and Predator ID files
    xml_path = None
    predator_txt = None

    for root, dirs, files in os.walk(TRAIN_EXTRACT_DIR):
        for f in files:
            full = os.path.join(root, f)
            if f.endswith(".xml"):
                xml_path = full
            elif "predator" in f.lower() and f.endswith(".txt"):
                predator_txt = full

    print(f"[*] Conversation XML: {xml_path}")
    print(f"[*] Predator List:     {predator_txt}")

    if not xml_path or not predator_txt:
        print("[!] Missing XML or predator list file.")
        return

    # 3. Read confirmed predator author hashes
    with open(predator_txt, "r", encoding="utf-8") as pf:
        predator_ids = set(line.strip() for line in pf if line.strip())
    print(f"[*] Loaded {len(predator_ids)} validated predator author profiles.")

    # 4. Stream-parse conversation messages
    print("[*] Parsing XML turns (extracting 3,000 predatory + 4,000 benign baseline)...")
    predatory_samples = []
    benign_samples = []

    context = ET.iterparse(xml_path, events=("end",))
    for event, elem in context:
        if elem.tag == "message":
            author = elem.find("author")
            text_el = elem.find("text")

            if author is not None and text_el is not None and text_el.text:
                raw_text = text_el.text.strip()
                # Exclude single words, timestamps, or system messages
                if len(raw_text) > 15:
                    if author.text in predator_ids:
                        if len(predatory_samples) < 3000:
                            predatory_samples.append({"text": raw_text, "label": 1})
                    else:
                        if len(benign_samples) < 4000:
                            benign_samples.append({"text": raw_text, "label": 0})
            elem.clear()

        if len(predatory_samples) >= 3000 and len(benign_samples) >= 4000:
            break

    print(f"[+] Extracted: {len(predatory_samples)} predatory turns and {len(benign_samples)} benign turns.")

    unified = predatory_samples + benign_samples
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as out:
        json.dump(unified, out, indent=2)

    print(f"[+] Successfully wrote {len(unified)} records to: {OUTPUT_CORPUS}")

if __name__ == "__main__":
    ingest_nested_pan12()
