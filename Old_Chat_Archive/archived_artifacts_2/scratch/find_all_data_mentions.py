import re

with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_pdf_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Let's search for sentences containing "dataset" or "database"
sentences = re.findall(r"[^.!?]*\b(?:dataset|data set|database|databases)\b[^.!?]*[.!?]", text, re.IGNORECASE)

print(f"Total sentences found: {len(sentences)}")

# Let's write them to a file to examine
with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\dataset_sentences.txt", "w", encoding="utf-8") as f:
    for idx, sentence in enumerate(sentences):
        # clean whitespace
        clean_sentence = " ".join(sentence.split())
        f.write(f"Sentence {idx + 1}: {clean_sentence}\n\n")

print("Done. Sentences written to dataset_sentences.txt")
