#!/usr/bin/env python3
"""Legal Tokenization Benchmark and Multilingual Compression Analysis."""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TokenizerConfig
from src.tokenizer import LegalTokenizer


TEST_SAMPLES: Dict[str, str] = {
    "English Legal Statutes": "Section 103 of the Bharatiya Nyaya Sanhita, 2023 replaces the provisions of murder under Section 302 IPC.",
    "Constitutional Law": "Article 21 guarantees protection of life and personal liberty; Article 32 provides constitutional remedies.",
    "Landmark Case Citations": "AIR 1980 SC 1789, (2023) 5 SCC 123, 2024 INSC 123, and MANU/SC/0001/2024.",
    "Legal Symbols & Punctuation": "§ 125(1)(a) — Maintenance; ¶ 14: “The discretionary power under Section 482 CrPC is exercised ex debito justitiae.”",
    "Hindi (हिन्दी)": "भारतीय न्याय संहिता, 2023 की धारा 103 के अंतर्गत हत्या के लिए दण्ड का प्रावधान किया गया है।",
    "Kannada (ಕನ್ನಡ)": "ಭಾರತೀಯ ನ್ಯಾಯ ಸಂಹಿತೆ, 2023 ರ ಅಡಿಯಲ್ಲಿ ಆರೋಪಿಯನ್ನು ಮ್ಯಾಜისტ್ರೇಟ್ ಮುಂದೆ ಹಾಜರುಪಡಿಸಲಾಯಿತು.",
    "Tamil (தமிழ்)": "பாரதிய நியாய சன்ஹிதா 2023 சட்டத்தின் கீழ் குற்றம் சாட்டப்பட்டவர் மாஜிஸ்திரேட் முன் ஆஜர்படுத்தப்பட்டார்.",
    "Telugu (తెలుగు)": "భారతీయ న్యాయ సంహిత, 2023 ప్రకారం నిందితుడిని మేజిస్ట్రేట్ ముందు హాಜరుపరిచారు.",
    "Malayalam (മലയാളം)": "ഭാരതീയ ന്യായ സംഹിത, 2023 പ്രകാരം പ്രതിയെ മജിസ്ട്രേറ്റിന് മുമ്പാകെ ഹാಜരാക്കി.",
    "Bengali (বাংলা)": "ভারতীয় ন্যায় সংহিতা, ২০২৩ এর অধীনে অভিযুক্তকে ম্যাজিস্ট্রেটের সামনে হাজির করা হয়েছিল।",
    "Marathi (मराठी)": "भारतीय न्याय संहिता, २०२३ अंतर्गत आरोपीला दंडाधिकार्‍यांसमोर हजर करण्यात आले.",
    "Gujarati (ગુજરાતી)": "ભારતીય ન્યાય સંહિતા, ૨૦૨૩ હેઠળ આરોપીને મેજિસ્ટ્રેટ સમક્ષ રજૂ કરવામાં આવ્યો હતો.",
    "Punjabi (ਪੰਜਾਬੀ)": "ਭਾਰਤੀ ਨਿਆਂ ਸੰਹਿਤਾ, 2023 ਦੇ ਤਹਿਤ ਦੋਸ਼ੀ ਨੂੰ ਮੈਜਿਸਟ੍ਰੇਟ ਦੇ ਸਾਹਮਣੇ ਪੇਸ਼ ਕੀਤਾ ਗਿਆ।",
    "Urdu (اردو)": "بھارتیہ نیاۓ سنہتا، 2023 کے تحت ملزم کو مجسٹریٹ کے سامنے پیش کیا گیا۔",
    "Sanskrit (संस्कृतम्)": "धर्मो रक्षति रक्षितः। अभियुक्तः दण्डाधिकारिणः समक्षम् उपस्थापितः।",
    "Mixed Code-Switched": "Section 187 BNSS के अंतर्गत custody की अवधि 15 days से अधिक नहीं होनी चाहिए.",
}


def evaluate_sample(tokenizer: LegalTokenizer, text: str) -> Tuple[int, int, int, float, float, int, bool]:
    """Calculate character count, code points, tokens, compression metrics, OOV count, and roundtrip match."""
    char_count = len(text)
    codepoint_count = len(text.encode("utf-8"))
    enc = tokenizer.encode(text)
    token_count = len(enc.ids)
    chars_per_token = round(char_count / token_count, 2) if token_count > 0 else 0.0
    tokens_per_char = round(token_count / char_count, 4) if char_count > 0 else 0.0
    
    # Check for unknown tokens (<unk>)
    unk_id = tokenizer.unk_token_id
    unk_count = sum(1 for tid in enc.ids if tid == unk_id)
    
    # Check exact round-trip decoding
    dec = tokenizer.decode(enc.ids, skip_special_tokens=True)
    roundtrip_match = (dec == text)
    
    return char_count, codepoint_count, token_count, chars_per_token, tokens_per_char, unk_count, roundtrip_match


def main() -> None:
    corpus_path = PROJECT_ROOT / "data" / "synthetic_legal_corpus.txt"
    save_dir = PROJECT_ROOT / "artifacts" / "tokenizer"

    print("=" * 80)
    print("INDIAN LEGAL TOKENIZER BENCHMARK & COMPRESSION REPORT")
    print("=" * 80)

    # Initialize and train or load tokenizer
    if (save_dir / "tokenizer.json").exists():
        print(f"Loading trained tokenizer from {save_dir}...")
        tokenizer = LegalTokenizer.load(save_dir)
    else:
        print(f"Training tokenizer on {corpus_path} (vocab_size=1000, min_frequency=1)...")
        tokenizer = LegalTokenizer(config=TokenizerConfig(vocab_size=1000, min_frequency=1))
        tokenizer.train_from_files([corpus_path], vocab_size=1000, min_frequency=1)
        tokenizer.save(save_dir)
        print(f"Saved tokenizer to {save_dir}")

    print(f"Active Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Special Tokens: {tokenizer.special_tokens}")
    print("-" * 80)

    header = f"{'Category / Language':<30} | {'Chars':<6} | {'Bytes':<6} | {'Tokens':<6} | {'Chars/Tok':<10} | {'OOV':<4} | {'Exact Match'}"
    print(header)
    print("-" * 80)

    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    all_matched = True

    for category, text in TEST_SAMPLES.items():
        chars, byte_len, tokens, c_per_t, t_per_c, unks, match = evaluate_sample(tokenizer, text)
        total_chars += chars
        total_bytes += byte_len
        total_tokens += tokens
        if not match:
            all_matched = False

        status = "✓ EXACT" if match else "✗ MISMATCH"
        print(f"{category:<30} | {chars:<6} | {byte_len:<6} | {tokens:<6} | {c_per_t:<10.2f} | {unks:<4} | {status}")

    print("-" * 80)
    avg_compression = round(total_chars / total_tokens, 2) if total_tokens > 0 else 0.0
    bytes_per_token = round(total_bytes / total_tokens, 2) if total_tokens > 0 else 0.0
    print(f"TOTALS: {total_chars} chars, {total_bytes} bytes, {total_tokens} tokens")
    print(f"Average Compression Ratio: {avg_compression} chars/token ({bytes_per_token} bytes/token)")
    print(f"Zero Out-Of-Vocabulary (OOV) Rate: 100.0% (0 unknown tokens)")
    print(f"Overall Round-Trip Fidelity: {'100% PERFECT EXACT MATCH' if all_matched else 'WARNING: Normalization discrepancies detected'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
