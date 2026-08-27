"""Comprehensive unit and multilingual tests for Indian Legal Tokenizer."""

from pathlib import Path
import tempfile
import pytest

from src.config import TokenizerConfig
from src.tokenizer import LegalTokenizer, TokenizerOutput


@pytest.fixture
def sample_corpus_path():
    """Return path to the synthetic legal corpus."""
    return Path(__file__).resolve().parent.parent / "data" / "synthetic_legal_corpus.txt"


@pytest.fixture
def trained_tokenizer(sample_corpus_path):
    """Provide a trained LegalTokenizer instance for testing."""
    tok = LegalTokenizer(config=TokenizerConfig(vocab_size=800, min_frequency=1))
    tok.train_from_files([sample_corpus_path], vocab_size=800, min_frequency=1)
    return tok


def test_tokenizer_initialization():
    """Verify tokenizer initializes with proper config and special tokens."""
    tokenizer = LegalTokenizer()
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.eos_token == "</s>"
    assert len(tokenizer.special_tokens) == 4


def test_tokenizer_training_from_iterator():
    """Verify training from in-memory string iterator."""
    corpus = [
        "Section 302 of the Indian Penal Code",
        "Article 21 Constitution of India",
        "Section 187 BNSS",
        "भारतीय न्याय संहिता",
    ]
    tok = LegalTokenizer(config=TokenizerConfig(vocab_size=400, min_frequency=1))
    tok.train_from_iterator(corpus, vocab_size=400, min_frequency=1)
    assert tok.vocab_size >= len(tok.special_tokens)
    assert tok.vocab_size <= 400


def test_tokenizer_training_from_files(sample_corpus_path):
    """Verify training from text files on disk."""
    tok = LegalTokenizer(config=TokenizerConfig(vocab_size=500, min_frequency=1))
    tok.train_from_files([sample_corpus_path], vocab_size=500, min_frequency=1)
    assert tok.vocab_size >= 256  # Initial byte alphabet + merges
    assert tok.vocab_size <= 500


def test_encode_and_decode_roundtrip(trained_tokenizer):
    """Verify standard text encodes and decodes losslessly."""
    text = "Section 302 of the Indian Penal Code provides punishment for murder."
    out = trained_tokenizer.encode(text)
    assert isinstance(out, TokenizerOutput)
    assert len(out.ids) > 0
    assert len(out.tokens) == len(out.ids)
    assert len(out.attention_mask) == len(out.ids)

    decoded = trained_tokenizer.decode(out.ids)
    assert decoded == text


def test_encode_with_special_tokens(trained_tokenizer):
    """Verify encoding with add_special_tokens=True injects BOS and EOS."""
    text = "Article 21 of the Constitution of India"
    out = trained_tokenizer.encode(text, add_special_tokens=True)

    assert out.ids[0] == trained_tokenizer.bos_token_id
    assert out.ids[-1] == trained_tokenizer.eos_token_id
    assert out.tokens[0] == trained_tokenizer.bos_token
    assert out.tokens[-1] == trained_tokenizer.eos_token

    # Decode skipping special tokens should return original text
    decoded_clean = trained_tokenizer.decode(out.ids, skip_special_tokens=True)
    assert decoded_clean == text


def test_special_token_id_mapping(trained_tokenizer):
    """Verify special token IDs match expected indices."""
    assert trained_tokenizer.token_to_id("<pad>") == trained_tokenizer.pad_token_id
    assert trained_tokenizer.token_to_id("<unk>") == trained_tokenizer.unk_token_id
    assert trained_tokenizer.token_to_id("<s>") == trained_tokenizer.bos_token_id
    assert trained_tokenizer.token_to_id("</s>") == trained_tokenizer.eos_token_id


def test_save_and_load_persistence(trained_tokenizer):
    """Verify tokenizer can be saved to disk and reloaded identically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_paths = trained_tokenizer.save(tmpdir)
        assert Path(saved_paths["tokenizer_file"]).exists()
        assert Path(saved_paths["config_file"]).exists()

        reloaded_tok = LegalTokenizer.load(tmpdir)
        assert reloaded_tok.vocab_size == trained_tokenizer.vocab_size
        assert reloaded_tok.special_tokens == trained_tokenizer.special_tokens

        text = "Section 187 of the Bharatiya Nagarik Suraksha Sanhita, 2023"
        orig_enc = trained_tokenizer.encode(text)
        reloaded_enc = reloaded_tok.encode(text)
        assert orig_enc.ids == reloaded_enc.ids
        assert reloaded_tok.decode(reloaded_enc.ids) == text


def test_unicode_and_legal_punctuation(trained_tokenizer):
    """Verify legal section marks, em-dashes, paragraph marks, and quotes round-trip perfectly."""
    legal_text = "§ 125(1)(a) — Maintenance of wives, children and parents; ¶ 14: “Quashing of FIR”."
    out = trained_tokenizer.encode(legal_text)
    decoded = trained_tokenizer.decode(out.ids)
    assert decoded == legal_text


@pytest.mark.parametrize(
    "language,text",
    [
        ("English", "The accused was produced before the Magistrate."),
        ("Hindi", "अभियुक्त को मजिस्ट्रेट के समक्ष पेश किया गया।"),
        ("Kannada", "ಆರೋಪಿಯನ್ನು ಮ್ಯಾಜಿಸ್ಟ್ರೇಟ್ ಮುಂದೆ ಹಾಜರುಪಡಿಸಲಾಯಿತು."),
        ("Tamil", "குற்றம் சாட்டப்பட்டவர் மாஜிஸ்திரேட் முன் ஆஜர்படுத்தப்பட்டார்."),
        ("Telugu", "నిందితుడిని మేజిస్ట్రేట్ ముందు హాజరుపరిచారు."),
        ("Malayalam", "പ്രതിയെ മജിസ്ട്രേറ്റിന് മുമ്പാകെ ഹാജരാക്കി."),
        ("Bengali", "অভিযুক্তকে ম্যাজিস্ট্রেটের সামনে হাজির করা হয়েছিল।"),
        ("Marathi", "आरोपीला दंडाधिकार्‍यांसमोर हजर करण्यात आले."),
        ("Gujarati", "આરોપીને મેજિસ્ટ્રેટ સમક્ષ રજૂ કરવામાં આવ્યો હતો."),
        ("Punjabi", "ਦੋਸ਼ੀ ਨੂੰ ਮੈਜਿਸਟ੍ਰੇਟ ਦੇ ਸਾਹਮਣੇ ਪੇਸ਼ ਕੀਤਾ ਗਿਆ।"),
        ("Urdu", "ملزم کو مجسٹریٹ کے سامنے پیش کیا گیا۔"),
        ("Sanskrit", "अभियुक्तः दण्डाधिकारिणः समक्षम् उपस्थापितः। धर्मो रक्षति रक्षितः।"),
    ],
)
def test_multilingual_indian_languages(trained_tokenizer, language, text):
    """Verify lossless tokenization and roundtrip reconstruction across 12 Indic languages."""
    out = trained_tokenizer.encode(text)
    assert len(out.ids) > 0
    # Zero unknown tokens due to byte-level fallback
    assert trained_tokenizer.unk_token_id not in out.ids
    decoded = trained_tokenizer.decode(out.ids)
    assert decoded == text, f"Failed roundtrip for {language}"


def test_complex_legal_citations(trained_tokenizer):
    """Verify complex Indian case citations are tokenized cleanly."""
    citations = [
        "2024 INSC 123",
        "AIR 1980 SC 1789",
        "(2023) 5 SCC 123",
        "2021 SCC OnLine SC 315",
        "MANU/SC/0001/2024",
        "Criminal Appeal No. 1024 of 2023 (Arising out of SLP (Crl.) No. 5432 of 2022)",
    ]
    for cit in citations:
        out = trained_tokenizer.encode(cit)
        assert trained_tokenizer.decode(out.ids) == cit


def test_mixed_language_code_switched(trained_tokenizer):
    """Verify code-mixed Indian English and Indic sentences."""
    mixed_text = "Section 187 BNSS के अंतर्गत custody की अवधि 15 days से अधिक नहीं होनी चाहिए."
    out = trained_tokenizer.encode(mixed_text)
    assert trained_tokenizer.decode(out.ids) == mixed_text


def test_empty_and_short_strings(trained_tokenizer):
    """Verify edge cases like empty string, whitespace, and single characters."""
    assert trained_tokenizer.decode(trained_tokenizer.encode("").ids) == ""
    assert trained_tokenizer.decode(trained_tokenizer.encode(" ").ids) == " "
    assert trained_tokenizer.decode(trained_tokenizer.encode("   \n\n\t  ").ids) == "   \n\n\t  "
    assert trained_tokenizer.decode(trained_tokenizer.encode("A").ids) == "A"
    assert trained_tokenizer.decode(trained_tokenizer.encode("§").ids) == "§"
    assert trained_tokenizer.decode(trained_tokenizer.encode("क").ids) == "क"


def test_long_multiparagraph_text(trained_tokenizer):
    """Verify tokenizing long multi-paragraph judgment text."""
    long_text = """IN THE SUPREME COURT OF INDIA
CRIMINAL APPELLATE JURISDICTION

Criminal Appeal No. 1024 of 2023
(Arising out of SLP (Crl.) No. 5432 of 2022)

State of Maharashtra ... Appellant(s)
VERSUS
XYZ ... Respondent(s)

J U D G M E N T

Dr Dhananjaya Y Chandrachud, CJI

1. Leave granted.
2. The present appeal arises out of the final judgment and order passed by the High Court of Judicature at Bombay under Section 482 of the Code of Criminal Procedure, 1973.
3. Having heard the learned senior counsel appearing for the parties, we are of the considered opinion that the High Court committed a palpable error of law.
4. Accordingly, the appeal is allowed and the impugned judgment is set aside.
"""
    out = trained_tokenizer.encode(long_text)
    assert len(out.ids) > 50
    assert trained_tokenizer.decode(out.ids) == long_text


def test_deterministic_behavior(trained_tokenizer):
    """Verify that multiple consecutive encodings yield identical token IDs."""
    text = "Section 103 Bharatiya Nyaya Sanhita, 2023"
    ids1 = trained_tokenizer.encode(text).ids
    ids2 = trained_tokenizer.encode(text).ids
    assert ids1 == ids2
