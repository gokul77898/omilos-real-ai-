"""Classification of court jurisdictions across Supreme Court and all 25 High Courts."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


ALL_25_HIGH_COURTS = [
    "Allahabad",
    "Andhra Pradesh",
    "Bombay",
    "Calcutta",
    "Chhattisgarh",
    "Delhi",
    "Gauhati",
    "Gujarat",
    "Himachal Pradesh",
    "Jammu & Kashmir and Ladakh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Madras",
    "Manipur",
    "Meghalaya",
    "Orissa",
    "Patna",
    "Punjab & Haryana",
    "Rajasthan",
    "Sikkim",
    "Telangana",
    "Tripura",
    "Uttarakhand",
]


class CourtClassifier:
    """Identifies judicial level (Supreme Court, High Court, Tribunal) and specific High Court jurisdiction."""

    HC_PATTERNS = {
        "Allahabad": [r"allahabad", r"high court of judicature at allahabad", r"lucknow bench"],
        "Andhra Pradesh": [r"andhra pradesh", r"high court of andhra pradesh", r"amaravati"],
        "Bombay": [r"bombay", r"high court of bombay", r"nagpur bench", r"aurangabad bench", r"goa bench"],
        "Calcutta": [r"calcutta", r"high court at calcutta", r"port blair bench", r"kolkata"],
        "Chhattisgarh": [r"chhattisgarh", r"high court of chhattisgarh", r"bilaspur"],
        "Delhi": [r"delhi", r"delhi high court", r"high court of delhi"],
        "Gauhati": [r"gauhati", r"guwahati", r"high court of assam"],
        "Gujarat": [r"gujarat", r"high court of gujarat", r"ahmedabad"],
        "Himachal Pradesh": [r"himachal pradesh", r"high court of himachal pradesh", r"shimla"],
        "Jammu & Kashmir and Ladakh": [r"jammu & kashmir", r"jammu and kashmir", r"srinagar bench", r"ladakh"],
        "Jharkhand": [r"jharkhand", r"high court of jharkhand", r"ranchi"],
        "Karnataka": [r"karnataka", r"high court of karnataka", r"dharwad bench", r"kalaburagi bench", r"bangalore"],
        "Kerala": [r"kerala", r"high court of kerala", r"ernakulam"],
        "Madhya Pradesh": [r"madhya pradesh", r"high court of madhya pradesh", r"jabalpur", r"indore bench", r"gwalior bench"],
        "Madras": [r"madras", r"high court of judicature at madras", r"madurai bench", r"chennai"],
        "Manipur": [r"manipur", r"high court of manipur", r"imphal"],
        "Meghalaya": [r"meghalaya", r"high court of meghalaya", r"shillong"],
        "Orissa": [r"orissa", r"odisha", r"high court of orissa", r"cuttack"],
        "Patna": [r"patna", r"high court of judicature at patna"],
        "Punjab & Haryana": [r"punjab & haryana", r"punjab and haryana", r"chandigarh"],
        "Rajasthan": [r"rajasthan", r"high court of judicature for rajasthan", r"jodhpur", r"jaipur bench"],
        "Sikkim": [r"sikkim", r"high court of sikkim", r"gangtok"],
        "Telangana": [r"telangana", r"high court for the state of telangana", r"hyderabad"],
        "Tripura": [r"tripura", r"high court of tripura", r"agartala"],
        "Uttarakhand": [r"uttarakhand", r"high court of uttarakhand", r"nainital"],
    }

    TRIBUNAL_PATTERNS = [
        r"nclt", r"nclat", r"itat", r"cat", r"ngt", r"aptel",
        r"drat", r"cestat", r"tribunal",
    ]

    LEGAL_DOMAINS = [
        "Constitutional", "Criminal", "Civil", "Commercial", "Corporate", "Tax",
        "Property", "Family", "Labour", "Administrative", "Environmental",
        "Intellectual Property", "Cyber", "Arbitration", "Consumer", "Insolvency",
        "Banking", "Evidence", "Procedure", "Human Rights", "Special Acts",
    ]

    @classmethod
    def classify(cls, text: str, metadata: Optional[Dict[str, str]] = None) -> Tuple[str, Optional[str], str]:
        """Classify judicial tier, specific High Court name (if applicable), and legal domain.

        Returns:
            Tuple of (court_level, high_court_name, legal_domain)
        """
        combined = f"{text[:2000]} {str(metadata or '')}".lower()

        # 1. Supreme Court
        if any(p in combined for p in ["supreme court of india", "in the supreme court", "insc", "scc", "hon'ble supreme court"]):
            court_level = "Supreme Court"
            hc_name = None
        # 2. Tribunals
        elif any(re.search(p, combined) for p in cls.TRIBUNAL_PATTERNS):
            court_level = "Tribunals"
            hc_name = None
        # 3. High Courts
        else:
            court_level = "High Courts"
            hc_name = None
            for hc, patterns in cls.HC_PATTERNS.items():
                if any(re.search(pat, combined) for pat in patterns):
                    hc_name = hc
                    break
            if not hc_name:
                court_level = "Legislation / General"

        # Legal Domain Inference
        domain = "General"
        if any(k in combined for k in ["constitution", "fundamental rights", "article 21", "article 32", "writ petition"]):
            domain = "Constitutional"
        elif any(k in combined for k in ["criminal", "bns", "bnss", "bsa", "ipc", "crpc", "bail", "murder", "theft", "penal code", "accused"]):
            domain = "Criminal"
        elif any(k in combined for k in ["cpc", "injunction", "suit", "plaintiff", "defendant", "specific relief"]):
            domain = "Civil"
        elif any(k in combined for k in ["companies act", "nclt", "ibc", "insolvency", "corporate", "director", "merger"]):
            domain = "Corporate / Commercial"
        elif any(k in combined for k in ["income tax", "gst", "itat", "excise", "customs", "taxation"]):
            domain = "Tax"
        elif any(k in combined for k in ["trademark", "patent", "copyright", "ipr", "infringement"]):
            domain = "Intellectual Property"
        elif any(k in combined for k in ["arbitration", "award", "section 34", "arbitrator", "conciliation"]):
            domain = "Arbitration"
        elif any(k in combined for k in ["pocso", "ndps", "pmla", "uapa", "special act"]):
            domain = "Special Acts"

        return court_level, hc_name, domain
