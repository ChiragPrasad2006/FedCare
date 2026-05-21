"""
Privacy utilities for presentation-friendly patient record handling.
"""
from __future__ import annotations

import random
import re
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


SENSITIVE_FIELDS = {
    "name",
    "full_name",
    "patient_name",
    "email",
    "phone",
    "address",
    "ssn",
    "date_of_birth",
    "dob",
    "insurance_id",
    "insurance_member_id",
    "emergency_contact",
}

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

FIRST_NAMES = ["Ava", "Noah", "Mia", "Ethan", "Ivy", "Liam", "Sophia", "Lucas"]
LAST_NAMES = ["Patel", "Reed", "Morris", "Nguyen", "Turner", "Brooks", "Kim", "Ali"]
CONDITIONS = ["Diabetes", "Hypertension", "Asthma", "Cardiac Risk", "Routine Screening"]
DEPARTMENTS = ["Emergency", "Cardiology", "Radiology", "Oncology", "General Medicine"]
STATUSES = ["Ready for Training", "Queued for Review", "Synced with Main Server"]


def _mask_value(field_name: str, value):
    if value in (None, ""):
        return value

    if field_name in {"ssn"}:
        return "***-**-" + str(value)[-4:]
    if field_name in {"phone"}:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return "***-***-" + digits[-4:] if len(digits) >= 4 else "[REDACTED]"
    if field_name in {"email"}:
        return "[REDACTED_EMAIL]"
    if field_name in {"address"}:
        return "[REDACTED_ADDRESS]"
    if field_name in {"name", "full_name", "patient_name", "emergency_contact"}:
        return "[REDACTED_NAME]"
    if field_name in {"date_of_birth", "dob"}:
        return "[REDACTED_DOB]"
    if field_name in {"insurance_id", "insurance_member_id"}:
        return "[REDACTED_INSURANCE]"
    return "[REDACTED]"


def analyze_and_anonymize_records(records: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Anonymize patient records and return a summary useful for dashboards.
    """
    anonymized_records = []
    detected_items = 0
    redacted_items = 0
    residual_items = 0

    for record in records:
        safe_record = deepcopy(record)

        for key, value in list(safe_record.items()):
            lowered = key.lower()
            if lowered in SENSITIVE_FIELDS:
                detected_items += 1
                safe_record[key] = _mask_value(lowered, value)
                redacted_items += 1
                continue

            if isinstance(value, str):
                replaced_value = value
                field_had_pii = False
                for pattern in PII_PATTERNS.values():
                    matches = pattern.findall(replaced_value)
                    if matches:
                        detected_items += len(matches)
                        redacted_items += len(matches)
                        replaced_value = pattern.sub("[REDACTED]", replaced_value)
                        field_had_pii = True

                if field_had_pii:
                    safe_record[key] = replaced_value

        residual_items += count_residual_pii(safe_record)
        anonymized_records.append(safe_record)

    removal_accuracy = round(
        max(redacted_items - residual_items, 0) / detected_items * 100, 2
    ) if detected_items else 100.0

    summary = {
        "total_records": len(records),
        "detected_pii_items": detected_items,
        "redacted_items": redacted_items,
        "residual_risk_items": residual_items,
        "removal_accuracy": removal_accuracy,
        "processed_at": datetime.utcnow().isoformat() + "Z",
    }
    return anonymized_records, summary


def count_residual_pii(record: Dict) -> int:
    """
    Count remaining PII-like items after anonymization.
    """
    count = 0
    for key, value in record.items():
        lowered = key.lower()
        if lowered in SENSITIVE_FIELDS and isinstance(value, str) and "[REDACTED" not in value:
            count += 1
        if isinstance(value, str):
            for pattern in PII_PATTERNS.values():
                count += len(pattern.findall(value))
    return count


def generate_demo_patient_records(hospital_id: str, count: int = 8) -> List[Dict]:
    """
    Generate synthetic patient records for UI demos and presentations.
    """
    records = []
    today = datetime.utcnow().date()

    for index in range(count):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        age = random.randint(24, 79)
        dob = today - timedelta(days=age * 365 + random.randint(0, 364))
        patient_number = f"{hospital_id.upper()}-{1000 + index}"
        full_name = f"{first} {last}"

        records.append({
            "patient_id": patient_number,
            "name": full_name,
            "date_of_birth": dob.isoformat(),
            "email": f"{first.lower()}.{last.lower()}{index}@example.com",
            "phone": f"555-01{index:02d}",
            "address": f"{100 + index} Health Ave, Suite {index + 1}",
            "department": random.choice(DEPARTMENTS),
            "condition": random.choice(CONDITIONS),
            "risk_score": round(random.uniform(0.18, 0.94), 2),
            "status": random.choice(STATUSES),
            "notes": (
                f"Patient follow-up created for {full_name}. "
                f"Primary contact: {first.lower()}{index}@mail.com."
            ),
            "last_updated": datetime.utcnow().isoformat() + "Z",
        })

    return records
