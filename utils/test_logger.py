from logger import save_alert


save_alert(
    crop="rice",
    stage="early",
    confidence=0.91,
    zone=2,
    risk="HIGH",
    response="ACTIVATE_DETERRENT"
)

print("Log saved successfully!")