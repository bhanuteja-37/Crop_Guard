from risk_engine import BirdRiskEngine

engine = BirdRiskEngine(
    crop="rice",
    growth_stage="early"
)

score, risk = engine.calculate_risk(
    confidence=0.91,
    repeated_detection=True,
    zone=2
)

print("Risk Score:", score)
print("Risk Level:", risk)