class BirdRiskEngine:

    def __init__(self, crop="rice", growth_stage="early"):
        self.crop = crop
        self.growth_stage = growth_stage

    def calculate_risk(
        self,
        confidence,
        repeated_detection,
        zone
    ):

        score = 0

        # Confidence
        if confidence >= 0.80:
            score += 40
        elif confidence >= 0.60:
            score += 20
        else:
            score += 5

        # Repeated detection
        if repeated_detection:
            score += 30

        # Crop
        if self.crop == "rice":
            score += 10
        elif self.crop == "maize":
            score += 10

        # Growth stage
        if self.growth_stage == "early":
            score += 20
        elif self.growth_stage == "maturity":
            score += 15
        else:
            score += 5

        # Risk level
        if score >= 70:
            risk = "HIGH"
        elif score >= 40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return score, risk