#!/usr/bin/env python3
"""Disease recommendation mapping and lookup helpers."""

from __future__ import annotations


RECOMMENDATIONS = {
    "Tomato_Late_Blight": {
        "High": "Apply copper-based fungicide immediately. Remove and destroy infected leaves. Isolate the plant. Avoid overhead watering.",
        "Medium": "Apply preventive fungicide every 7 days. Remove lower infected leaves. Improve air circulation.",
        "Low": "Monitor every 3 days. Avoid wetting foliage. Ensure good drainage.",
    },
    "Tomato_Early_Blight": {
        "High": "Remove infected leaves. Apply mancozeb or chlorothalonil. Mulch around base to prevent soil splash.",
        "Medium": "Apply neem oil every 5-7 days. Remove lower infected leaves. Rotate crops next season.",
        "Low": "Monitor weekly. Avoid overhead irrigation. Keep foliage dry.",
    },
    "Tomato_Bacterial_Spot": {
        "High": "Apply copper bactericide + mancozeb. Remove infected tissues. Avoid working with wet plants.",
        "Medium": "Use copper spray every 7 days. Improve air circulation.",
        "Low": "Use preventive copper spray and monitor new growth.",
    },
    "Tomato_Fusarium": {
        "High": "Remove severely affected plants and improve drainage. Use rotation and resistant varieties.",
        "Medium": "Reduce plant stress and remove infected tissue.",
        "Low": "Monitor wilting symptoms and keep soil well-managed.",
    },
    "Tomato_Leaf_Curl": {
        "High": "Control whiteflies/aphids urgently and remove severely affected plants.",
        "Medium": "Use sticky traps and targeted vector control.",
        "Low": "Monitor vectors and sanitize field edges.",
    },
    "Tomato_Mosaic": {
        "High": "No chemical cure. Remove infected plants and disinfect tools.",
        "Medium": "Remove symptomatic leaves and minimize handling.",
        "Low": "Monitor spread and maintain hygiene.",
    },
    "Tomato_Septoria": {
        "High": "Remove infected leaves and apply copper/chlorothalonil fungicide.",
        "Medium": "Preventive copper spray and improve airflow.",
        "Low": "Monitor and avoid prolonged leaf wetness.",
    },
    "Tomato_Healthy": {
        "High": "Plant appears healthy. Continue regular monitoring.",
        "Medium": "Plant appears healthy. Continue regular monitoring.",
        "Low": "Plant appears healthy. Continue regular monitoring.",
    },
    "Pepper_Bacterial_Spot": {
        "High": "Apply copper bactericide immediately. Remove infected leaves/fruits and disinfect tools.",
        "Medium": "Copper spray every 7 days and remove infected tissue.",
        "Low": "Preventive copper application and monitor new growth.",
    },
    "Pepper_Cercospora": {
        "High": "Apply fungicide promptly, remove infected foliage, and improve canopy airflow.",
        "Medium": "Apply preventive spray and prune crowded foliage.",
        "Low": "Monitor symptoms and avoid overhead watering.",
    },
    "Pepper_Early_Blight": {
        "High": "Remove infected leaves and apply broad-spectrum fungicide. Reduce leaf wetness.",
        "Medium": "Use preventive spray and increase spacing.",
        "Low": "Monitor frequently and keep foliage dry.",
    },
    "Pepper_Fusarium": {
        "High": "Remove severely affected plants, improve drainage, and use rotation with resistant varieties.",
        "Medium": "Reduce irrigation stress and remove infected tissue.",
        "Low": "Monitor wilt symptoms and avoid overwatering.",
    },
    "Pepper_Late_Blight": {
        "High": "Apply fungicide immediately, remove infected tissue, and avoid overhead irrigation.",
        "Medium": "Use preventive fungicide and improve airflow.",
        "Low": "Monitor closely during humid weather.",
    },
    "Pepper_Leaf_Blight": {
        "High": "Apply fungicide and remove infected leaves. Sanitize tools between plants.",
        "Medium": "Preventive spray and better spacing.",
        "Low": "Monitor and keep foliage dry.",
    },
    "Pepper_Leaf_Curl": {
        "High": "No chemical cure; the virus is systemic. Remove and destroy severely infected plants immediately. Apply thiamethoxam or imidacloprid to control whiteflies (primary vector). Install yellow sticky traps and reflective silver mulch. Disinfect tools between plants.",
        "Medium": "Apply imidacloprid to reduce whitefly population every 7 days. Install yellow sticky traps around affected plants. Remove the most visibly infected leaves. Avoid field work during hot midday periods when whitefly activity is highest.",
        "Low": "Monitor whitefly activity daily, especially leaf undersides. Install yellow sticky traps for early warning. Apply neem oil on leaf undersides. Remove severely curled new growth promptly.",
    },
    "Pepper_Leaf_Mosaic": {
        "High": "No direct cure. Remove heavily infected plants and disinfect tools.",
        "Medium": "Remove symptomatic leaves and control vectors.",
        "Low": "Monitor spread and keep strict hygiene.",
    },
    "Pepper_Septoria": {
        "High": "Remove infected leaves and apply copper/chlorothalonil fungicide.",
        "Medium": "Preventive fungicide and improved airflow.",
        "Low": "Monitor and avoid prolonged leaf wetness.",
    },
    "Pepper_Healthy": {
        "High": "Plant appears healthy. Continue regular monitoring.",
        "Medium": "Plant appears healthy. Continue regular monitoring.",
        "Low": "Plant appears healthy. Continue regular monitoring.",
    },
    "Corn_Cercospora_Leaf_Spot": {
        "High": "Apply strobilurin fungicide (azoxystrobin or pyraclostrobin) immediately. Remove and destroy heavily infected leaves. Avoid overhead irrigation. Do not enter field while foliage is wet. Use resistant hybrids next season.",
        "Medium": "Apply fungicide at first sign of lesion spread. Improve field drainage and airflow. Remove lower infected leaves. Monitor every 3 days and avoid excessive nitrogen.",
        "Low": "Monitor every 3-4 days. Maintain adequate spacing for airflow. Apply preventive fungicide if humid weather is forecast. Avoid excessive nitrogen fertilization.",
    },
    "Corn_Common_Rust": {
        "High": "Apply triazole/strobilurin fungicide and use resistant hybrids next season.",
        "Medium": "Apply fungicide in humid conditions and monitor spread.",
        "Low": "Monitor weekly. Fungicide may not be needed now.",
    },
    "Corn_Northern_Leaf_Blight": {
        "High": "Apply propiconazole or azoxystrobin. Remove infected leaves.",
        "Medium": "Apply fungicide at early tassel stage and monitor weather.",
        "Low": "Monitor and consider preventive spray in wet conditions.",
    },
    "Corn_Streak": {
        "High": "Rogue severely infected plants and control insect vectors. Remove nearby host weeds.",
        "Medium": "Control vectors and monitor neighboring plants for spread.",
        "Low": "Continue scouting every few days and maintain field hygiene.",
    },
    "Corn_Healthy": {
        "High": "Plant appears healthy. Continue regular monitoring.",
        "Medium": "Plant appears healthy. Continue regular monitoring.",
        "Low": "Plant appears healthy. Continue regular monitoring.",
    },
}


def get_recommendation(disease_class: str, severity_level: str) -> str:
    """Get treatment recommendation for a predicted class + severity level."""
    normalized = disease_class.strip().replace(" ", "_").replace("-", "_")
    rec = RECOMMENDATIONS.get(normalized)

    if rec is None:
        for key in RECOMMENDATIONS:
            if key.lower() == normalized.lower():
                rec = RECOMMENDATIONS[key]
                break

    if rec is None:
        lower = normalized.lower()
        if "healthy" in lower:
            return "Plant appears healthy. Continue routine monitoring and preventive care."
        if "blight" in lower:
            return "Apply suitable fungicide, remove infected leaves, and reduce leaf wetness duration."
        if "rust" in lower:
            return "Monitor spread and apply fungicide if humidity remains high."
        if "mosaic" in lower or "virus" in lower or "curl" in lower:
            return "Focus on vector control and remove severely infected plants. Disinfect tools regularly."
        if "bacterial" in lower:
            return "Use copper-based bactericide, sanitize tools, and avoid handling wet plants."
        return (
            f"Disease '{disease_class}' is not in the recommendation database. "
            "Consult a local agricultural extension expert."
        )

    return rec.get(
        severity_level,
        "No specific recommendation for this severity level. Consult a local agriculture expert.",
    )


def list_supported_diseases() -> list[str]:
    return list(RECOMMENDATIONS.keys())
