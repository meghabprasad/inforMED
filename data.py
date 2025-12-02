"""
Knowledge Base for Bayesian Diagnostic System

This module contains all the domain knowledge:
- Diagnosis classes
- Symptom questions
- Conditional Probability Table (CPT)
"""

# ============================================================================
# Diagnosis Classes (D_i)
# ============================================================================

DIAGNOSES = [
    "Migraine",
    "Tension-Type",
    "Cluster",
    "Sinus",
    "Medication Overuse",
    "Cervicogenic",
    "Trigeminal Neuralgia",
    "Hypertensive",
]
NUM_DIAGNOSES = len(DIAGNOSES)

# Uniform prior probability for the initial state
PRIOR_PROBABILITY = 1.0 / NUM_DIAGNOSES  # 0.125 (1/8)

# ============================================================================
# Symptom Questions (S_i)
# ============================================================================

SYMPTOMS = {
    # Pain characteristics
    "S1": "Is your pain best described as throbbing or pulsating?",
    "S2": "Is the pain located primarily on only one side of your head?",
    "S3": "Have you experienced nausea or vomiting with this headache?",
    "S4": "Does bright light or loud noise bother you during the headache?",
    "S5": "Would you rate the pain intensity as severe or disabling?",
    "S6": "Did you experience eye redness, tearing, or nasal congestion on the side of the pain?",
    "S7": "Does the pain typically last 4-72 hours if untreated?",
    "S8": "Does physical activity make the headache worse?",
    "S9": "Have you experienced visual disturbances (aura) before the headache?",
    "S10": "Is there a family history of similar headaches?",
    "S11": "Does the pain occur in brief attacks (15-180 minutes)?",
    "S12": "Do you feel restless or agitated during the headache?",
    "S13": "Does the pain always occur on the same side?",
    "S14": "Do you experience drooping eyelid or constricted pupil on the pain side?",
    "S15": "Is there facial pressure or fullness, especially around sinuses?",
    "S16": "Do you have thick nasal discharge (yellow or green)?",
    "S17": "Is the pain often described as a tight band around the head?",
    "S18": "Does the pain feel like steady pressure rather than pulsating?",
    "S19": "Can you continue daily activities despite the headache?",
    "S20": "Did the headache follow a recent illness (e.g., cold, flu)?",
    # Additional detailed symptoms (S21-S35)
    "S21": "Do you take pain medication more than 10-15 days per month?",
    "S22": "Have your headaches become more frequent over time?",
    "S23": "Does the pain worsen or start upon waking in the morning?",
    "S24": "Is the pain triggered by neck movement or sustained awkward postures?",
    "S25": "Do you have limited range of motion in your neck?",
    "S26": "Is the pain described as sharp, shooting, or electric shock-like?",
    "S27": "Does touching certain areas of your face trigger the pain?",
    "S28": "Does the pain last only seconds to 2 minutes per episode?",
    "S29": "Do you have high blood pressure or hypertension?",
    "S30": "Does the pain feel like pressure at the back of the head?",
    "S31": "Have you experienced dizziness or visual changes with the headache?",
    "S32": "Does the pain radiate from the neck to the front of the head?",
    "S33": "Is the headache present daily or nearly every day?",
    "S34": "Do you experience muscle tenderness in the neck or shoulders?",
    "S35": "Does the pain occur in multiple episodes throughout the day?",
    "S36": "Have you recently stopped or significantly reduced your caffeine intake?",
}

# ============================================================================
# Conditional Probability Table (CPT)
# P(S_i = Yes | D_k) - Probability of symptom given diagnosis
# ============================================================================

CPT = {
    "Migraine": {
        "S1": 0.90,  # Throbbing/pulsating - very characteristic of migraine
        "S2": 0.75,  # Unilateral - common in migraine
        "S3": 0.85,  # Nausea/vomiting - very common
        "S4": 0.90,  # Photophobia/phonophobia - hallmark of migraine
        "S5": 0.80,  # Severe intensity - typically disabling
        "S6": 0.10,  # Autonomic symptoms - rare in migraine
        "S7": 0.85,  # Duration 4-72 hours - diagnostic criterion
        "S8": 0.85,  # Worsened by activity - typical
        "S9": 0.30,  # Aura - present in ~25-30% of migraines
        "S10": 0.70,  # Family history - strong genetic component
        "S11": 0.05,  # Brief attacks - not typical
        "S12": 0.10,  # Restlessness - not characteristic
        "S13": 0.40,  # Always same side - less common
        "S14": 0.05,  # Horner's syndrome - rare
        "S15": 0.15,  # Sinus pressure - sometimes reported
        "S16": 0.05,  # Nasal discharge - rare
        "S17": 0.10,  # Band-like pain - not typical
        "S18": 0.15,  # Steady pressure - less common
        "S19": 0.20,  # Can continue activities - usually can't
        "S20": 0.15,  # Recent illness - not strongly associated
        "S21": 0.40,  # Medication overuse - possible in chronic migraine
        "S22": 0.50,  # Increasing frequency - can occur
        "S23": 0.30,  # Morning pain - sometimes
        "S24": 0.15,  # Neck movement trigger - not typical
        "S25": 0.20,  # Limited neck ROM - not characteristic
        "S26": 0.05,  # Sharp/shooting - not typical
        "S27": 0.05,  # Trigger points - not typical
        "S28": 0.02,  # Seconds duration - not typical
        "S29": 0.15,  # Hypertension - not associated
        "S30": 0.20,  # Back of head - less common
        "S31": 0.35,  # Dizziness/visual - can occur with aura
        "S32": 0.10,  # Neck radiation - not typical
        "S33": 0.25,  # Daily headaches - in chronic migraine
        "S34": 0.30,  # Muscle tenderness - sometimes
        "S35": 0.15,  # Multiple episodes - not typical pattern
        "S36": 0.40,  # Caffeine withdrawal - can trigger migraine
    },
    "Tension-Type": {
        "S1": 0.10,  # Throbbing - not typical
        "S2": 0.20,  # Unilateral - usually bilateral
        "S3": 0.10,  # Nausea/vomiting - rare
        "S4": 0.20,  # Light/sound sensitivity - mild if present
        "S5": 0.30,  # Severe intensity - usually mild to moderate
        "S6": 0.05,  # Autonomic symptoms - very rare
        "S7": 0.40,  # 4-72 hour duration - variable
        "S8": 0.20,  # Worsened by activity - not typical
        "S9": 0.02,  # Aura - extremely rare
        "S10": 0.30,  # Family history - some genetic component
        "S11": 0.05,  # Brief attacks - not typical
        "S12": 0.10,  # Restlessness - not characteristic
        "S13": 0.10,  # Always same side - rare
        "S14": 0.02,  # Horner's syndrome - not seen
        "S15": 0.10,  # Sinus pressure - sometimes reported
        "S16": 0.05,  # Nasal discharge - not typical
        "S17": 0.90,  # Band-like pain - HALLMARK symptom
        "S18": 0.85,  # Steady pressure - very characteristic
        "S19": 0.70,  # Can continue activities - usually can
        "S20": 0.10,  # Recent illness - not strongly associated
        "S21": 0.35,  # Medication overuse - possible
        "S22": 0.40,  # Increasing frequency - can occur
        "S23": 0.40,  # Morning pain - can occur
        "S24": 0.30,  # Neck movement trigger - sometimes
        "S25": 0.25,  # Limited neck ROM - sometimes
        "S26": 0.05,  # Sharp/shooting - not typical
        "S27": 0.15,  # Trigger points - sometimes
        "S28": 0.02,  # Seconds duration - not typical
        "S29": 0.15,  # Hypertension - not associated
        "S30": 0.40,  # Back of head - can occur
        "S31": 0.15,  # Dizziness/visual - rare
        "S32": 0.20,  # Neck radiation - sometimes
        "S33": 0.45,  # Daily headaches - common in chronic form
        "S34": 0.70,  # Muscle tenderness - very common
        "S35": 0.10,  # Multiple episodes - not typical pattern
        "S36": 0.50,  # Caffeine withdrawal - commonly triggers tension headache
    },
    "Cluster": {
        "S1": 0.40,  # Throbbing - sometimes present
        "S2": 0.95,  # Unilateral - almost always
        "S3": 0.15,  # Nausea/vomiting - less common
        "S4": 0.30,  # Light/sound sensitivity - mild
        "S5": 0.95,  # Severe intensity - EXTREMELY severe
        "S6": 0.95,  # Autonomic symptoms - HALLMARK of cluster
        "S7": 0.10,  # 4-72 hours - too long for cluster
        "S8": 0.30,  # Worsened by activity - not typical
        "S9": 0.05,  # Aura - not typical
        "S10": 0.15,  # Family history - less genetic than migraine
        "S11": 0.95,  # Brief attacks 15-180 min - DIAGNOSTIC
        "S12": 0.90,  # Restlessness/agitation - very characteristic
        "S13": 0.85,  # Always same side - very typical
        "S14": 0.60,  # Horner's syndrome - common
        "S15": 0.20,  # Sinus pressure - sometimes reported
        "S16": 0.10,  # Nasal discharge - nasal congestion yes, discharge less
        "S17": 0.05,  # Band-like pain - not typical
        "S18": 0.10,  # Steady pressure - pain is severe and sharp
        "S19": 0.05,  # Can continue activities - impossible during attack
        "S20": 0.05,  # Recent illness - not associated
        "S21": 0.10,  # Medication overuse - not typical
        "S22": 0.30,  # Increasing frequency - can occur in clusters
        "S23": 0.40,  # Morning pain - can occur at specific times
        "S24": 0.10,  # Neck movement trigger - not typical
        "S25": 0.10,  # Limited neck ROM - not typical
        "S26": 0.30,  # Sharp/shooting - can be described this way
        "S27": 0.10,  # Trigger points - not typical
        "S28": 0.05,  # Seconds duration - too brief
        "S29": 0.15,  # Hypertension - not associated
        "S30": 0.10,  # Back of head - not typical location
        "S31": 0.20,  # Dizziness/visual - sometimes
        "S32": 0.10,  # Neck radiation - not typical
        "S33": 0.40,  # Daily headaches - during cluster periods
        "S34": 0.20,  # Muscle tenderness - not typical
        "S35": 0.85,  # Multiple episodes - very characteristic
        "S36": 0.10,  # Caffeine withdrawal - not associated with cluster
    },
    "Sinus": {
        "S1": 0.30,  # Throbbing - sometimes present
        "S2": 0.50,  # Unilateral - can be one-sided
        "S3": 0.10,  # Nausea/vomiting - rare
        "S4": 0.15,  # Light/sound sensitivity - not typical
        "S5": 0.40,  # Severe intensity - usually moderate
        "S6": 0.30,  # Autonomic symptoms - nasal congestion common
        "S7": 0.50,  # 4-72 hours - variable duration
        "S8": 0.40,  # Worsened by activity - can worsen with bending
        "S9": 0.02,  # Aura - not typical
        "S10": 0.10,  # Family history - less genetic
        "S11": 0.05,  # Brief attacks - usually longer
        "S12": 0.05,  # Restlessness - not typical
        "S13": 0.40,  # Always same side - can be consistent
        "S14": 0.02,  # Horner's syndrome - not seen
        "S15": 0.95,  # Sinus pressure - HALLMARK symptom
        "S16": 0.85,  # Nasal discharge - very characteristic
        "S17": 0.10,  # Band-like pain - not typical
        "S18": 0.60,  # Steady pressure - often pressure-like
        "S19": 0.50,  # Can continue activities - variable
        "S20": 0.80,  # Recent illness - strongly associated
        "S21": 0.15,  # Medication overuse - not typical
        "S22": 0.25,  # Increasing frequency - not typical
        "S23": 0.60,  # Morning pain - worse in morning
        "S24": 0.20,  # Neck movement trigger - not typical
        "S25": 0.15,  # Limited neck ROM - not typical
        "S26": 0.10,  # Sharp/shooting - not typical
        "S27": 0.15,  # Trigger points - facial tenderness sometimes
        "S28": 0.02,  # Seconds duration - not typical
        "S29": 0.15,  # Hypertension - not associated
        "S30": 0.20,  # Back of head - not typical location
        "S31": 0.25,  # Dizziness/visual - sometimes
        "S32": 0.10,  # Neck radiation - not typical
        "S33": 0.30,  # Daily headaches - during acute sinusitis
        "S34": 0.20,  # Muscle tenderness - not typical
        "S35": 0.10,  # Multiple episodes - not typical pattern
        "S36": 0.15,  # Caffeine withdrawal - not typically associated
    },
    "Medication Overuse": {
        "S1": 0.50,  # Throbbing - can occur
        "S2": 0.50,  # Unilateral - variable
        "S3": 0.40,  # Nausea/vomiting - can occur
        "S4": 0.50,  # Light/sound sensitivity - can occur
        "S5": 0.60,  # Severe intensity - often moderate to severe
        "S6": 0.10,  # Autonomic symptoms - not typical
        "S7": 0.60,  # 4-72 hours - can be prolonged
        "S8": 0.40,  # Worsened by activity - variable
        "S9": 0.10,  # Aura - not typical
        "S10": 0.40,  # Family history - underlying headache disorder
        "S11": 0.05,  # Brief attacks - usually longer
        "S12": 0.20,  # Restlessness - sometimes irritability
        "S13": 0.30,  # Always same side - variable
        "S14": 0.02,  # Horner's syndrome - not seen
        "S15": 0.20,  # Sinus pressure - sometimes
        "S16": 0.05,  # Nasal discharge - not typical
        "S17": 0.40,  # Band-like pain - can occur
        "S18": 0.50,  # Steady pressure - often dull pressure
        "S19": 0.40,  # Can continue activities - variable impairment
        "S20": 0.10,  # Recent illness - not associated
        "S21": 0.95,  # Medication overuse - DIAGNOSTIC CRITERION
        "S22": 0.90,  # Increasing frequency - HALLMARK symptom
        "S23": 0.85,  # Morning pain - very characteristic
        "S24": 0.15,  # Neck movement trigger - not typical
        "S25": 0.20,  # Limited neck ROM - not typical
        "S26": 0.10,  # Sharp/shooting - not typical
        "S27": 0.15,  # Trigger points - not typical
        "S28": 0.02,  # Seconds duration - not typical
        "S29": 0.15,  # Hypertension - not associated
        "S30": 0.30,  # Back of head - can occur
        "S31": 0.30,  # Dizziness/visual - sometimes
        "S32": 0.20,  # Neck radiation - sometimes
        "S33": 0.90,  # Daily headaches - VERY characteristic
        "S34": 0.40,  # Muscle tenderness - sometimes
        "S35": 0.30,  # Multiple episodes - continuous rather than episodic
        "S36": 0.60,  # Caffeine withdrawal - caffeine overuse common in MOH
    },
    "Cervicogenic": {
        "S1": 0.20,  # Throbbing - not typical
        "S2": 0.80,  # Unilateral - typically one-sided
        "S3": 0.20,  # Nausea/vomiting - sometimes
        "S4": 0.30,  # Light/sound sensitivity - mild if present
        "S5": 0.50,  # Severe intensity - moderate to severe
        "S6": 0.15,  # Autonomic symptoms - rare
        "S7": 0.60,  # 4-72 hours - variable duration
        "S8": 0.40,  # Worsened by activity - not primary feature
        "S9": 0.05,  # Aura - not typical
        "S10": 0.15,  # Family history - not genetic
        "S11": 0.10,  # Brief attacks - usually longer
        "S12": 0.10,  # Restlessness - not characteristic
        "S13": 0.75,  # Always same side - very typical
        "S14": 0.05,  # Horner's syndrome - rare
        "S15": 0.15,  # Sinus pressure - not typical
        "S16": 0.05,  # Nasal discharge - not typical
        "S17": 0.30,  # Band-like pain - sometimes
        "S18": 0.60,  # Steady pressure - often pressure-like
        "S19": 0.40,  # Can continue activities - variable
        "S20": 0.10,  # Recent illness - not associated
        "S21": 0.30,  # Medication overuse - possible comorbidity
        "S22": 0.40,  # Increasing frequency - can occur
        "S23": 0.60,  # Morning pain - often worse in morning
        "S24": 0.90,  # Neck movement trigger - HALLMARK symptom
        "S25": 0.85,  # Limited neck ROM - very characteristic
        "S26": 0.25,  # Sharp/shooting - sometimes
        "S27": 0.40,  # Trigger points - neck tenderness common
        "S28": 0.02,  # Seconds duration - not typical
        "S29": 0.15,  # Hypertension - not associated
        "S30": 0.70,  # Back of head - very characteristic origin
        "S31": 0.35,  # Dizziness/visual - sometimes
        "S32": 0.90,  # Neck radiation - HALLMARK symptom
        "S33": 0.50,  # Daily headaches - can be chronic
        "S34": 0.90,  # Muscle tenderness - very characteristic
        "S35": 0.20,  # Multiple episodes - usually sustained
        "S36": 0.20,  # Caffeine withdrawal - not typically associated
    },
    "Trigeminal Neuralgia": {
        "S1": 0.05,  # Throbbing - not typical
        "S2": 0.95,  # Unilateral - almost always one-sided
        "S3": 0.05,  # Nausea/vomiting - not typical
        "S4": 0.10,  # Light/sound sensitivity - not typical
        "S5": 0.98,  # Severe intensity - EXTREMELY severe
        "S6": 0.20,  # Autonomic symptoms - sometimes tearing
        "S7": 0.02,  # 4-72 hours - too long
        "S8": 0.10,  # Worsened by activity - not typical
        "S9": 0.02,  # Aura - not typical
        "S10": 0.10,  # Family history - not strongly genetic
        "S11": 0.05,  # Brief attacks 15-180 min - too long
        "S12": 0.30,  # Restlessness - fear of triggering pain
        "S13": 0.90,  # Always same side - very typical
        "S14": 0.02,  # Horner's syndrome - not typical
        "S15": 0.10,  # Sinus pressure - not typical
        "S16": 0.05,  # Nasal discharge - not typical
        "S17": 0.05,  # Band-like pain - not typical
        "S18": 0.05,  # Steady pressure - not typical
        "S19": 0.10,  # Can continue activities - very disabling
        "S20": 0.05,  # Recent illness - not associated
        "S21": 0.15,  # Medication overuse - not typical
        "S22": 0.40,  # Increasing frequency - can worsen over time
        "S23": 0.20,  # Morning pain - no time pattern
        "S24": 0.15,  # Neck movement trigger - not primary
        "S25": 0.10,  # Limited neck ROM - not typical
        "S26": 0.98,  # Sharp/shooting - HALLMARK electric shock pain
        "S27": 0.95,  # Trigger points - HALLMARK feature
        "S28": 0.95,  # Seconds duration - DIAGNOSTIC (seconds to 2 min)
        "S29": 0.15,  # Hypertension - not associated
        "S30": 0.10,  # Back of head - not typical location
        "S31": 0.15,  # Dizziness/visual - not typical
        "S32": 0.10,  # Neck radiation - not typical
        "S33": 0.30,  # Daily headaches - multiple attacks per day
        "S34": 0.15,  # Muscle tenderness - not typical
        "S35": 0.90,  # Multiple episodes - VERY characteristic
        "S36": 0.10,  # Caffeine withdrawal - not associated
    },
    "Hypertensive": {
        "S1": 0.40,  # Throbbing - sometimes pulsating
        "S2": 0.30,  # Unilateral - usually bilateral
        "S3": 0.30,  # Nausea/vomiting - sometimes
        "S4": 0.25,  # Light/sound sensitivity - mild if present
        "S5": 0.60,  # Severe intensity - can be severe
        "S6": 0.15,  # Autonomic symptoms - sometimes flushing
        "S7": 0.40,  # 4-72 hours - variable
        "S8": 0.50,  # Worsened by activity - can worsen
        "S9": 0.10,  # Aura - sometimes visual changes
        "S10": 0.25,  # Family history - hypertension may be familial
        "S11": 0.10,  # Brief attacks - usually sustained
        "S12": 0.20,  # Restlessness - sometimes anxiety
        "S13": 0.25,  # Always same side - not typical
        "S14": 0.02,  # Horner's syndrome - not typical
        "S15": 0.15,  # Sinus pressure - not typical
        "S16": 0.05,  # Nasal discharge - not typical
        "S17": 0.30,  # Band-like pain - sometimes
        "S18": 0.60,  # Steady pressure - often pressure-like
        "S19": 0.40,  # Can continue activities - variable
        "S20": 0.10,  # Recent illness - not typically associated
        "S21": 0.20,  # Medication overuse - not typical
        "S22": 0.50,  # Increasing frequency - can increase with BP
        "S23": 0.70,  # Morning pain - characteristic morning headache
        "S24": 0.20,  # Neck movement trigger - not typical
        "S25": 0.20,  # Limited neck ROM - not typical
        "S26": 0.15,  # Sharp/shooting - not typical
        "S27": 0.10,  # Trigger points - not typical
        "S28": 0.02,  # Seconds duration - not typical
        "S29": 0.95,  # Hypertension - DIAGNOSTIC feature
        "S30": 0.75,  # Back of head - VERY characteristic location
        "S31": 0.70,  # Dizziness/visual - very common
        "S32": 0.30,  # Neck radiation - sometimes
        "S33": 0.60,  # Daily headaches - can be persistent
        "S34": 0.30,  # Muscle tenderness - not typical
        "S35": 0.25,  # Multiple episodes - variable pattern
        "S36": 0.25,  # Caffeine withdrawal - not strongly associated
    },
}
