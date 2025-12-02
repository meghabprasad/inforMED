"""
informed - Intelligent Headache Diagnosis Using Information Theory

Interactive diagnostic system with real-time visualizations of:
- Probability distributions
- Information gain for candidate questions
- Entropy reduction over time
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from data import DIAGNOSES, SYMPTOMS, PRIOR_PROBABILITY, NUM_DIAGNOSES, CPT
from engine import (
    calculate_entropy,
    bayesian_update,
    get_next_question,
    calculate_information_gain,
)

# Page configuration
st.set_page_config(
    page_title="informed - Intelligent Diagnosis",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS - started with initial styling from Streamlit theme and then customized
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #1f77b4;
        font-weight: 400;
    }
    .med-part {
        text-decoration: underline;
        text-decoration-color: #e74c3c;
        text-decoration-thickness: 2px;
        text-underline-offset: 4px;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
def init_session():
    """Initialize or reset the diagnostic session"""
    st.session_state.current_P = [PRIOR_PROBABILITY] * NUM_DIAGNOSES
    st.session_state.asked_symptoms = set()
    st.session_state.entropy_history = [3.0]
    st.session_state.question_count = 0
    st.session_state.answers_history = []


if "current_P" not in st.session_state:
    init_session()


# Header with Logo
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{}' style='max-width: 100px; height: auto; display: block; margin-left: auto; margin-right: auto;'>
        <div class="main-header">infor<span class="med-part">med</span></div>
    </div>
    """.format(
        __import__("base64")
        .b64encode(open("informed_logo_2.png", "rb").read())
        .decode()
    ),
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Bayesian Diagnosis Engine • Information Theory • CS 109</div>',
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Diagnosis", "More Insights", "How This Works", "Sources & Appendix"]
)

# Current State Info (appears under tabs, above content)
with tab1:
    # Current state metrics at the top
    max_prob = max(st.session_state.current_P)
    best_idx = st.session_state.current_P.index(max_prob)
    best_diagnosis = DIAGNOSES[best_idx]
    current_entropy = calculate_entropy(st.session_state.current_P)

    # State info bar
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        st.metric("Questions", st.session_state.question_count)
    with col2:
        st.metric("Entropy", f"{current_entropy:.3f} bits")
    with col3:
        st.metric("Leading Diagnosis", best_diagnosis)
    with col4:
        st.metric("Confidence", f"{max_prob*100:.1f}%")
    with col5:
        if st.button("Reset", use_container_width=True):
            init_session()
            st.rerun()

    # High confidence indicator
    if max_prob >= 0.90:
        st.success(f"High Confidence: {best_diagnosis} ({max_prob*100:.0f}%)")

    st.markdown("---")

    # Diagnostic Session (full width)
    st.header("Diagnostic Session")

    # Get next question
    question_text, question_key, ig_value = get_next_question(
        st.session_state.current_P, st.session_state.asked_symptoms
    )

    if question_text is None:
        st.success("### ✓ Diagnosis Complete")
        st.info(
            f"**Final Diagnosis: {best_diagnosis}** with {max_prob*100:.1f}% confidence"
        )
        st.balloons()  # streamlit effect
    else:
        # Check if confidence is high enough to stop
        if max_prob >= 0.90:
            st.success("### ✓ Diagnosis Complete")
            st.markdown(f"""
            **{best_diagnosis}** — {max_prob*100:.1f}% confidence

            Reached diagnostic threshold (≥90%) in **{st.session_state.question_count} questions**.
            """)

            # Show current probabilities
            cols = st.columns(4)
            for i, (diagnosis, prob) in enumerate(
                zip(DIAGNOSES, st.session_state.current_P)
            ):
                with cols[i % 4]:
                    st.metric(diagnosis, f"{prob*100:.1f}%")
            st.balloons()  # streamlit effect
            st.markdown("---")

            # Option to continue anyway (could be moved to a helper do it if there is time)
            with st.expander("⚙️ Continue Asking Questions"):
                st.caption(f"Next question has {ig_value:.3f} bits of information gain")
                st.markdown(f"**{question_text}**")

                # Answer buttons inside expander
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "YES (Continue)",
                        use_container_width=True,
                        key="yes_continue",
                    ):
                        new_P = bayesian_update(
                            st.session_state.current_P,
                            question_key,
                            answer_is_yes=True,
                        )
                        st.session_state.current_P = new_P
                        st.session_state.asked_symptoms.add(question_key)
                        st.session_state.entropy_history.append(
                            calculate_entropy(new_P)
                        )
                        st.session_state.question_count += 1
                        st.session_state.answers_history.append(
                            {
                                "question": question_text,
                                "answer": "yes",
                                "key": question_key,
                            }
                        )
                        st.rerun()

                with col2:
                    if st.button(
                        "NO (Continue)", use_container_width=True, key="no_continue"
                    ):
                        new_P = bayesian_update(
                            st.session_state.current_P,
                            question_key,
                            answer_is_yes=False,
                        )
                        st.session_state.current_P = new_P
                        st.session_state.asked_symptoms.add(question_key)
                        st.session_state.entropy_history.append(
                            calculate_entropy(new_P)
                        )
                        st.session_state.question_count += 1
                        st.session_state.answers_history.append(
                            {
                                "question": question_text,
                                "answer": "no",
                                "key": question_key,
                            }
                        )
                        st.rerun()

        # NORMAL FLOW - Continue asking questions
        else:
            st.markdown(f"### Question {st.session_state.question_count + 1}")
            st.caption(f"Information Gain: {ig_value:.3f} bits")

            # Display the question
            st.markdown(f"**{question_text}**")

            # Answer buttons
            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "YES", use_container_width=True, type="primary", key="yes_main"
                ):
                    # Update with YES answer
                    new_P = bayesian_update(
                        st.session_state.current_P, question_key, answer_is_yes=True
                    )
                    st.session_state.current_P = new_P
                    st.session_state.asked_symptoms.add(question_key)
                    st.session_state.entropy_history.append(calculate_entropy(new_P))
                    st.session_state.question_count += 1
                    st.session_state.answers_history.append(
                        {
                            "question": question_text,
                            "answer": "yes",
                            "key": question_key,
                        }
                    )
                    st.rerun()

            with col2:
                if st.button(
                    "NO", use_container_width=True, type="secondary", key="no_main"
                ):
                    # Update with NO answer
                    new_P = bayesian_update(
                        st.session_state.current_P,
                        question_key,
                        answer_is_yes=False,
                    )
                    st.session_state.current_P = new_P
                    st.session_state.asked_symptoms.add(question_key)
                    st.session_state.entropy_history.append(calculate_entropy(new_P))
                    st.session_state.question_count += 1
                    st.session_state.answers_history.append(
                        {
                            "question": question_text,
                            "answer": "no",
                            "key": question_key,
                        }
                    )
                    st.rerun()

    # Question History
    st.markdown("---")
    if st.session_state.answers_history:
        with st.expander(
            f"Question History ({len(st.session_state.answers_history)} questions)",
            expanded=False,
        ):
            for i, ans in enumerate(st.session_state.answers_history):
                st.caption(f"Q{i+1}: {ans['answer'].upper()}")
                st.text(ans["question"])
                if i < len(st.session_state.answers_history) - 1:
                    st.markdown("---")

    # Live Visualizations (in a row below diagnostic session)
    st.markdown("---")
    st.header("Real-Time Analytics")

    viz_col1, viz_col2, viz_col3 = st.columns(3)

    with viz_col1:
        # 1. PROBABILITY CHART
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame(
            {"Diagnosis": DIAGNOSES, "Probability": st.session_state.current_P}
        )

        fig_prob = px.bar(
            prob_df,
            x="Diagnosis",
            y="Probability",
            color="Probability",
            color_continuous_scale="Blues",
            text=prob_df["Probability"].apply(lambda x: f"{x*100:.1f}%"),
        )
        fig_prob.update_traces(textposition="outside")
        fig_prob.update_layout(
            yaxis_range=[0, 1],
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with viz_col2:
        # 2. ENTROPY TIMELINE
        st.subheader("Uncertainty Reduction")
        st.caption("Entropy (bits)")

        entropy_df = pd.DataFrame(
            {
                "Question Number": range(len(st.session_state.entropy_history)),
                "Entropy (bits)": st.session_state.entropy_history,
            }
        )

        fig_entropy = px.line(
            entropy_df, x="Question Number", y="Entropy (bits)", markers=True
        )
        fig_entropy.add_hline(
            y=0,
            line_dash="dash",
            line_color="green",
            annotation_text="Complete Certainty",
            annotation_position="right",
        )
        fig_entropy.add_hline(
            y=3,
            line_dash="dash",
            line_color="red",
            annotation_text="Maximum Uncertainty",
            annotation_position="right",
        )
        fig_entropy.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_entropy, use_container_width=True)

    with viz_col3:
        # 3. INFORMATION GAIN CHART
        st.subheader("Question Rankings")
        st.caption("Top unasked by information gain")

        unasked_keys = [
            k for k in SYMPTOMS.keys() if k not in st.session_state.asked_symptoms
        ]

        if unasked_keys:
            ig_data = []
            for key in unasked_keys:
                ig = calculate_information_gain(st.session_state.current_P, key)
                ig_data.append(
                    {
                        "Key": key,
                        "Question": SYMPTOMS[key][:50] + "..."
                        if len(SYMPTOMS[key]) > 50
                        else SYMPTOMS[key],
                        "Information Gain": ig,
                    }
                )

            ig_df = pd.DataFrame(ig_data).sort_values(
                "Information Gain", ascending=False
            )

            # Show top 10
            fig_ig = px.bar(
                ig_df.head(10),
                y="Key",
                x="Information Gain",
                orientation="h",
                color="Information Gain",
                color_continuous_scale="Viridis",
                hover_data=["Question"],
            )
            fig_ig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_ig, use_container_width=True)
        else:
            st.info("All questions have been asked!")

with tab2:
    # Insights Tab
    # Note: The data shown below was computed using analysis.py
    # Run `python analysis.py` to see the full computational process
    st.header("Insights: Beyond the Basics")

    st.markdown("""
    We conducted three analyses to understand how information theory actually works in practice.
    Each analysis answers a specific question about our diagnostic system.
    """)

    # Analysis 1: Mutual Information Landscape
    st.subheader("1. Which Symptoms Are Most Informative?")

    st.markdown("""
    **The Question:** Are some symptoms *always* more informative than others,
    regardless of what we already know?

    **What We Did:** We calculated information gain for all 36 symptoms starting from
    the beginning (when all 8 diagnoses are equally likely at 12.5% each).

    **Remember:** Information gain measures how much a question reduces our uncertainty.
    Higher bits = more informative
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top-5 Most Informative Symptoms**")
        top_symptoms = pd.DataFrame(
            {
                "Symptom": [
                    "Pain lasts seconds (S28)",
                    "Brief attacks 15-180 min (S11)",
                    "Shock-like pain (S26)",
                    "Multiple episodes (S35)",
                    "Facial triggers (S27)",
                ],
                "MI (bits)": [0.407, 0.337, 0.313, 0.309, 0.282],
            }
        )
        st.dataframe(top_symptoms, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Bottom-5 Least Informative Symptoms**")
        bottom_symptoms = pd.DataFrame(
            {
                "Symptom": [
                    "Aura (S9)",
                    "Dizziness (S31)",
                    "Increasing frequency (S22)",
                    "Family history (S10)",
                    "Morning pain (S23)",
                ],
                "MI (bits)": [0.059, 0.091, 0.110, 0.127, 0.128],
            }
        )
        st.dataframe(bottom_symptoms, use_container_width=True, hide_index=True)

    st.success("""
    **The Insight:** Questions about *when and how long* the pain lasts are 3-4× more informative
    than questions about *where* it hurts! Cool! 🤯

    This is surprising! You'd think location matters most, but timing patterns are actually
    the important factor to distinguishing headache types.
    """)

    # Create bar chart
    all_mi_data = {
        "S28": 0.407,
        "S11": 0.337,
        "S26": 0.313,
        "S35": 0.309,
        "S27": 0.282,
        "S16": 0.276,
        "S6": 0.270,
        "S17": 0.259,
        "S13": 0.251,
        "S18": 0.250,
        "S9": 0.059,
        "S31": 0.091,
        "S22": 0.110,
        "S10": 0.127,
        "S23": 0.128,
    }

    mi_df = pd.DataFrame(
        list(all_mi_data.items()), columns=["Symptom", "Mutual Information (bits)"]
    )
    mi_df = mi_df.sort_values("Mutual Information (bits)", ascending=True)

    fig_mi = px.bar(
        mi_df,
        x="Mutual Information (bits)",
        y="Symptom",
        orientation="h",
        color="Mutual Information (bits)",
        color_continuous_scale="Viridis",
        title="Mutual Information Across Key Symptoms",
    )
    fig_mi.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_mi, use_container_width=True)

    st.markdown("---")

    # Analysis 2: Conditional Entropy Chains
    st.subheader("2. Do Later Questions Become More Powerful?")

    st.markdown("""
    **The Question:** Does each question reduce entropy by the same amount, or does it change
    as we learn more?

    **What We Did:** We traced through an actual diagnostic session (diagnosing migraine)
    and tracked how much each question reduced entropy.
    """)

    # Migraine case study
    st.markdown("**The Data:**")

    entropy_chain_data = pd.DataFrame(
        {
            "Question #": [1, 2, 3, 7, 8],
            "Entropy Before": [3.000, 2.849, 2.682, 2.066, 1.152],
            "Entropy After": [2.849, 2.682, 2.559, 1.152, 0.622],
            "Reduction (bits)": [0.151, 0.167, 0.123, 0.914, 0.530],
        }
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(entropy_chain_data, use_container_width=True, hide_index=True)

    with col2:
        fig_chain = px.line(
            entropy_chain_data,
            x="Question #",
            y="Reduction (bits)",
            markers=True,
            title="Entropy Reduction Accelerates",
        )
        fig_chain.update_layout(height=250)
        st.plotly_chart(fig_chain, use_container_width=True)

    st.success("""
    **What We Found:** Look at the "Reduction" column!
    - Early questions (Q1-Q3): only ~0.15 bits each
    - Later questions (Q7-Q8): 0.5-0.9 bits each — that's **5× more powerful!**
    """)

    st.info("""
    Information doesn't just add up—it *compounds*! Each answer makes the next
    question more powerful. Like narrowing down suspects: the fewer remain, the more each clue matters.
    """)

    st.markdown("---")

    # Analysis 3: Baseline Comparisons
    st.subheader("3. Does Information Gain Beat Simpler Approaches?")

    st.markdown("""
    **The Question:** Our algorithm uses information gain to pick questions. But maybe simpler
    strategies work just as well?

    **What We Compared:**
    1. **Random ordering:** Pick questions randomly (no strategy)
    2. **Frequency-based:** Ask about common symptoms first (intuitive—"start with what's most likely")
    3. **Information gain (ours):** Pick questions that reduce entropy most

    **The Test:** We simulated diagnosing 4 different headache types and counted questions needed
    to reach 90% confidence.
    """)

    baseline_data = pd.DataFrame(
        {
            "Strategy": [
                "Random ordering",
                "Frequency-based\n(ask common symptoms first)",
                "Information Gain\n(ours)",
            ],
            "Avg Questions": [11.4, 16.8, 5.2],
            "Efficiency": ["100% (baseline)", "68%", "219%"],
        }
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(baseline_data, use_container_width=True, hide_index=True)

    with col2:
        fig_baseline = px.bar(
            baseline_data,
            x="Strategy",
            y="Avg Questions",
            color="Avg Questions",
            color_continuous_scale="RdYlGn_r",
            title="Average Questions to 90% Confidence",
        )
        fig_baseline.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_baseline, use_container_width=True)

    st.success("""
    **What We Found:**
    - Our information gain approach: **5.2 questions** on average
    - Random ordering: **11.4 questions** (more than 2× slower!)
    - Frequency-based: **16.8 questions** (more than 3× slower!)

    Information gain is **2.2× faster than random**!
    """)

    st.warning("""
    Frequency-based ordering (asking about common symptoms first) is actually
    *worse* than random!

    **Why?** Because common symptoms appear in many diagnoses, so they don't help distinguish between them.

    **Example:**
    - "Does your head hurt?" → 100% for all types → gives us ZERO information
    - "Does pain last only seconds?" → Rare but specific to one type → highly informative!

    **The Insight:** Your intuition says "ask about common things first," but information theory
    says "ask about *distinctive* things first." Pretty cool :) 
    """)

with tab3:
    # Educational content
    st.header("How This System Works")

    st.markdown("""
    ### Core Algorithm

    #### Bayes' Rule
    Updates diagnosis probabilities after each answer:

    ```
    P(Diagnosis | Symptom) = P(Symptom | Diagnosis) × P(Diagnosis) / P(Symptom)
    ```

    #### Entropy
    Measures diagnostic uncertainty in bits:

    ```
    H = -Σ P(i) × log₂(P(i))
    ```

    - **3.0 bits**: Maximum (8 equally likely diagnoses)
    - **0.0 bits**: Complete certainty

    #### Information Gain
    Selects questions that reduce uncertainty most:

    ```
    IG = H(current) - E[H(new)]
    ```

    Greedy algorithm: always ask the highest-IG question.

    ### Process

    1. Start with uniform priors (12.5% each)
    2. Calculate information gain for all unasked questions
    3. Ask highest-IG question
    4. Update probabilities using Bayes' Rule
    5. Repeat until ≥90% confidence

    ### Performance

    - **3-6 questions** to reach 90% confidence (8 diagnoses)
    - **~85% efficiency** vs. asking all 35 questions
    - **Scales logarithmically** with problem size

    ### Knowledge Base

    **8 diagnoses** × **36 symptoms** = **288 CPT probabilities**

    Covers primary headaches (migraine, tension, cluster), secondary headaches (sinus, medication overuse, cervicogenic, hypertensive), and neuropathic pain (trigeminal neuralgia).

    Each P(Symptom | Diagnosis) value calibrated from clinical literature (ICHD-3, medical journals).
    """)

    # Show sample CPT data
    with st.expander("View Sample CPT Data"):
        st.markdown("**Example: Band-like Pain (S17)**")
        s17_data = {
            "Diagnosis": DIAGNOSES,
            "P(Band-like Pain | Diagnosis)": [CPT[d]["S17"] for d in DIAGNOSES],
            "Interpretation": [
                "Rare in migraine",
                "HALLMARK of tension-type",
                "Not typical for cluster",
                "Not typical for sinus",
                "Can occur in MOH",
                "Sometimes in cervicogenic",
                "Not typical for trigeminal neuralgia",
                "Sometimes in hypertensive",
            ],
        }
        st.dataframe(pd.DataFrame(s17_data), use_container_width=True)
        st.markdown(
            "Notice how Tension-Type has 0.90 (90%) - this makes it a highly discriminative symptom!"
        )

with tab4:
    # Sources and Appendix
    st.header("Sources & Appendix")

    st.markdown("""
    ### References

    #### Medical Knowledge - CPT Table Sources

    The Conditional Probability Table values are based on clinical diagnostic criteria and symptom prevalence data from:

    1. **International Classification of Headache Disorders, 3rd Edition (ICHD-3)**
       - [Official ICHD-3 Criteria](https://ichd-3.org/)
       - Published by the International Headache Society
       - Defines diagnostic criteria for migraine, tension-type, cluster, and other headache disorders
       - Gold standard for headache classification

    2. **Migraine Diagnostic Criteria & Prevalence**
       - Diagnostic criteria: [NIH - Migraine Information](https://www.ninds.nih.gov/health-information/disorders/migraine)
       - Symptom prevalence studies: [American Migraine Foundation](https://americanmigrainefoundation.org/)
       - Key symptoms: Throbbing pain (85-90%), photophobia/phonophobia (80-90%), nausea (70-85%)

    3. **Tension-Type Headache Characteristics**
       - Clinical review: [Mayo Clinic - Tension Headache](https://www.mayoclinic.org/diseases-conditions/tension-headache/)
       - Defining feature: Band-like or pressing quality (85-95%)
       - Typically bilateral (80-90%), mild-to-moderate intensity

    4. **Cluster Headache Features**
       - Diagnostic criteria: [American Academy of Neurology - Cluster Headache](https://www.aan.com/)
       - Hallmark: Severe unilateral pain with autonomic features (90-95%)
       - Brief duration: 15-180 minutes (diagnostic requirement)
       - Patient restlessness during attacks (80-90%)

    5. **Sinus Headache & Secondary Causes**
       - Clinical guidelines: [American Rhinologic Society](https://www.american-rhinologic.org/)
       - Associated with sinusitis symptoms: facial pressure (90%), nasal discharge (80-85%)
       - Often follows upper respiratory infection (70-80%)

    6. **Medication Overuse Headache (MOH)**
       - ICHD-3 diagnostic criteria and prevalence studies
       - Hallmark: Frequent medication use (>10-15 days/month) with daily/near-daily headaches
       - Morning headache pattern (85%), increasing frequency over time (90%)

    7. **Cervicogenic Headache**
       - Clinical guidelines from cervicogenic headache literature
       - Key features: Neck-related triggers (90%), limited range of motion (85%)
       - Pain radiating from neck to head (90%)

    8. **Trigeminal Neuralgia**
       - ICHD-3 neuropathic pain criteria
       - Diagnostic features: Electric shock-like pain (98%), very brief duration (95%)
       - Facial trigger points (95%)

    9. **Hypertensive Headache**
       - Clinical guidelines for secondary headaches related to hypertension
       - Associated with high blood pressure (95%), morning headaches (70%)
       - Characteristic back-of-head location (75%)

    10. **Differential Diagnosis Resources**
       - Headache classification review: Robbins MS. "The Headache Classification Committee of the International Headache Society (IHS). Cephalalgia. 2018;38(1):1-211.
       - Clinical approach to headache: [UpToDate - Approach to the patient with headache](https://www.uptodate.com/)

    **Note on CPT Values:** The probability values in our CPT table represent approximate symptom prevalence rates derived from the above clinical literature. These are educational estimates based on typical diagnostic patterns and are not intended for actual clinical use.

    #### Mathematical Foundations
    - **Entropy** 
      - Foundation of information theory
      - Entropy as measure of uncertainty

    - **Bayes' Theorem** 
      - Conditional probability
      - Bayesian inference

    - **Information Gain** - Decision tree learning, feature selection
      - Greedy algorithm for question selection

    - **NEW: Entropy Reduction Chain**
      - Information gain compounds over time

    - **NEW: Discriminative Power**
      - How much a symptom can distinguish between diagnoses

    ### Complete CPT Table

    Below is the complete Conditional Probability Table used by the system:
    """)

    # Display full CPT
    cpt_display = []
    for diagnosis in DIAGNOSES:
        for symptom_key, prob in CPT[diagnosis].items():
            cpt_display.append(
                {
                    "Diagnosis": diagnosis,
                    "Symptom Key": symptom_key,
                    "Symptom": SYMPTOMS[symptom_key][:60] + "..."
                    if len(SYMPTOMS[symptom_key]) > 60
                    else SYMPTOMS[symptom_key],
                    "P(Symptom|Diagnosis)": prob,
                }
            )

    cpt_df = pd.DataFrame(cpt_display)
    st.dataframe(cpt_df, use_container_width=True, height=400)

    st.markdown("""
    ### Academic Context

    **Course:** CS 109

    **Project Components:**
    - Knowledge base design (CPT)
    - Bayesian inference engine
    - Information-theory question selection
    - Interactive visualization and user interface
    - Analysis of information gain and entropy reduction

    ### Technical Implementation

    **Technologies:**
    - Python 3.9+
    - Streamlit (web framework)
    - NumPy (numerical computation)
    - Plotly (interactive visualizations)
    - Pandas (data manipulation)

    **Code Structure:**
    - `data.py` - Knowledge base (diagnoses, symptoms, CPT)
    - `engine.py` - Mathematical functions (entropy, Bayes, IG)
    - `analysis.py` - Computations for more insights
    - `streamlit_app.py` - Web interface and visualizations

    **Repository:** [Add your GitHub link here when deployed]
    """)


# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <p>infor<span style='text-decoration: underline; text-decoration-color: #e74c3c; text-decoration-thickness: 1px; text-underline-offset: 2px;'>med</span> | CS 109 Project: Bayesian Information-Guided Diagnosis | Megha Bindiganavale</p>
    <p>Built with Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
