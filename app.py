import streamlit as st
import json
from career_matcher import CareerMatcher

# Paths to your CSV files
CAREERS_CSV = "data/careers.csv"
CAREER_SKILLS_CSV = "data/career_skills.csv"

# Load the model
matcher = CareerMatcher(CAREERS_CSV, CAREER_SKILLS_CSV)

st.title("🎯 Career Matcher")
st.markdown("Provide your skills, education, experience, and interests to get career suggestions.")

# Skills input
all_skills = matcher.skill_vocab
user_skills = st.multiselect("Select your skills:", all_skills)

# Education level
education_options = ["Any", "Diploma", "Bachelor", "Master", "PhD"]
education = st.radio("Select your education level:", education_options)

# Experience input
experience = st.text_input("Enter your experience (e.g., '2 years', '6 months'):")

# Interests input
interests = st.text_input("Enter your interests (comma separated):")

# Submit button
if st.button("Find Careers"):
    payload = {
        "skills": user_skills,
        "education": education,
        "experience": experience,
        "interests": interests
    }
    results = matcher.rank(payload)

    # Display primary career
    primary = results['primary']
    st.subheader("🏆 Primary Career Match")
    st.markdown(f"### {primary['career']} — **{primary['prob']*100:.0f}% Match**")

    # Matched Skills
    st.markdown("**Matched Skills:**")
    st.write(", ".join(primary['why_primary']['matched_skills']))

    # Skills to Develop
    st.markdown("**Skills to Develop:**")
    st.write(", ".join(primary['why_primary']['missing_high_value_skills']))

    # Skill Coverage
    st.markdown("**Skill Coverage:**")
    skill_cov = int(primary['why_primary']['skill_coverage']*100)
    st.progress(skill_cov)

    # Interest Alignment
    st.markdown("**Interest Alignment:**")
    interest_align = int(primary['why_primary']['interest_similarity']*100)
    st.progress(interest_align)

    # Education & Experience Fit
    col1, col2 = st.columns(2)
    col1.metric("Education Fit", primary['why_primary']['education_fit'])
    col2.metric("Experience Fit", primary['why_primary']['experience_fit'])

    # Alternative Careers
    st.subheader("💡 Alternative Career Options")
    for alt in results["alternatives"]:
        st.markdown(f"**{alt['career']}**")
        st.progress(int(alt['prob']*100))