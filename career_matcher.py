
import re
import json
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EDU_LEVELS = {
    "any": 0,
    "diploma": 1,
    "bachelor": 2,
    "master": 3,
    "phd": 4
}

EDU_ALIASES = {
    "bsc": "bachelor",
    "b.eng": "bachelor",
    "ba": "bachelor",
    "bs": "bachelor",
    "msc": "master",
    "ms": "master",
    "m.eng": "master",
    "ph.d": "phd",
    "doctorate": "phd",
    "hnd": "diploma",
    "higher national diploma": "diploma",
    "nvq": "diploma",
}

SKILL_SYNONYMS = {
    "py": "python",
    "python3": "python",
    "ml": "machine learning",
    "dl": "deep learning",
    "viz": "data visualization",
    "vcs": "version control",
    "git": "version control",
    "sql server": "sql",
    "ms excel": "excel",
    "powerbi": "power bi",
    "tf": "tensorflow",
}

def _canon(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def parse_education_level(text: str) -> int:
    t = _canon(text)
    # look for explicit keywords
    for k,v in EDU_ALIASES.items():
        if k in t:
            return EDU_LEVELS[v]
    for k in EDU_LEVELS.keys():
        if k in t:
            return EDU_LEVELS[k]
    # defaults: if "bachelor" like string appears
    if re.search(r"b\w*", t):
        return EDU_LEVELS["bachelor"]
    return EDU_LEVELS["any"]

def parse_experience_years(text: str) -> float:
    t = _canon(text)
    # 1) "X+ years", "X years", "X year"
    m = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:year|years|yr|yrs)", t)
    years = max([float(x) for x in m], default=0.0)
    # 2) months: "6 months"
    m2 = re.findall(r"(\d+(?:\.\d+)?)\s*(?:month|months|mo)", t)
    months = max([float(x) for x in m2], default=0.0)
    years = max(years, months/12.0)
    return years

def normalize_skill(s: str) -> str:
    k = _canon(s)
    return SKILL_SYNONYMS.get(k, k)

def normalize_skills(skills: List[str]) -> List[str]:
    norm = []
    for s in skills:
        if not s: 
            continue
        k = normalize_skill(s)
        norm.append(k)
    # de-dup while preserving order
    seen = set()
    out = []
    for k in norm:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out

def softmax(x, temperature=0.8):
    x = np.array(x, dtype=float)
    x = x / max(temperature, 1e-6)
    x = x - x.max()
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-9)

class CareerMatcher:
    def __init__(self, careers_csv: str, career_skills_csv: str, alpha=0.6, beta=0.2, gamma=0.1, delta=0.1, temperature=0.8):
        self.careers = pd.read_csv(careers_csv)
        self.career_skills = pd.read_csv(career_skills_csv)
        self.alpha = alpha; self.beta = beta; self.gamma = gamma; self.delta = delta
        self.temperature = temperature
        # Build skill taxonomy
        self.skill_vocab = sorted(self.career_skills["skill"].str.lower().unique().tolist())
        self.skill_index = {s:i for i,s in enumerate(self.skill_vocab)}
        # Career -> weighted skill vector (len V)
        self.career_vecs = self._build_career_vectors()
        # TF-IDF on career descriptions
        self.tfidf = TfidfVectorizer(min_df=1, max_features=2000, ngram_range=(1,2))
        self.career_desc_matrix = self.tfidf.fit_transform(self.careers["description"].fillna(""))

    def _build_career_vectors(self):
        V = len(self.skill_vocab)
        vecs = np.zeros((len(self.careers), V), dtype=float)
        for _, row in self.career_skills.iterrows():
            cid = int(row["career_id"]) - 1  # career_id starts at 1 in file
            skill = row["skill"].lower()
            imp = float(row["importance"])
            j = self.skill_index.get(skill)
            if j is not None and 0 <= cid < len(self.careers):
                vecs[cid, j] = max(vecs[cid, j], imp)  # take max importance if duplicates
        return vecs

    def _skill_coverage(self, user_vec, career_vec):
        # coverage = sum(importance for skills the user has) / sum(total importance)
        denom = career_vec.sum()
        if denom <= 0:
            return 0.0
        matched = (user_vec > 0).astype(float) * career_vec
        return float(matched.sum() / denom)

    def _education_fit(self, user_level, career_min):
        req = parse_education_level(career_min)
        return 1.0 if user_level >= req else 0.0

    def _experience_fit(self, user_years, pref_years):
        pref = float(pref_years or 0.0)
        if pref <= 0:
            return 1.0
        return float(min(1.0, user_years / pref))

    def _interest_similarity(self, interests_text):
        q = self.tfidf.transform([interests_text or ""])
        sims = cosine_similarity(q, self.career_desc_matrix).ravel()
        return sims

    def rank(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Parse & encode user
        user_skills = normalize_skills(payload.get("skills") or [])
        user_level = parse_education_level(payload.get("education","any"))
        user_years = parse_experience_years(payload.get("experience",""))
        interests = payload.get("interests","")
        # Build user vector (binary on our vocab)
        V = len(self.skill_vocab)
        uvec = np.zeros(V, dtype=float)
        for s in user_skills:
            j = self.skill_index.get(s)
            if j is not None:
                uvec[j] = 1.0

        # Components per career
        coverages = np.array([self._skill_coverage(uvec, self.career_vecs[i]) for i in range(len(self.careers))])
        edu_fit = np.array([self._education_fit(user_level, self.careers.iloc[i]["min_education"]) for i in range(len(self.careers))], dtype=float)
        exp_fit = np.array([self._experience_fit(user_years, self.careers.iloc[i]["pref_exp_years"]) for i in range(len(self.careers))], dtype=float)
        interest_sims = self._interest_similarity(interests)

        # Score
        score = self.alpha*coverages + self.beta*interest_sims + self.gamma*edu_fit + self.delta*exp_fit
        probs = softmax(score, self.temperature)
        idx = np.argsort(-probs)
        primary = idx[0]; alts = idx[1:3]

        def top_matched_skills(career_idx, k=6):
            # pick career skills sorted by importance and intersect with user skills
            cskills = self.career_skills[self.career_skills["career_id"] == (career_idx+1)].sort_values("importance", ascending=False)
            matched = [s for s in cskills["skill"].tolist() if s in user_skills]
            return matched[:k]

        def top_missing_skills(career_idx, k=3):
            cskills = self.career_skills[self.career_skills["career_id"] == (career_idx+1)].sort_values("importance", ascending=False)
            missing = [s for s in cskills["skill"].tolist() if s not in user_skills]
            return missing[:k]

        # Build explanation (lightweight rationale for v1)
        why = {
            "matched_skills": top_matched_skills(primary),
            "missing_high_value_skills": top_missing_skills(primary),
            "skill_coverage": float(coverages[primary]),
            "interest_similarity": float(interest_sims[primary]),
            "education_fit": float(edu_fit[primary]),
            "experience_fit": float(exp_fit[primary]),
        }

        def pack(i):
            return {
                "career": self.careers.iloc[i]["career_title"],
                "prob": float(probs[i])
            }

        return {
            "primary": {**pack(primary), "why_primary": why},
            "alternatives": [pack(i) for i in alts],
            "debug": {
                "components": {
                    "skill_coverage": coverages.tolist(),
                    "interest_similarity": interest_sims.tolist(),
                    "education_fit": edu_fit.tolist(),
                    "experience_fit": exp_fit.tolist()
                },
                "order": idx.tolist()
            }
        }

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--careers", required=True)
    p.add_argument("--career_skills", required=True)
    p.add_argument("--payload", required=False, help="JSON payload file or inline JSON")
    args = p.parse_args()
    matcher = CareerMatcher(args.careers, args.career_skills)
    if args.payload:
        try:
            if args.payload.strip().startswith("{"):
                payload = json.loads(args.payload)
            else:
                with open(args.payload, "r") as f:
                    payload = json.load(f)
        except Exception as e:
            print("Failed to read payload:", e, file=sys.stderr)
            sys.exit(1)
    else:
        payload = {
            "skills": ["python","sql","statistics","excel","machine learning","deep learning","pandas"],
            "education": "BSc in Computer Science",
            "experience": "6 months analytics internship; 1 year freelance ML projects",
            "interests": "machine learning, research, data visualization"
        }
    res = matcher.rank(payload)
    print(json.dumps(res, indent=2))
