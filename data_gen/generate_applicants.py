"""Generate synthetic university applicants (profiles + documents).

Produces per applicant:
  data/raw/APP-XXXX/profile.json      structured profile
  data/raw/APP-XXXX/sop.txt           statement of purpose
  data/raw/APP-XXXX/lor_1.txt         recommendation letter
  data/raw/APP-XXXX/lor_2.txt         recommendation letter
  data/raw/APP-XXXX/cv.txt            CV / resume
  data/raw/APP-XXXX/transcript.txt    course grades

Entirely synthetic (Faker) → GDPR-safe by construction, but the pipeline
still treats fields like name/email as PII to demonstrate pseudonymization.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from faker import Faker

fake = Faker()

PROGRAMS = [
    "MSc Computer Science", "MSc Data Science", "MSc Artificial Intelligence",
    "MSc Software Engineering", "MSc Robotics",
]
INTERESTS = [
    "natural language processing", "computer vision", "reinforcement learning",
    "distributed systems", "human-computer interaction", "AI safety",
    "recommender systems", "edge computing", "knowledge graphs", "MLOps",
]
COURSES = [
    "Data Structures", "Algorithms", "Linear Algebra", "Probability & Statistics",
    "Machine Learning", "Databases", "Operating Systems", "Computer Networks",
    "Software Engineering", "Discrete Mathematics", "Deep Learning", "Compilers",
]
GRADES = ["A", "A-", "B+", "B", "B-", "C+"]
GRADE_POINTS = {"A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7, "C+": 2.3}


def make_profile(i: int) -> dict:
    strength = random.choices(["strong", "average", "weak"], weights=[3, 5, 2])[0]
    gpa_range = {"strong": (3.6, 4.0), "average": (3.0, 3.6), "weak": (2.4, 3.0)}[strength]
    courses = random.sample(COURSES, 8)
    transcript = []
    for c in courses:
        pool = {"strong": GRADES[:3], "average": GRADES[1:5], "weak": GRADES[3:]}[strength]
        transcript.append({"course": c, "grade": random.choice(pool)})
    gpa = round(sum(GRADE_POINTS[t["grade"]] for t in transcript) / len(transcript), 2)
    gpa = max(min(gpa, gpa_range[1]), gpa_range[0] - 0.2)

    return {
        "applicant_id": f"APP-{i:04d}",
        "name": fake.name(),
        "email": fake.email(),
        "nationality": fake.country(),
        "program": random.choice(PROGRAMS),
        "undergrad_degree": random.choice(["BSc Computer Science", "BSc Mathematics",
                                           "BEng Electrical Engineering", "BSc Physics"]),
        "undergrad_university": f"{fake.city()} University",
        "gpa": gpa,
        "gpa_scale": 4.0,
        "ielts": round(random.uniform(6.0, 8.5), 1),
        "gre_quant": random.randint(150, 170),
        "interests": random.sample(INTERESTS, 2),
        "work_experience_years": random.choice([0, 0, 1, 1, 2, 3]),
        "publications": random.choices([0, 0, 0, 1, 2], weights=[5, 3, 2, 2, 1])[0],
        "_strength": strength,  # hidden ground-truth label, useful for eval
    }


def make_sop(p: dict) -> str:
    i1, i2 = p["interests"]
    proj = fake.bs()
    return (
        f"Statement of Purpose — {p['program']}\n\n"
        f"My fascination with {i1} began during my {p['undergrad_degree']} at "
        f"{p['undergrad_university']}, where a course project on {proj} showed me how "
        f"theory translates into real impact. Since then I have deepened my skills in "
        f"{i1} and {i2}, completing an undergraduate thesis on "
        f"\"{i1.title()} approaches to {fake.catch_phrase().lower()}\".\n\n"
        f"{'During ' + str(p['work_experience_years']) + ' year(s) in industry, I built production systems and learned to ship reliable software under real constraints. ' if p['work_experience_years'] else 'While I lack industry experience, I have invested heavily in self-driven projects and open-source contributions. '}"
        f"{'I have co-authored ' + str(p['publications']) + ' peer-reviewed publication(s), which taught me rigorous experimental methodology. ' if p['publications'] else ''}\n\n"
        f"Your program's strengths in {i2} align directly with my goal of pursuing "
        f"research at the intersection of {i1} and {i2}. Long term, I aim to "
        f"{random.choice(['pursue a PhD', 'lead an applied research team', 'build products that make AI trustworthy'])}. "
        f"I am confident my background in {p['undergrad_degree']} (GPA {p['gpa']}/4.0) "
        f"prepares me to contribute meaningfully to your community.\n"
    )


def make_lor(p: dict, kind: str) -> str:
    tone = {"strong": "one of the top 5% of students I have taught",
            "average": "a solid and dependable student",
            "weak": "a student who showed potential but inconsistent performance"}[p["_strength"]]
    ref = fake.name()
    role = "Professor" if kind == "academic" else "Engineering Manager"
    return (
        f"Letter of Recommendation\n\nTo the Admissions Committee,\n\n"
        f"I am writing to recommend the applicant for the {p['program']} program. "
        f"As their {role}, I consider them {tone}. "
        f"They demonstrated particular aptitude in {p['interests'][0]}, notably in a "
        f"project involving {fake.bs()}. "
        f"{'Their analytical writing and initiative were exceptional.' if p['_strength'] == 'strong' else 'With mentorship, they improved steadily over the term.'}\n\n"
        f"I recommend them {'without reservation' if p['_strength'] == 'strong' else 'with confidence' if p['_strength'] == 'average' else 'with some reservations noted above'}.\n\n"
        f"Sincerely,\n{ref}, {role}\n"
    )


def make_cv(p: dict) -> str:
    skills = random.sample(["Python", "PyTorch", "TensorFlow", "SQL", "Docker",
                            "C++", "Git", "Linux", "AWS", "Spark"], 6)
    lines = [
        "Curriculum Vitae",
        f"\nEducation\n- {p['undergrad_degree']}, {p['undergrad_university']} — GPA {p['gpa']}/4.0",
        f"\nSkills\n- {', '.join(skills)}",
        "\nProjects",
    ]
    for _ in range(random.randint(2, 4)):
        lines.append(f"- {fake.catch_phrase()}: {fake.bs()} using {random.choice(skills)}")
    if p["work_experience_years"]:
        lines.append(f"\nExperience\n- Software Engineer, {fake.company()} "
                     f"({p['work_experience_years']} yr): {fake.bs()}")
    if p["publications"]:
        lines.append(f"\nPublications\n- {p['publications']} peer-reviewed paper(s) "
                     f"on {p['interests'][0]}")
    return "\n".join(lines) + "\n"


def make_transcript(p: dict) -> str:
    lines = [f"Official Transcript — {p['undergrad_university']}",
             f"Degree: {p['undergrad_degree']}   Cumulative GPA: {p['gpa']}/4.0", ""]
    random.seed(p["applicant_id"])
    for c in random.sample(COURSES, 8):
        pool = {"strong": GRADES[:3], "average": GRADES[1:5], "weak": GRADES[3:]}[p["_strength"]]
        lines.append(f"  {c:<28} {random.choice(pool)}")
    return "\n".join(lines) + "\n"


def main(n: int, out_dir: str, seed: int) -> None:
    random.seed(seed)
    Faker.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        p = make_profile(i)
        d = out / p["applicant_id"]
        d.mkdir(exist_ok=True)
        (d / "profile.json").write_text(json.dumps(p, indent=2))
        (d / "sop.txt").write_text(make_sop(p))
        (d / "lor_1.txt").write_text(make_lor(p, "academic"))
        (d / "lor_2.txt").write_text(make_lor(p, "industry"))
        (d / "cv.txt").write_text(make_cv(p))
        (d / "transcript.txt").write_text(make_transcript(p))
    print(f"Generated {n} applicants in {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.n, args.out, args.seed)
