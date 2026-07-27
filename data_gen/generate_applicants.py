"""Generate synthetic university applicants for University of Stuttgart MSc programs.

v2 — realistic long-form documents:
  profile.json      structured profile (incl. German equivalent grade)
  sop.txt           Statement of Purpose, 5 paragraphs
  lor_1.txt         academic LOR, full formal letter, 4-5 paragraphs
  lor_2.txt         industry/second LOR, full formal letter, 4-5 paragraphs
  cv.txt            detailed CV: summary, experience, projects, skills, activities
  transcript.txt    semester-wise Bachelor Transcript of Records with module codes,
                    credits, grades, CGPA and German equivalent grade (Bavarian formula)

Target programs (University of Stuttgart):
  MSc Computer Science | MSc Information Technology (INFOTECH) | MSc Electrical Engineering

Entirely synthetic (Faker). The pipeline still treats name/email as PII
to demonstrate GDPR pseudonymization.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

fake = Faker()

# --------------------------------------------------------------------------
# Domain model: programs, specializations, module catalogues
# --------------------------------------------------------------------------

PROGRAMS = {
    "MSc Computer Science": {
        "code": "CS",
        "specializations": ["Artificial Intelligence", "Software Engineering",
                            "Distributed Systems", "Visual Computing"],
        "qualifying_degrees": ["BSc Computer Science", "BSc Information Technology",
                               "BE Computer Engineering"],
        "msc_courses": ["Machine Learning", "Distributed Systems",
                        "Advanced Software Engineering", "Deep Learning",
                        "Data Structures and Algorithms II", "Computer Vision"],
        "institutes": ["Institute for Parallel and Distributed Systems (IPVS)",
                       "Institute of Architecture of Application Systems (IAAS)",
                       "Institute for Artificial Intelligence",
                       "Institute for Visualization and Interactive Systems (VIS)"],
    },
    "MSc Information Technology (INFOTECH)": {
        "code": "INFOTECH",
        "specializations": ["Embedded Systems", "Communications",
                            "Computer Hardware/Software Engineering",
                            "Micro- and Optoelectronics"],
        "qualifying_degrees": ["BE Electronics & Telecommunication",
                               "BSc Computer Science", "BEng Electrical Engineering",
                               "BSc Information Technology"],
        "msc_courses": ["Embedded Systems Engineering", "Electronic Design Automation",
                        "Advanced Higher Mathematics for Infotech", "Deep Learning",
                        "Hardware Platforms and Programming of Embedded Systems",
                        "Service Management and Cloud Computing"],
        "institutes": ["Institute of Technical Informatics (ITI)",
                       "Institute of Signal Processing and System Theory (ISS)",
                       "Institute of Communication Networks and Computer Engineering (IKR)",
                       "Institute of Smart Sensors"],
    },
    "MSc Electrical Engineering": {
        "code": "EE",
        "specializations": ["Micro- and Nanoelectronics", "Signal Processing and Communications",
                            "Automation Technology", "Sustainable Electrical Energy Systems"],
        "qualifying_degrees": ["BEng Electrical Engineering",
                               "BE Electronics & Telecommunication",
                               "BSc Electrical and Electronics Engineering"],
        "msc_courses": ["Power Electronics", "Advanced Control Systems",
                        "Statistical Signal Processing", "Smart Grids",
                        "Semiconductor Devices", "Industrial Automation Systems"],
        "institutes": ["Institute of Power Electronics and Electrical Drives (ILEA)",
                       "Institute of Signal Processing and System Theory (ISS)",
                       "Institute of Robust Power Semiconductor Systems (ILH)",
                       "Institute for System Dynamics (ISYS)"],
    },
}

# Bachelor module catalogue keyed by broad discipline, grouped roughly by year.
BACHELOR_MODULES = {
    "common_y1": [
        "Engineering Mathematics I", "Engineering Mathematics II", "Engineering Physics",
        "Fundamentals of Programming I", "Fundamentals of Programming II",
        "Engineering Graphics", "Basic Electrical Engineering", "Engineering Chemistry",
        "Workshop Practices", "Engineering Mechanics",
    ],
    "cs_core": [
        "Data Structures and Algorithms", "Discrete Mathematics", "Computer Organization",
        "Object Oriented Programming", "Operating Systems", "Database Management Systems",
        "Theory of Computation", "Computer Networks", "Software Engineering",
        "Design and Analysis of Algorithms", "Compiler Design", "Web Technologies",
    ],
    "cs_adv": [
        "Machine Learning", "Artificial Intelligence", "Distributed Computing",
        "Cloud Computing", "Information Security", "Data Mining",
        "Mobile Application Development", "High Performance Computing",
        "Natural Language Processing", "Big Data Analytics",
    ],
    "ee_core": [
        "Signals & Systems", "Electronic Devices & Circuits", "Digital Electronics",
        "Electric Circuits & Machines", "Electromagnetics", "Control Systems",
        "Microcontrollers", "Analog Communication", "Integrated Circuits",
        "Electrical Measurements & Instrumentation", "Network Analysis",
    ],
    "ee_adv": [
        "Digital Signal Processing", "Power Electronics", "Digital Communication",
        "Embedded Systems & RTOS", "VLSI Design & Technology", "Microwave Engineering",
        "Mobile Communication", "Wireless Sensor Networks", "Mechatronics",
        "Advanced Processors", "Power Systems", "Broadband Communication Systems",
        "Machine Learning", "Artificial Intelligence",
    ],
    "labs": [
        "Programming Laboratory", "Electronics Laboratory", "Signal Processing & Communication Lab",
        "Microcontroller & Mechatronics Lab", "VLSI & Embedded Lab", "Networks Laboratory",
        "Machine Learning Laboratory", "Project Stage I", "Project Stage II",
    ],
    "soft": [
        "Business Management", "Employability Skill Development", "Cyber Crime and Law",
        "Human Behaviour", "Road Safety Management", "Team Building, Leadership & Fitness",
        "Technical Communication", "Professional Ethics",
    ],
}

GRADE_LETTERS = [("O", 10), ("A+", 9), ("A", 8), ("B+", 7), ("B", 6), ("C", 5)]

NATIONALITIES = ["India", "China", "Turkey", "Iran", "Pakistan", "Vietnam", "Egypt",
                 "Brazil", "Indonesia", "Bangladesh", "Mexico", "Nigeria", "Colombia"]
APS_COUNTRIES = {"China", "Vietnam", "Mongolia"}

UNI_SUFFIXES = ["Institute of Technology", "University of Technology",
                "College of Engineering", "Technical University", "State University"]

COMPANIES_ROLES = [
    ("Software Engineering Intern", "developed REST APIs and data pipelines using Python and SQL"),
    ("Embedded Systems Intern", "wrote Embedded C firmware for microcontrollers with CAN and SPI drivers"),
    ("Working Student - IoT Analytics", "built real-time telemetry pipelines using MQTT, InfluxDB and Grafana"),
    ("R&D Intern - Machine Learning", "trained and evaluated CNN models and built preprocessing pipelines"),
    ("Data Engineering Intern", "implemented ETL workflows and optimized SQL queries for reporting"),
    ("Backend Developer Intern", "implemented microservices with Docker, CI/CD and automated testing"),
    ("Hardware Design Intern", "designed PCBs and validated sensor interfaces in lab conditions"),
]

SKILLS = {
    "CS": ["Python", "C++", "Java", "SQL", "Docker", "Git", "Linux", "PyTorch",
           "TensorFlow", "REST APIs", "Kubernetes", "React"],
    "INFOTECH": ["C", "C++", "Python", "Embedded C", "MQTT", "Docker", "Git", "Linux",
                 "FreeRTOS", "OPC-UA", "InfluxDB", "MATLAB"],
    "EE": ["MATLAB", "Simulink", "C", "Python", "VHDL", "PCB Design", "PSpice",
           "Embedded C", "LabVIEW", "PLC Programming", "Git", "Linux"],
}

ACTIVITIES = [
    "organized technical workshops and hackathons as an IEEE student branch coordinator",
    "led the avionics subsystem of the university rocketry team",
    "served as a teaching assistant for undergraduate programming courses",
    "volunteered as a coding mentor for first-year students",
    "captained the university robotics club in national competitions",
    "coordinated the annual departmental technical symposium",
    "contributed to open-source projects in the embedded and ML ecosystem",
    "led a student chapter organizing guest lectures with industry professionals",
]


# --------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------

def german_grade(cgpa: float, nmax: float = 10.0, nmin: float = 5.0) -> float:
    """Modified Bavarian formula: 1.0 (best) .. 4.0 (pass)."""
    g = 1.0 + 3.0 * (nmax - cgpa) / (nmax - nmin)
    return round(min(max(g, 1.0), 4.0), 2)


def make_profile(i: int) -> dict:
    strength = random.choices(["strong", "average", "weak"], weights=[3, 5, 2])[0]
    program_name = random.choice(list(PROGRAMS))
    prog = PROGRAMS[program_name]
    cgpa = round(random.uniform(*{"strong": (8.4, 9.6), "average": (7.2, 8.4),
                                  "weak": (6.0, 7.2)}[strength]), 2)
    nationality = random.choice(NATIONALITIES)
    english_test = random.choice(["IELTS", "TOEFL iBT"])
    if english_test == "IELTS":
        score = round(random.uniform(*{"strong": (7.0, 8.5), "average": (6.5, 7.5),
                                       "weak": (6.0, 7.0)}[strength]) * 2) / 2
    else:
        score = random.randint(*{"strong": (95, 118), "average": (85, 100),
                                 "weak": (75, 92)}[strength])
    grad_year = random.randint(2020, 2024)
    return {
        "applicant_id": f"APP-{i:04d}",
        "name": fake.name(),
        "email": fake.email(),
        "date_of_birth": fake.date_of_birth(minimum_age=21, maximum_age=28).isoformat(),
        "nationality": nationality,
        "aps_required": nationality in APS_COUNTRIES,
        "target_university": "University of Stuttgart",
        "program": program_name,
        "intended_specialization": random.choice(prog["specializations"]),
        "undergrad_degree": random.choice(prog["qualifying_degrees"]),
        "undergrad_university": f"{fake.city()} {random.choice(UNI_SUFFIXES)}",
        "undergrad_duration_years": 4,
        "graduation_year": grad_year,
        "cgpa": cgpa,
        "cgpa_scale": 10.0,
        "german_equivalent_grade": german_grade(cgpa),
        "english_test": english_test,
        "english_score": score,
        "gre_quant": random.randint(152, 170) if random.random() < 0.5 else None,
        "work_experience_years": round(min(max(0, 2026 - grad_year - random.randint(0, 2)), 4)),
        "internships": random.randint(1, 3) if strength != "weak" else random.randint(0, 2),
        "publications": random.choices([0, 0, 0, 1, 2], weights=[5, 3, 2, 2, 1])[0],
        "interests": random.sample(
            {"CS": ["machine learning", "distributed systems", "computer vision",
                    "natural language processing", "software architecture", "AI safety"],
             "INFOTECH": ["embedded systems", "edge AI", "hardware-software co-design",
                          "communication networks", "IoT platforms", "real-time systems"],
             "EE": ["power electronics", "control systems", "signal processing",
                    "smart grids", "semiconductor devices", "e-mobility"]}[prog["code"]], 2),
        "_strength": strength,
    }


# --------------------------------------------------------------------------
# Transcript of Records (detailed, semester-wise)
# --------------------------------------------------------------------------

def _pick_grade(strength: str) -> tuple[str, int]:
    pools = {"strong": GRADE_LETTERS[:3], "average": GRADE_LETTERS[1:5],
             "weak": GRADE_LETTERS[2:]}
    return random.choice(pools[strength])


def _semester_courses(prog_code: str, sem: int) -> list[str]:
    """Assemble a plausible course list for a given semester."""
    m = BACHELOR_MODULES
    core = m["cs_core"] if prog_code == "CS" else m["ee_core"]
    adv = m["cs_adv"] if prog_code == "CS" else m["ee_adv"]
    if prog_code == "INFOTECH":  # blend, mirroring the CS+EE character of INFOTECH
        core, adv = m["ee_core"] + m["cs_core"][:6], m["ee_adv"] + m["cs_adv"][:4]
    if sem <= 2:
        pool = m["common_y1"]
    elif sem <= 5:
        pool = core
    else:
        pool = adv
    k = min(5, len(pool))
    courses = random.sample(pool, k)
    if sem >= 3 and random.random() < 0.8:
        courses.append(random.choice(m["labs"]))
    if random.random() < 0.5:
        courses.append(random.choice(m["soft"]))
    return courses


def make_transcript(p: dict) -> str:
    random.seed(p["applicant_id"] + "tr")
    prog_code = PROGRAMS[p["program"]]["code"]
    start_year = p["graduation_year"] - 4
    lines = [
        f"{p['undergrad_university'].upper()}",
        "OFFICE OF THE CONTROLLER OF EXAMINATIONS",
        "=" * 78,
        "TRANSCRIPT OF RECORDS",
        "=" * 78,
        f"Name of Student   : {p['name']}",
        f"Degree            : {p['undergrad_degree']} (4-year program)",
        f"Enrollment Number : {random.randint(10**8, 10**9 - 1)}",
        f"Period of Study   : {start_year} - {p['graduation_year']}",
        f"Medium of Instruction: English",
        "",
        "Grading scale: O=10, A+=9, A=8, B+=7, B=6, C=5 (grade points, 10-point CGPA scale)",
        "-" * 78,
    ]
    total_credits = 0
    used: set[str] = set()
    for sem in range(1, 9):
        year = start_year + (sem - 1) // 2
        term = "Winter" if sem % 2 else "Summer"
        lines.append(f"\nSEMESTER {sem}  ({term} {year})")
        lines.append(f"{'Code':<10}{'Module Title':<46}{'Cr':>3} {'Grade':>6} {'Status':>7}")
        sem_pts = sem_cr = 0
        for course in _semester_courses(prog_code, sem):
            if course in used:
                continue
            used.add(course)
            code = f"{prog_code[:2]}{sem}{random.randint(100, 999)}"
            credits = random.choice([2, 3, 3, 4, 4])
            letter, pts = _pick_grade(p["_strength"])
            lines.append(f"{code:<10}{course:<46}{credits:>3} {letter:>6} {'PASS':>7}")
            sem_pts += pts * credits
            sem_cr += credits
        total_credits += sem_cr
        lines.append(f"{'':<10}{'Semester GPA':<46}{'':>3} {sem_pts / max(sem_cr, 1):>6.2f}")
    lines += [
        "-" * 78,
        f"TOTAL CREDITS EARNED : {total_credits}",
        f"CUMULATIVE GPA (CGPA): {p['cgpa']:.2f} / 10.00",
        f"CLASSIFICATION       : "
        + ("FIRST CLASS WITH DISTINCTION" if p["cgpa"] >= 8.25 else
           "FIRST CLASS" if p["cgpa"] >= 7.0 else "SECOND CLASS"),
        f"GERMAN EQUIVALENT GRADE (modified Bavarian formula): {p['german_equivalent_grade']:.2f}",
        "",
        f"Date of Issue: {fake.date_between(date(p['graduation_year'], 6, 1), date(p['graduation_year'], 12, 1))}",
        "This transcript lists all examination results recorded by the Office of the",
        "Controller of Examinations and is issued for the purpose of higher-studies",
        "applications. Grades once confirmed by the board of examiners are final.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Statement of Purpose (5 paragraphs)
# --------------------------------------------------------------------------

def make_sop(p: dict) -> str:
    prog = PROGRAMS[p["program"]]
    i1, i2 = p["interests"]
    spark_course = random.choice(_semester_courses(prog["code"], 5))
    thesis = f"{i1.title()}-Based {random.choice(['Optimization', 'Monitoring', 'Prediction', 'Classification'])} of {fake.catch_phrase()}"
    tools = ", ".join(random.sample(SKILLS[prog["code"]], 3))
    msc_courses = random.sample(prog["msc_courses"], 2)
    institute = random.choice(prog["institutes"])
    role, task = random.choice(COMPANIES_ROLES)
    company = fake.company()
    activity = random.choice(ACTIVITIES)
    target_role = random.choice(["R&D engineer", "systems architect", "research scientist",
                                 "senior development engineer"])
    longterm = random.choice(
        ["pursue a doctoral degree and contribute to applied research",
         "lead an engineering team building products with real societal impact",
         "drive the adoption of trustworthy, efficient technology in industry"])

    paras = [
        # 1 — Introduction and academic spark
        f"My fascination with {i1} began during my undergraduate studies at "
        f"{p['undergrad_university']}, where I pursued a {p['undergrad_degree']}. While taking "
        f"a course on {spark_course}, I became captivated by how the principles of {i1} can be "
        f"applied to solve tangible engineering problems, and this pivotal experience transformed "
        f"a general curiosity into a dedicated career ambition. I am applying to the "
        f"{p['program']} at the University of Stuttgart, with an intended specialization in "
        f"{p['intended_specialization']}, because I want to master the advanced methodologies "
        f"required to bridge the gap between theoretical knowledge and practical solutions in "
        f"this rapidly evolving field.",

        # 2 — Academic background and achievements
        f"Throughout my academic journey I have intentionally built a robust foundation in the "
        f"core principles of {i1} and {i2}. Alongside maintaining a CGPA of {p['cgpa']}/10 "
        f"(German equivalent grade {p['german_equivalent_grade']}), I actively sought out "
        f"challenges that pushed my intellectual boundaries. My most significant academic "
        f"milestone was my final-year thesis, \"{thesis}\", through which I gained extensive "
        f"hands-on experience with {tools} while refining my ability to analyze complex data, "
        f"design experiments, and draw evidence-based conclusions."
        + (f" This work further led to {p['publications']} peer-reviewed publication(s), which "
           f"taught me the rigor of academic writing and review." if p["publications"] else
           " The project taught me to navigate the trials of independent research and confirmed "
           "my readiness for graduate-level coursework."),

        # 3 — Professional experience and extracurriculars
        f"Complementing my academic work, my practical experiences have given me a realistic "
        f"understanding of industry demands. During my time as a {role} at {company}, I "
        f"{task}, which required me to apply academic insights to live projects under real "
        f"constraints and taught me the value of cross-functional teamwork, tight deadlines and "
        f"adaptable problem-solving."
        + (f" Over {p['work_experience_years']} year(s) of professional experience, I have "
           f"learned to take ownership of technical problems end-to-end." if p["work_experience_years"] else "")
        + f" Additionally, I {activity}, which helped me develop the communication and "
        f"leadership skills necessary to thrive in a diverse, collaborative environment.",

        # 4 — Why Stuttgart / this program
        f"The {p['program']} at the University of Stuttgart is the ideal catalyst for my goals "
        f"due to its rigorous curriculum, specifically the courses in "
        f"{msc_courses[0]} and {msc_courses[1]}, which align directly with my interest in "
        f"{i2}. I am especially eager to engage with the research conducted at the "
        f"{institute}, whose work sits precisely at the intersection of {i1} and {i2}. "
        f"Stuttgart's unique position within one of Europe's strongest engineering and "
        f"high-tech regions, with its close ties between university institutes and industry, "
        f"offers exactly the environment in which I want to grow.",

        # 5 — Future goals and closing
        f"Looking ahead, my immediate goal after graduation is to work as a {target_role} "
        f"where I can apply {i1} to problems that matter. In the long term, I aspire to "
        f"{longterm}. I am confident that my academic record, practical experience and genuine "
        f"enthusiasm for {i1} have prepared me to contribute meaningfully to your program, and "
        f"I would be honored to bring my perspective, dedication and background to the upcoming "
        f"cohort at the University of Stuttgart.",
    ]
    header = f"STATEMENT OF PURPOSE\nApplicant: {p['name']}\nProgram: {p['program']}, University of Stuttgart\n"
    return header + "\n" + "\n\n".join(paras) + "\n"


# --------------------------------------------------------------------------
# Letters of Recommendation (full formal letters, 4-5 paragraphs)
# --------------------------------------------------------------------------

def _lor_header(p: dict, letter_date: date) -> str:
    return (
        f"{letter_date.strftime('%d %B %Y')}\n\n"
        f"Admissions Committee\n"
        f"{p['program']}\n"
        f"University of Stuttgart\n"
        f"Keplerstrasse 7, 70174 Stuttgart, Germany\n\n"
        f"Subject: Letter of Recommendation for {p['name']}\n\n"
        f"Dear Members of the Admissions Committee,\n"
    )


def _lor_footer(ref_name: str, ref_title: str, ref_org: str) -> str:
    return (
        f"\nSincerely,\n\n{ref_name}\n{ref_title}\n{ref_org}\n"
        f"{fake.email()} | {fake.phone_number()}\n"
    )


def make_academic_lor(p: dict) -> str:
    random.seed(p["applicant_id"] + "lor1")
    prog = PROGRAMS[p["program"]]
    ref_name = fake.name()
    field = random.choice(["Computer Engineering", "Electrical Engineering",
                           "Electronics", "Computer Science"])
    ref_title = f"Professor of {field}"
    course1, course2 = random.sample(_semester_courses(prog["code"], 5), 2)
    years_teaching = random.randint(8, 25)
    known_years = random.randint(2, 4)
    project = f"{p['interests'][0].title()} approaches to {fake.bs()}"
    s = p["_strength"]

    standing = {"strong": f"stands out as one of the top 5% of students I have taught in my "
                          f"{years_teaching} years at this institution",
                "average": f"has consistently been a solid, dependable and hardworking student "
                           f"within the top quarter of their cohort",
                "weak": f"showed genuine potential, although their performance across "
                        f"courses was at times inconsistent"}[s]
    academic = {"strong": "possesses a rare combination of analytical rigor and creative "
                          "problem-solving. They routinely demonstrated a deep conceptual "
                          "understanding that went far beyond the standard curriculum, and their "
                          "capstone work was equivalent to that of a seasoned graduate student, "
                          "ultimately earning one of the highest marks in the department.",
                "average": "combines sound analytical skills with steady diligence. While not "
                           "always the first to volunteer an answer, their submitted work was "
                           "consistently thorough, well-structured and delivered on time, and "
                           "their capstone project earned a very respectable grade.",
                "weak": "grasps core concepts adequately when engaged, and their capstone "
                        "project showed flashes of insight. However, I must candidly note that "
                        "their depth of preparation varied between courses, which is reflected "
                        "in an uneven grade profile."}[s]
    collab = {"strong": "naturally stepped into the role of facilitator during group seminars, "
                        "ensuring all team members contributed while keeping the group focused. "
                        "Peers look up to them for clarity of thought and willingness to help "
                        "others with difficult concepts. As my research assistant, they displayed "
                        "meticulous attention to detail and unwavering academic integrity.",
              "average": "worked constructively in group settings and could be relied upon to "
                         "complete their share of shared projects to a good standard. With "
                         "guidance, they steadily improved their presentation and scientific "
                         "writing skills over the terms I supervised them.",
              "weak": "was cooperative in group settings, though they tended to take a "
                      "supporting rather than leading role. When given clear structure and "
                      "mentorship, they responded well and improved noticeably over the term."}[s]
    verdict = {"strong": "my highest, unreserved recommendation",
               "average": "my confident recommendation",
               "weak": "my recommendation, with the reservations noted above"}[s]

    body = [
        f"It is a distinct pleasure to write this letter of recommendation for {p['name']} as "
        f"they apply for admission to the {p['program']} at the University of Stuttgart. As a "
        f"{ref_title} at {p['undergrad_university']}, I have interacted with many promising "
        f"students over the past {years_teaching} years; {p['name']} {standing}. I have known "
        f"them for {known_years} years, during which time they took my courses in {course1} "
        f"and {course2}.",

        f"Academically, {p['name']} {academic} For their final-year assignment they worked on "
        f"a project titled \"{project}\", which investigated how {p['interests'][0]} techniques "
        f"can be applied in a practical engineering context. Their ability to survey the "
        f"literature, formulate a clear hypothesis and execute a sound methodology left a "
        f"lasting impression on the examination committee.",

        f"Beyond individual academic achievement, {p['name']} {collab}",

        f"On a personal level, {p['name']} is a mature and motivated individual who handles "
        f"heavy academic workloads and unexpected setbacks with a calm, positive demeanor. "
        f"They participate actively in departmental activities and show a genuine willingness "
        f"to give back to the academic community. I am confident that they possess the "
        f"intellectual capacity, work ethic and character necessary to thrive in your rigorous, "
        f"English-taught program and to make meaningful contributions to your university.",

        f"Therefore, I give {p['name']} {verdict}. Please feel free to contact me if you "
        f"require any further information.",
    ]
    letter_date = fake.date_between(date(2025, 10, 1), date(2026, 1, 15))
    return (_lor_header(p, letter_date) + "\n" + "\n\n".join(body)
            + _lor_footer(ref_name, ref_title, p["undergrad_university"]))


def make_industry_lor(p: dict) -> str:
    random.seed(p["applicant_id"] + "lor2")
    ref_name = fake.name()
    company = fake.company()
    ref_title = random.choice(["Engineering Manager", "Senior Team Lead",
                               "Head of Software Development", "R&D Group Leader"])
    role, task = random.choice(COMPANIES_ROLES)
    months = random.choice([3, 6, 9, 12])
    s = p["_strength"]
    tech = ", ".join(random.sample(SKILLS[PROGRAMS[p["program"]]["code"]], 3))

    impact = {"strong": "consistently delivered beyond the scope of their role. They "
                        "independently identified process improvements, and one of their "
                        "contributions was adopted by the whole team as standard practice. Their "
                        "code reviews were of a quality I would expect from engineers with "
                        "several years of experience.",
              "average": "reliably delivered the tasks assigned to them and grew visibly during "
                         "their time with us. They asked the right questions, incorporated "
                         "feedback quickly, and by the end of their tenure required minimal "
                         "supervision on routine tasks.",
              "weak": "completed the majority of their assigned tasks satisfactorily. They "
                      "occasionally needed additional guidance on complex work, but their "
                      "attitude remained positive and they showed clear improvement toward the "
                      "end of their engagement."}[s]
    teamwork = {"strong": "became a person colleagues actively sought out for help, and "
                          "bridged communication between our software and hardware groups with "
                          "notable ease",
                "average": "integrated smoothly into our team and communicated their progress "
                           "clearly in stand-ups and reviews",
                "weak": "was pleasant and cooperative, and benefited from the structured "
                        "mentoring our team provides"}[s]
    verdict = {"strong": "without any reservation, and I would rehire them immediately",
               "average": "with confidence",
               "weak": "believing that the structured environment of a Master's program will "
                       "help them realize their potential"}[s]

    body = [
        f"I am writing to recommend {p['name']} for admission to the {p['program']} at the "
        f"University of Stuttgart. In my capacity as {ref_title} at {company}, I directly "
        f"supervised {p['name']} for {months} months during their engagement as {role} in our "
        f"organization.",

        f"During this period, {p['name']} {task}, working primarily with {tech}. In terms of "
        f"delivery and impact, they {impact}",

        f"What distinguished {p['name']} in a professional environment was their combination "
        f"of technical curiosity and dependability. They {teamwork}. They also documented "
        f"their work carefully, a habit that made handover at the end of their engagement "
        f"seamless and that speaks to the professional maturity your program values.",

        f"I understand that the {p['program']} is demanding and strongly research-oriented. "
        f"Based on my direct observation of {p['name']}'s ability to learn new technologies "
        f"quickly and to function effectively under real project constraints, I believe they "
        f"are well prepared for this challenge and will represent your institution well in "
        f"internships and industry collaborations.",

        f"I therefore recommend {p['name']} {verdict}. Please do not hesitate to contact me "
        f"for any further details.",
    ]
    letter_date = fake.date_between(date(2025, 10, 1), date(2026, 1, 15))
    return (_lor_header(p, letter_date) + "\n" + "\n\n".join(body)
            + _lor_footer(ref_name, ref_title, company))


# --------------------------------------------------------------------------
# CV (detailed)
# --------------------------------------------------------------------------

def make_cv(p: dict) -> str:
    random.seed(p["applicant_id"] + "cv")
    prog = PROGRAMS[p["program"]]
    code = prog["code"]
    skills = random.sample(SKILLS[code], 8)
    i1, i2 = p["interests"]

    lines = [
        p["name"].upper(),
        f"{p['email']} | {fake.phone_number()} | {fake.city()}, {p['nationality']}",
        f"Date of Birth: {p['date_of_birth']} | LinkedIn: linkedin.com/in/{p['name'].lower().replace(' ', '-')}",
        "",
        "PROFESSIONAL SUMMARY",
        "-" * 70,
        f"{p['undergrad_degree']} graduate ({p['graduation_year']}) with a focus on {i1} and "
        f"{i2}, holding a CGPA of {p['cgpa']}/10 (German equivalent {p['german_equivalent_grade']}). "
        f"Experienced through {p['internships']} internship(s)"
        + (f" and {p['work_experience_years']} year(s) of professional work" if p["work_experience_years"] else "")
        + f", with hands-on skills in {', '.join(skills[:4])}. Applying for the {p['program']} "
        f"at the University of Stuttgart (intended specialization: {p['intended_specialization']}).",
        "",
        "EDUCATION",
        "-" * 70,
        f"{p['graduation_year'] - 4} - {p['graduation_year']}   {p['undergrad_degree']}",
        f"                {p['undergrad_university']}",
        f"                CGPA: {p['cgpa']}/10 | German equivalent grade: {p['german_equivalent_grade']}",
        "",
        "EXPERIENCE",
        "-" * 70,
    ]

    n_jobs = max(1, p["internships"] + (1 if p["work_experience_years"] else 0))
    year = 2026
    for _ in range(min(n_jobs, 4)):
        role, task = random.choice(COMPANIES_ROLES)
        company = fake.company()
        dur = random.choice([3, 6, 9, 12])
        end_m, start_m = random.randint(1, 12), 0
        start_y = year - (1 if end_m - dur < 1 else 0)
        start_m = (end_m - dur - 1) % 12 + 1
        lines += [
            f"{start_m:02d}/{start_y} - {end_m:02d}/{year}   {role}, {company}",
            f"    - {task[0].upper() + task[1:]}.",
            f"    - {random.choice(['Collaborated with cross-functional engineering teams in an agile environment', 'Documented modules and validation procedures following engineering standards', 'Automated testing and reporting workflows, improving team productivity', 'Presented results to stakeholders and incorporated iterative feedback'])}.",
            f"    - {random.choice(['Improved processing performance through profiling and optimization', 'Built monitoring dashboards for continuous data acquisition', 'Validated the solution against real datasets from production systems', 'Containerized the workflow for reproducible deployment'])}.",
            "",
        ]
        year = start_y

    thesis_tools = ", ".join(random.sample(SKILLS[code], 3))
    lines += [
        "PROJECTS",
        "-" * 70,
        f"Final-Year Thesis: {i1.title()}-Based {random.choice(['Monitoring', 'Prediction', 'Optimization'])} System",
        f"    - Designed and implemented an end-to-end prototype using {thesis_tools}.",
        f"    - Evaluated the approach on benchmark data and documented results in a formal thesis"
        + (f"; findings contributed to {p['publications']} publication(s)." if p["publications"] else "."),
        f"{random.choice(['Smart Campus IoT Platform', 'Real-Time Anomaly Detection Pipeline', 'Autonomous Line-Following Robot', 'Energy Consumption Forecasting Tool', 'Gesture-Controlled Embedded Interface'])}",
        f"    - Team project applying {i2} concepts; responsible for "
        f"{random.choice(['system architecture and integration', 'data pipeline and evaluation', 'firmware and hardware bring-up', 'model training and deployment'])}.",
        "",
        "TECHNICAL SKILLS",
        "-" * 70,
        f"Programming & Tools : {', '.join(skills)}",
        f"Domains             : {i1}, {i2}, software engineering practices (Git, code review, CI)",
        f"Languages           : English ({'C1' if p['_strength'] != 'weak' else 'B2'}, "
        f"{p['english_test']} {p['english_score']}), German (A1-A2, learning)",
        "",
        "ACTIVITIES & LEADERSHIP",
        "-" * 70,
        f"    - {random.choice(ACTIVITIES)[0].upper() + random.choice(ACTIVITIES)[1:]}.",
        f"    - {random.choice(ACTIVITIES)[0].upper() + random.choice(ACTIVITIES)[1:]}.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

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
        (d / "lor_1.txt").write_text(make_academic_lor(p))
        (d / "lor_2.txt").write_text(make_industry_lor(p))
        (d / "cv.txt").write_text(make_cv(p))
        (d / "transcript.txt").write_text(make_transcript(p))
        print(f"  {p['applicant_id']}  {PROGRAMS[p['program']]['code']:<9} "
              f"CGPA {p['cgpa']:.2f} (DE {p['german_equivalent_grade']:.2f})  [{p['_strength']}]")
    print(f"\nGenerated {n} applicants in {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.n, args.out, args.seed)