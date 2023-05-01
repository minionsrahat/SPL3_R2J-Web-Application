import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from resume_screening import utils
import os
import spacy
from spacy.matcher import Matcher

# init params of skill extractor
nlp = spacy.load("en_core_web_lg")

resume_dir = './Dataset/Resume files'

outdir = './Dataset'
if not os.path.exists(outdir):
    os.mkdir(outdir)


def parser(resume_file):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    details = {
        'name': None,
        'email': None,
        'mobile_number': None,
        'designation': None,
        'skills': None,
        'degrees': None,
        'experience': None,
        'majors': None,
    }
    text_raw = utils.extract_text_from_file(
        os.path.join(resume_dir, resume_file))
    text = ' '.join(text_raw.split())
    __nlp = nlp(text)
    name = utils.extract_name(__nlp, matcher=matcher)
    email = utils.extract_email(text)
    mobile = utils.extract_mobile_number(text)
    skills = utils.extract_skills(text_raw)
    degrees = utils.extract_degrees(text_raw)
    experience = utils.extract_experience(text)
    majors = utils.extract_majors(text_raw)
    designation = utils.extract_designation(text)
    details['name'] = name
    details['email'] = email
    details['mobile_number'] = mobile
    details['skills'] = skills
    details['degrees'] = degrees
    details['majors'] = majors
    details['experience'] = experience
    details['designation'] = designation

    return details


def resume_score(resume_details):
    weight = {
        'designation': 0.1,
        'skills': 0.6,
        'degrees': 0.1,
        'experience': 0.1,
        'majors': 0.1,
    }
    total_weight = 0
    if resume_details['skills']:
        skills = resume_details['skills'].split(',')
        num_skills = len(skills)
        if num_skills > 25:
            total_weight += 0.6
        elif num_skills > 20:
            total_weight += 0.5
        elif num_skills > 15:
            total_weight += 0.4
        elif num_skills > 10:
            total_weight += 0.3
        elif num_skills > 5:
            total_weight += 0.2
        else:
            total_weight += 0.1
    if len(resume_details['degrees']) > 0:
        total_weight += weight['degrees']
    if len(resume_details['majors']) > 0:
        total_weight +=  weight['majors']
    if len(resume_details['experience']) > 0:
        total_weight += weight['experience']
    if len(resume_details['designation']) > 0:
        total_weight +=  weight['designation']
    return total_weight


def main():
    resume_file = "CV2.pdf"
    details = parser(resume_file)
    print(details)
    print("Resume Score:" +str(resume_score(details)))


if __name__ == '__main__':
    main()
    print('-----------Resume Info Extraction from resume is complete.-----------')
