from spacy.matcher import Matcher
import utils as utils
import spacy
# init params of skill extractor
nlp = spacy.load("en_core_web_lg")
import os

resume_dir = './Dataset/Resume files'

outdir = './Dataset'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def parser(resume_file):
        nlp = spacy.load('en_core_web_sm')
        matcher = Matcher(nlp.vocab)
        details = {
            'name'              : None,
            'email'             : None,
            'mobile_number'     : None,
            'designation'       : None,
            'skills'            : None,
            'degrees'           : None,
            'experience'        : None,
            'majors'            : None,
        }
        text_raw    = utils.extract_text_from_file(resume_file)
        text        = ' '.join(text_raw.split())
        __nlp         = nlp(text)
        name       = utils.extract_name(__nlp, matcher=matcher)
        email      = utils.extract_email(text)
        mobile     = utils.extract_mobile_number(text)
        skills     = utils.extract_skills(text_raw)
        degrees        = utils.extract_degrees(text_raw)
        experience = utils.extract_experience(text)
        majors   = utils.extract_majors(text_raw)
        designation=utils.extract_designation(text)
        details['name'] = name
        details['email'] = email
        details['mobile_number'] = mobile
        details['skills'] = skills
        details['degrees'] = degrees
        details['majors'] = majors
        details['experience'] = experience
        details['designation'] = designation

        return details

def main():
     resume_file="ASH1825022M-Rahat Uddin Azad.pdf"
     details=parser(os.path.join(resume_dir, resume_file))
     skills_weight=0.4
     degree_weight=0.15
     majors_weight=0.15
     designation_weight=0.1
     experience_weight=0.1
     print(details)
     return details
 
if __name__ == '__main__':
      main()
      print('-----------Resume Info Extraction from resume is complete.-----------')
    