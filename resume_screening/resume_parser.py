from pyresparser import ResumeParser
import docx2txt
import PyPDF2
import spacy
from spacy.matcher import PhraseMatcher
from spacy.lang.en import English
# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor
# init params of skill extractor
nlp = spacy.load("en_core_web_lg")
import os
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
outdir = './Dataset'
if not os.path.exists(outdir):
    os.mkdir(outdir)


def match_majors_by_spacy(job):
        majors_patterns_path = os.path.join(outdir, 'Resources/data/majors.jsonl') 
        nlp = English()
        # Add the pattern to the matcher
        patterns_path = majors_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        acceptable_majors = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'MAJOR':
                if labels_parts[2].replace('-', ' ') not in acceptable_majors:
                    acceptable_majors.append(labels_parts[2].replace('-', ' '))
                if labels_parts[2].replace('-', ' ') not in acceptable_majors:
                    acceptable_majors.append(labels_parts[2].replace('-', ' '))
        return acceptable_majors

def match_degrees_by_spacy(job):
        degrees_patterns_path = os.path.join(outdir, 'Resources/data/degrees.jsonl') 
        nlp = English()
        # Add the pattern to the matcher
        patterns_path = degrees_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        degree_levels = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'DEGREE':
                if labels_parts[1] not in degree_levels:
                    degree_levels.append(labels_parts[1])
        return degree_levels

def match_skills_by_spacy(job):
        skills_patterns_path =os.path.join(outdir, 'Resources/data/skills.jsonl') 
        nlp = English()
        patterns_path = skills_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        job_skills = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'SKILL':
                if labels_parts[1].replace('-', ' ') not in job_skills:
                    job_skills.append(labels_parts[1].replace('-', ' '))
        return ','.join(str(x) for x in job_skills)

def extract_attributes(data):
    skills=(','.join(word for word in data['skills']))
    return skills

# Extract text from .docx file
def extract_text_from_docx(file_path):
    text = docx2txt.process(file_path)
    return text

# Extract text from .pdf file
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfFileReader(pdf_file)
        text = ''
        for page in range(reader.getNumPages()):
            page_obj = reader.getPage(page)
            text += page_obj.extractText()
        return text
    
# Extract text from .txt file
def extract_text_from_txt(file_path):
    with open(file_path, 'r') as txt_file:
        text = txt_file.read()
        return text

def get_full_match(annotations):
    skills=set()
    full_match=annotations['results']['full_matches']
    for dict_item in full_match:
        skills.add(dict_item['doc_node_value'])
    return skills

def get_sub_matches(annotations):
    skills=set()
    sub_match=annotations['results']['ngram_scored']
    for dict_item in sub_match:
        if dict_item['score'] >=0.6:
            skills.add(dict_item['doc_node_value'])  
    return skills

def extract_skills_by_extractor(des):
    try:
        annotations = skill_extractor.annotate(des)
        full_match=get_full_match(annotations)
        print("Full Match:",len(full_match))
        sub_match=get_sub_matches(annotations)
        print("Sub Match:",len(sub_match))
        full_match=full_match.union(sub_match)
        return ','.join(str(x) for x in full_match)
    except:
        return " "

def extract_skills_by_spacy(text):
    resume={}
    resume['majors']=(match_majors_by_spacy(text))
    resume['degrees']=(match_degrees_by_spacy(text))
    resume['skill']=(match_skills_by_spacy(text))
    return resume;

# Extract text from file based on file type
def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f'Unsupported file type: {file_extension}')

def parser(resume_file):
    print("Resume Name"+resume_file)
    resume_text = extract_text_from_file(os.path.join(outdir,"Resume files/"+resume_file))
    resume_skills=extract_skills_by_extractor(resume_text)
    return resume_skills

def main():
     resume_file=""
     return parser(resume_file)
 
if __name__ == '__main__':
      main()
      print('-----------Skill Extraction from resume is complete.-----------')
    