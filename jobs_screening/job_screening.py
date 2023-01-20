import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
import pandas as pd
import os

filename = 'Skill Extract.csv'
outdir = './Dataset'
if not os.path.exists(outdir):
    os.mkdir(outdir)
fullname = os.path.join(outdir, filename) 

nlp = spacy.load("en_core_web_lg")
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

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

def extract_skills(des):
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


def main():
    df=pd.read_csv(os.path.join(outdir, 'Scraped Jobs.csv') )
    df['skill']=df['description'].apply(extract_skills)
    df.to_csv(fullname)

      

if __name__ == '__main__':
      main()
      print('-----------Skill Extraction of job data is complete. Check the csv file.-----------')