from spacy.lang.en import English
import os
import pandas as pd

DEGREES_IMPORTANCE = {'high school': 0, 'associate': 1, 'BS-LEVEL': 2, 'MS-LEVEL': 3, 'PHD-LEVEL': 4}
ENTITIES = ['BS-LEVEL', 'MS-LEVEL', 'PHD-LEVEL', 'DEV', 'AI', 'CODING', 'DATA SCIENCES', 'AUTOMATION', 'BIG DATA',
            'WEB-DEVELOPMENT', 'MOBILE-DEVELOPMENT']
dataset = 'Skill Extract.csv'
outdir = './Dataset/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
dataset = os.path.join(outdir, dataset) 

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


def get_minimum_degree(degrees):
    if len(degrees) != 0:
        """get the minimum degree that the candidate has"""
        d = {degree: DEGREES_IMPORTANCE[degree] for degree in degrees}
        return min(d, key=d.get)
    else:
        return ''
    
def main():
    df=pd.read_csv(dataset)
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    df=df.reset_index(drop=True)
    df['Acceptable majors']=df['description'].apply(match_majors_by_spacy)
    df['Minimum degree level']=df['description'].apply(match_degrees_by_spacy)
    df['Minimum degree level']=df['Minimum degree level'].apply(get_minimum_degree)
    df.to_csv(os.path.join(outdir,'Extracted Jobs Info.csv') )

if __name__ == '__main__':
      main()
      print('-----------Jobs Info Extracted Check the csv file.-----------')


