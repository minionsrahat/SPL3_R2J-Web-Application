from pyresparser import ResumeParser
import os

outdir = './Dataset/Resume files'

def extract_attributes(data):
    skills=(','.join(word for word in data['skills']))
    return skills

def parser(resume_file):
    print("Resume Name"+resume_file)
    data=ResumeParser(os.path.join(outdir,resume_file)).get_extracted_data()
    skills=extract_attributes(data)
    return skills

def main():
     resume_file=""
     return parser(resume_file)
 
if __name__ == '__main__':
      main()
      print('-----------Skill Extraction from resume is complete.-----------')
    