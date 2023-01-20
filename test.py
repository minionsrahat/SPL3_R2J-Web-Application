from resume_screening import resume_parser
from recommendation import R2J

def main():
     resume_file="ASH1825022M-Rahat Uddin Azad.pdf"
     resume_text=resume_parser.parser(resume_file)
     print(resume_text)
    #Job Vacancy Recommendations
     recommend_jobs=R2J.get_recommendations(resume_text)
     print(recommend_jobs)
 
if __name__ == '__main__':
      main()
      print('-----------Skill Extraction from resume is complete.-----------')