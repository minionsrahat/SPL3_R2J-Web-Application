import pandas as pd
import scraping;
import os

filename = 'Scraped Jobs.csv'
outdir = './Dataset'
if not os.path.exists(outdir):
    os.mkdir(outdir)
fullname = os.path.join(outdir, filename)    

def get_jobs_links(keywords,limit,jobsite):
    job_links=[]
    for i in range(len(keywords)):
        temp_job=[]
        if(jobsite=='carrierbuilder'):
            temp_job=scraping.get_job_links_from_carrierbuilder(keywords[i],limit)
        job_links.extend(temp_job)
    return list(set(job_links))

def parse_job_links(job_links):
    job_details=[]
    i=1
    for link in job_links:
        job_details.append(scraping.parse_job_carrier_builder(link,i))
        i=i+1
    return job_details

def save_processed_dataframe(job_details):
    df=pd.DataFrame(job_details)
    df=(df.sort_values(by='description', ascending=False)).drop_duplicates(subset='description').reset_index(drop=True)
    df.to_csv(fullname)

def main():
      keywords = ["Software Developer", "Full Stack Developer","Backend Developer", 
      "Frontend Developer", "Web Developer", "Mobile App Developer","Cloud Developer", 
      "DevOps Engineer", "Systems Engineer", "Software Architect","IoT Engineer",
      "Software Quality Engineer", "Software Requirements Engineer",
      "IT Cybersecurity Analyst","Business Systems Analyst", "Business Intelligence Analyst",
      "QA Engineer","Data Architect", "Data Mining Engineer","Data Analyst" ]
      job_links = get_jobs_links(keywords,3,'carrierbuilder')
      print("Total Job Links:",len(job_links))
      print(job_links)
    #   job_details=parse_job_links(job_links)
    #   save_processed_dataframe(job_details)

if __name__ == '__main__':
      main()
      print('-----------Extraction of data is complete. Check the csv file.-----------')


    


