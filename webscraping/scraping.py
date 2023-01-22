import requests
from bs4 import BeautifulSoup
import urllib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,'
                  'application/signed-exchange;v=b3;q=0.9',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}

def get_job_links_from_carrierbuilder(keyword,limit):
      base_url="https://www.careerbuilder.com"
      url=base_url+"/jobs?"
      links = []
      params={
         'keywords':keyword,
         'page_number':1
      }
      for i in range(0, limit):
            extention = ""
            if i != 0:
                  params['page_number']=params['page_number']+1
            url_to_scrape = url + urllib.parse.urlencode(params)
            print(url_to_scrape)
            r = requests.get(url_to_scrape,headers=headers)
            soup = BeautifulSoup(r.content, 'html.parser')
            joblisting = soup.select('.data-results-content-parent')
            print(len(joblisting))
            for job in joblisting:
                if job.select('a.data-results-content'):
                    job_link = job.select_one('a.data-results-content')['href']
                    complete_link = base_url + job_link
                    links.append(complete_link)
      return links

def parse_job_carrier_builder(url,index):
    try:
        print(index)
        # session = requests.Session()
        # retry = Retry(connect=3, backoff_factor=0.5)
        # adapter = HTTPAdapter(max_retries=retry)
        # session.mount('http://', adapter)
        # session.mount('https://', adapter)
        r = requests.get(url,verify=False, timeout=5)
        soup = BeautifulSoup(r.content, 'html.parser')
        title ="R2J_"+str(index)+"_"+soup.select_one('.jdp_title_header').getText().strip()
        desc = soup.select_one('.jdp-description-details>.col-2>.jdp-left-content').getText().strip()
        company = soup.select_one('.data-details').getText().strip()
        jobs = {
        'position': title,
            'company': company,
            'description': desc,
            'link' : url,
            }
        return jobs
    except:
        time.sleep(3)
        print("Not Found")


