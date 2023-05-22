import pandas as pd
from flask import Flask, render_template, redirect, request
import os
import json
from resume_screening import resume_parser
from recommendation import R2J
from recommendation import search_jobs
from recommendation import resume_filtering
from flask_cors import CORS
from jobs_screening import jobinfoextraction


outdir = './Dataset/Resume files'
outdirforcsv = './Dataset/'
resume_excelsheet_dir = './Dataset/Resume files/ResumeExcelSheet'
if not os.path.exists(outdir):
    os.mkdir(outdir)

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/get_jobs",methods=['GET'])
def get_jobs():

    response={
        'jobs':search_jobs.get_cluster_wise_jobs()
    }
    json_response = json.dumps(response)
    return json_response



@app.route("/parse_resume_excel_sheet",methods=['POST'])
def parse_resume_excel_sheets():
    if request.method == 'POST':
        f = request.files['excel_file']
        f.save(os.path.join(resume_excelsheet_dir, f.filename))
    # Parse Resume
    df= resume_parser.resume_excelsheet_parser(f.filename)
    resumes=df.to_dict(orient='records')
    response={
        'resumes':resumes
    }
    json_response = json.dumps(response)
    print(json_response)
    return json_response

@app.route("/sort_resumes",methods=['POST'])
def sort_resumes():
    if request.method == 'POST':
         data = request.get_json()
         resumes = json.loads(data['resume'])
         jobdes=data['job_desc']
    # Parse Resume
    resumes=pd.DataFrame(resumes)
    job={}
    job['Acceptable majors']=jobinfoextraction.match_majors_by_spacy(jobdes)
    job['degrees']=jobinfoextraction.match_degrees_by_spacy(jobdes)
    job['Minimum degree level']=jobinfoextraction.get_minimum_degree(job['degrees'])
    job['skills']=jobinfoextraction.match_skills_by_spacy(jobdes)
    sorted_resumes=resume_filtering.resume_filtering(resumes,job,[0.8,0.1,0.1])
    response={
        'sorted_resumes':sorted_resumes.to_dict(orient='records')
    }
    json_response = json.dumps(response)
    return json_response


@app.route("/extract_resume_info",methods=['POST'])
def extract_resume_info():
    if request.method == 'POST':
        f = request.files['resume_file']
        f.save(os.path.join(outdir, f.filename))

    # Parse Resume
    resume_details= resume_parser.parser(f.filename)
    resume_score=resume_parser.resume_score(resume_details)
    print("Resume Score:" ,str(resume_score))
    response={
        'resume':resume_details,
        'score':resume_score
    }
    json_response = json.dumps(response)
    return json_response

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    print("Got the request")
    if request.method == 'POST':
         data = request.get_json()
         resume_details = json.loads(data['resume'])
    # Job Vacancy Recommendations
    recommendation_response = R2J.give_recommendations(resume_details,[0.8,0.1,0.1])
    # Convert dictionary to JSON string
    response = {
        'resume_text': resume_details['skills'],
        # 'recommendation': recommendation_response['recommend_jobs'],
        'all_cluster_jobs': recommendation_response['all_cluster_jobs'],
        'total_cluster': recommendation_response['total_cluster'],
        'predicted_cluster': recommendation_response['predicted_cluster_no']
    }
    json_response = json.dumps(response)
    return json_response



if __name__ == "__main__":
    app.run(debug=True)
