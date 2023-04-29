import pandas as pd
from flask import Flask, render_template, redirect, request
import os
import json
from resume_screening import resume_parser
from recommendation import R2J
from flask_cors import CORS


outdir = './Dataset/Resume files'
outdirforcsv = './Dataset/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template('about.html')


@app.route("/home")
def home():
    return redirect('/')


@app.route('/submitFile', methods=['POST'])
def submit_file():
    print("Got the request")
    if request.method == 'POST':
        f = request.files['resume_file']
        f.save(os.path.join(outdir, f.filename))

    # Parse Resume
    resume_details= resume_parser.parser(f.filename)
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




@app.route('/submit', methods=['POST'])
def submit_data():
    if request.method == 'POST':
        f = request.files['userfile']
        f.save(os.path.join(outdir, f.filename))

    # Parse Resume
    resume_text = resume_parser.parser(f.filename)
    print(resume_text)

    # Job Vacancy Recommendations
    recommendation_response = R2J.get_recommendations(resume_text)
    recommend_jobs = recommendation_response['recommend_jobs']
    # recommend_jobs=R2J.get_recommendations(resume_text)
    scraped_jobs = pd.read_csv(os.path.join(
        outdirforcsv, 'Clustered Jobs.csv'))
    scraped_jobs.set_index(scraped_jobs.position, inplace=True)
    recommend_jobs = recommend_jobs.join(scraped_jobs)
    recommend_jobs = recommend_jobs.sort_values(
        'score', ascending=False).reset_index(drop=True).head(20)
    column_names = ['score', 'position', 'company', 'link']
    return render_template('index.html', column_names=column_names, row_data=list(recommend_jobs.values.tolist()),
                           link_column="link", zip=zip)


if __name__ == "__main__":
    app.run(debug=True)
