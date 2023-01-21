import pandas as pd
from flask import Flask,render_template,redirect,request
import os
from resume_screening import resume_parser
from recommendation import R2J



outdir = './Dataset/Resume files'
outdirforcsv='./Dataset/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

app=Flask(__name__)

@app.route('/') 
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/home")
def home():
    return redirect('/')


@app.route('/submit',methods=['POST'])
def submit_data():
    if request.method == 'POST':        
        f=request.files['userfile']
        f.save(os.path.join(outdir, f.filename))
    
    #Parse Resume
    resume_text=resume_parser.parser(f.filename)

    #Job Vacancy Recommendations
    recommend_jobs=R2J.get_recommendations(resume_text)
    scraped_jobs=pd.read_csv(os.path.join(outdirforcsv,'Clustered Jobs.csv'))
    scraped_jobs.set_index(scraped_jobs.position,inplace=True)
    recommend_jobs=recommend_jobs.join(scraped_jobs)
    recommend_jobs=recommend_jobs.sort_values('score',ascending=False).reset_index(drop=True).head(20)
    column_names=['score','position','company','link']
    return render_template('index.html', column_names=column_names, row_data=list(recommend_jobs.values.tolist()),
                           link_column="link", zip=zip)

if __name__ =="__main__":
    app.run(debug=True)