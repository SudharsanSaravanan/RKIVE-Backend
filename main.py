# 1. IMPORTS
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ortools.sat.python import cp_model
import io
import json
import re
import os # Added for path handling

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conint
from typing import Dict, List

# 2. INITIALIZE APP & LOAD MODEL
app = FastAPI(
    title="Optimized Candidate Selection API",
    description="An API to score candidates and provide a final shortlist. It outputs two CSV files: 'scored-candidates.csv' and 'shortlisted-candidates.csv'.",
    version="1.1.0"
)

print("Loading NER model...")
try:
    ner_pipeline = pipeline(
        "ner",
        model="Shrav20/job-ner-deberta",
        tokenizer="Shrav20/job-ner-deberta",
        aggregation_strategy="simple"
    )
    print("NER model loaded successfully.")
except Exception as e:
    print(f"Error loading NER model: {e}")
    ner_pipeline = None

# 3. HELPER FUNCTIONS (No changes here, same as before)
def score_and_filter_candidates(
    job_description: str,
    df: pd.DataFrame,
    job_requirements: dict
) -> pd.DataFrame:
    """
    Takes raw candidate data and a job description, then filters and calculates a 'final_score'.
    """
    if not ner_pipeline:
        raise HTTPException(status_code=500, detail="NER model is not available.")
    
    # --- Part 1a: Filter candidates based on basic requirements ---
    def extract_education(education_str):
        if pd.isna(education_str): return None
        education_str = str(education_str).strip()
        if re.search(r'B\.?Tech|B\.?E\.?|Bachelor.*Technology|Bachelor.*Engineering', education_str, re.IGNORECASE):
            return "B.Tech"
        if re.search(r'B\.?Sc|B\.?S\.?|Bachelor.*Science', education_str, re.IGNORECASE):
            return "B.Sc"
        return education_str

    # Ensure required columns exist to prevent KeyErrors
    required_cols = ['UG_Course', 'Skills', '10th_score', '12th_score', 'Undergraduate_Score']
    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing required column in candidates CSV: '{col}'")


    df['extracted_education'] = df['UG_Course'].apply(extract_education)
    education_filter = df['extracted_education'].isin(job_requirements.get('education', []))
    eligible_df = df[education_filter].drop(columns=['extracted_education']).copy()

    if eligible_df.empty:
        return pd.DataFrame() 

    # --- Part 1b: Score candidates based on skill similarity ---
    results = ner_pipeline(job_description)
    job_skills = [r["word"] for r in results if r["entity_group"].lower() == "skill"]
    job_skills_text = " ".join(job_skills)

    similarity_scores = []
    vectorizer = TfidfVectorizer()
    for _, row in eligible_df.iterrows():
        candidate_skills_text = str(row.get("Skills", "")).replace(",", " ")
        if not candidate_skills_text.strip() or not job_skills_text.strip():
            similarity_scores.append(0.0)
            continue
        tfidf_matrix = vectorizer.fit_transform([job_skills_text, candidate_skills_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        similarity_scores.append(similarity)

    eligible_df["similarity_score"] = similarity_scores
    
    eligible_df["final_score"] = (
        eligible_df["10th_score"]/500 + 
        eligible_df["12th_score"]/500 + 
        eligible_df["Undergraduate_Score"]/10 + 
        eligible_df["similarity_score"]
    )
    
    eligible_df['final_score'] = eligible_df['final_score'].fillna(0)
    eligible_df.reset_index(drop=True, inplace=True)
    
    return eligible_df

def optimize_selection(
    scored_df: pd.DataFrame,
    total_selections: int,
    quotas: dict
) -> pd.DataFrame:
    """
    Takes the scored candidates and applies OR-Tools to find the optimal selection.
    """
    model = cp_model.CpModel()
    num_candidates = len(scored_df)
    
    if num_candidates == 0:
        return pd.DataFrame()

    x = [model.NewBoolVar(f'candidate_{i}') for i in range(num_candidates)]

    model.Add(sum(x) <= total_selections)

    penalty_terms = []
    # Use .get(category, pd.Series()) to avoid error if quota column not in df
    for category, rules in quotas.items():
        if category not in scored_df.columns:
            print(f"Warning: Quota category '{category}' not found in candidate columns. Skipping this quota.")
            continue
        for value, details in rules.items():
            target = details.get("Target", 0)
            penalty = details.get("Penalty", 0)
            shortfall = model.NewIntVar(0, target, f'shortfall_{category}_{value}')
            
            count_of_selected = sum(x[i] for i, cand in scored_df.iterrows() if cand[category] == value)
            model.Add(count_of_selected + shortfall >= target)
            penalty_terms.append(penalty * shortfall)

    final_scores = scored_df['final_score'].tolist()
    # Multiply by 1000 and cast to int to handle floats in the solver
    score_term = sum(int(final_scores[i] * 1000) * x[i] for i in range(num_candidates))

    total_penalty_term = sum(penalty_terms * 1000) # Scale penalty to match score
    model.Maximize(score_term - total_penalty_term)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected_indices = [i for i in range(num_candidates) if solver.Value(x[i]) == 1]
        return scored_df.iloc[selected_indices]
    else:
        return pd.DataFrame()


# 4. API ENDPOINT (Updated Logic)
@app.post("/select-candidates/")
async def create_selection(
    total_selections: int = Form(..., description="The total number of candidates to select."),
    quotas: str = Form(..., description='JSON string defining quotas. E.g., `{"Category": {"SC": {"Target": 2, "Penalty": 10}}, "Gender": {"Female": {"Target": 1, "Penalty": 10}}}`'),
    job_requirements: str = Form(..., description='JSON string for basic filters. E.g., `{"education": ["B.Tech", "B.Sc"]}`'),
    job_description_file: UploadFile = File(..., description="A .txt file containing the job description."),
    candidates_file: UploadFile = File(..., description="A .csv file with candidate profiles.")
):
    """
    Processes candidate data to produce two files:
    1.  `scored-candidates.csv`: All eligible candidates, sorted by score.
    2.  `shortlisted-candidates.csv`: The final optimized selection.
    """
    # --- Input Validation and Parsing ---
    if candidates_file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file type for candidates. Please upload a CSV.")
    if job_description_file.content_type != 'text/plain':
        raise HTTPException(status_code=400, detail="Invalid file type for job description. Please upload a TXT file.")

    try:
        quotas_dict = json.loads(quotas)
        job_req_dict = json.loads(job_requirements)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for quotas or job_requirements.")

    # --- Read Input Files ---
    job_description_content = (await job_description_file.read()).decode("utf-8")
    candidates_content = await candidates_file.read()
    candidates_df = pd.read_csv(io.StringIO(candidates_content.decode("utf-8")))

    # --- Part 1: Score all candidates ---
    scored_candidates_df = score_and_filter_candidates(
        job_description=job_description_content,
        df=candidates_df,
        job_requirements=job_req_dict
    )

    if scored_candidates_df.empty:
        return {"message": "No eligible candidates found after initial filtering.", "outputs": {}}

    # --- Sort and save the full scored list ---
    scored_candidates_df = scored_candidates_df.sort_values(by='final_score', ascending=False)
    scored_filename = "scored-candidates.csv"
    scored_candidates_df.to_csv(scored_filename, index=False)
    print(f"Successfully saved all scored candidates to '{scored_filename}'")


    # --- Part 2: Get the optimized selection ---
    final_selection_df = optimize_selection(
        scored_df=scored_candidates_df,
        total_selections=total_selections,
        quotas=quotas_dict
    )

    # --- Save the final shortlisted candidates ---
    shortlisted_filename = "shortlisted-candidates.csv"
    if not final_selection_df.empty:
        final_selection_df.to_csv(shortlisted_filename, index=False)
        print(f"Successfully saved shortlisted candidates to '{shortlisted_filename}'")
        message = "Optimal selection completed successfully."
    else:
        # Create an empty file if no solution was found
        pd.DataFrame().to_csv(shortlisted_filename, index=False)
        message = "An optimal selection could not be found. 'shortlisted-candidates.csv' is empty."


    # --- Format and Return Comprehensive JSON Response ---
    shortlisted_json = json.loads(final_selection_df.to_json(orient="records"))
    scored_json = json.loads(scored_candidates_df.to_json(orient="records"))
    
    return {
        "message": message,
        "outputs": {
            "scored_candidates_file": os.path.abspath(scored_filename),
            "shortlisted_candidates_file": os.path.abspath(shortlisted_filename)
        },
        "shortlisted_candidates": shortlisted_json,
        "all_scored_candidates": scored_json
    }