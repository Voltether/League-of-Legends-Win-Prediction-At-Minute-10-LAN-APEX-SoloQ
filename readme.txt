This project uses Python 3.10+ and Jupyter Notebook.


All requirements are listed in requirements.txt (generated from a raw pip freeze).
Create a .env file with:
RIOT_API_KEY=YOUR_API_KEY

and provide your Google Cloud Service Account credentials as a JSON file (do not commit this file). Point to it via an environment variable, e.g.:
GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/service-account.json
  Note: Some pinned packages in requirements.txt require Python ≥ 3.10. If you’re on older Python versions, installation may fail; consider upgrading Python.

Each notebook includes a brief explanation of what the code does. It’s currently available in Spanish; an English translation is in progress.



Suggested order for the repository

1. population_scripts.ipynb
    Retrieves the study population (e.g., top ~450 players on the LAN ladder—this is configurable) and generates a .csv with multiple matches per player (no duplicates) plus key match features. See
    sample_apex_enriched_csv.csv for details.

2. plots_scripts.ipynb
    Produces exploratory plots to analyze correlations and distributions of the selected variables.

3. LogReg_n_ML_enriched.ipynb
    Trains a supervised ML model (Logistic Regression) to predict the match outcome (0 = defeat, 1 = victory) using 10-minute match stats. This notebook also includes an alternative variant that excludes “first- 
    blood” features (work in progress) to compare accuracy and model certainty. Spoiler: the enriched version performs better.

4. (Optional) LogReg_n_ML.ipynb
    A simplified version that uses only gold_diff and kill_diff as predictors.

IMPORTANT
Don’t commit secrets: .env, Google credentials JSON, API keys.
Ensure functions.py (and any helper scripts) are present; they’re required for the project to run smoothly.

DISCLAIMER

* This project is part of my Physics bachelor’s degree. Some mathematical/statistical concepts may be non-trivial without prior background in statistics.

* I’m a physicist transitioning into data science; the project is a work in progress and will continue to evolve toward a more formal data-science standard (I’m targeting a Data Science master’s program).

* I know it isn’t perfect. This is a university project. Feedback is welcome!
  Questions or suggestions: eliancervantes4411@alumnos.udg.mx
