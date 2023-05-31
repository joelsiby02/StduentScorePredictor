from flask import Flask, render_template, request
from src.logger import logging
from src.exception import CustomException
from src.utils import load_model_and_preprocessor
from src.pipeline.predict_pipeline import CustomData

application = Flask(__name__)
app = application

# Route for home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "GET":
        try:
            return render_template("home.html")
        except Exception as e:
            logging.exception(f"An error occurred: {str(e)}")
            return render_template('error.html', error_message="Error occurred")
    else:
        try:
            # This will be in predictPipeline
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                math_score=float(request.form.get('math_score')),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score')),
                total_score=float(request.form.get('total_score'))
            )
            
            pred_df = data.get_data_as_DataFrame()
            print(pred_df)
            
            # Load the model and preprocessor
            model, preprocessor = load_model_and_preprocessor()

            predicted_scores = data.predict_score(model, preprocessor, pred_df)
            
            return render_template('home.html', results=predicted_scores[0])

        except Exception as e:
            logging.exception(f"Prediction failed: {str(e)}")
            return render_template('error.html', error_message="Prediction failed")

if __name__ == "__main__":
    app.run(debug=True)
