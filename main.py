import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time
import base64
import numpy as np

# Load the pre-trained numerical imputer, scaler, and model using joblib
num_imputer = joblib.load('numerical_imputer.joblib')
scaler = joblib.load('scaler.joblib')
model = joblib.load('Final_model.joblib')

# Define a function to preprocess the input data
def preprocess_input_data(input_data):
    input_data_df = pd.DataFrame(input_data, columns=['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance'])
    num_columns = input_data_df.select_dtypes(include='number').columns

    input_data_imputed_num = num_imputer.transform(input_data_df[num_columns])
    input_scaled_df = pd.DataFrame(scaler.transform(input_data_imputed_num), columns=num_columns)

    return input_scaled_df


# Define a function to make the sepsis prediction
def predict_sepsis(input_data):
    input_scaled_df = preprocess_input_data(input_data)
    prediction = model.predict(input_scaled_df)[0]
    probabilities = model.predict_proba(input_scaled_df)[0]
    sepsis_status = "Positive" if prediction == 1 else "Negative"

    status_icon = "✔" if prediction == 1 else "✘"  # Red 'X' icon for positive sepsis prediction, green checkmark icon for negative sepsis prediction
    sepsis_explanation = "Sepsis is a life-threatening condition caused by an infection. A positive prediction suggests that the patient might be exhibiting sepsis symptoms and requires immediate medical attention." if prediction == 1 else "Sepsis is a life-threatening condition caused by an infection. A negative prediction suggests that the patient is not currently exhibiting sepsis symptoms."

    output_df = pd.DataFrame(input_data, columns=['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance'])
    output_df['Prediction'] = sepsis_status
    output_df['Negative Probability'] = probabilities[0]
    output_df['Positive Probability'] = probabilities[1]

    return output_df, probabilities, status_icon, sepsis_explanation

# Create a Streamlit app
def main():
    st.title('Sepsis Prediction App')

    st.image("Strealit_.jpg")

    # How to use
    st.sidebar.title('How to Use')
    st.sidebar.markdown('1. Adjust the input parameters on the left sidebar.')
    st.sidebar.markdown('2. Click the "Predict" button to initiate the prediction.')
    st.sidebar.markdown('3. The app will simulate a prediction process with a progress bar.')
    st.sidebar.markdown('4. Once the prediction is complete, the results will be displayed below.')


    st.sidebar.title('Input Parameters')


    # Input parameter explanations
    st.sidebar.markdown('**PRG:** Plasma Glucose')
    PRG = st.sidebar.number_input('PRG', value=0.0)

    st.sidebar.markdown('**PL:** Blood Work Result 1')
    PL = st.sidebar.number_input('PL', value=0.0)

    st.sidebar.markdown('**PR:** Blood Pressure Measured')
    PR = st.sidebar.number_input('PR', value=0.0)

    st.sidebar.markdown('**SK:** Blood Work Result 2')
    SK = st.sidebar.number_input('SK', value=0.0)

    st.sidebar.markdown('**TS:** Blood Work Result 3')
    TS = st.sidebar.number_input('TS', value=0.0)

    st.sidebar.markdown('**M11:** BMI')
    M11 = st.sidebar.number_input('M11', value=0.0)

    st.sidebar.markdown('**BD2:** Blood Work Result 4')
    BD2 = st.sidebar.number_input('BD2', value=0.0)

    st.sidebar.markdown('**Age:** What is the Age of the Patient: ')
    Age = st.sidebar.number_input('Age', value=0.0)

    st.sidebar.markdown('**Insurance:** Does the patient have Insurance?')
    insurance_options = {0: 'NO', 1: 'YES'}
    Insurance = st.sidebar.radio('Insurance', list(insurance_options.keys()), format_func=lambda x: insurance_options[x])


    input_data = [[PRG, PL, PR, SK, TS, M11, BD2, Age, Insurance]]

    if st.sidebar.button('Predict'):
        with st.spinner("Predicting..."):
            # Simulate a long-running process
            progress_bar = st.progress(0)
            step = 20 # A big step will reduce the execution time
            for i in range(0, 100, step):
                time.sleep(0.1)
                progress_bar.progress(i + step)

            output_df, probabilities, status_icon, sepsis_explanation = predict_sepsis(input_data)

             # Display prediction outcome and explanations
            st.subheader('Sepsis Prediction Result')
            prediction_text = "Positive" if status_icon == "✔" else "Negative"

            st.markdown(f"Prediction: **{prediction_text}**")

            if prediction_text == "Positive":
                st.image("pos_image.jpg", caption="Positive Sepsis Prediction", width=300)
            else:
                st.image("neg_result.jpg", caption="Negative Sepsis Prediction",  width=300)


            st.markdown(f"{status_icon} {sepsis_explanation}")


            # Display detailed prediction probabilities
            # Find the highest prediction probability and its corresponding sepsis status
            max_prob_index = np.argmax(probabilities)
            highest_prob = probabilities[max_prob_index]
            sepsis_status = 'Negative' if max_prob_index == 0 else 'Positive'

            # Define emojis for better visualization
            emojis = {
            'Negative': '😌',
            'Positive': '😟'
            }

            # Display the highest prediction probability with emoji and explanation
            emoji = emojis[sepsis_status]
            st.write(f"{emoji} The patient's sepsis status is {sepsis_status.lower()} {emoji} with a probability of {highest_prob:.2f}.")
            st.write("This probability represents the likelihood of the patient having the corresponding sepsis status based on the input parameters.")
            st.write("A higher probability value indicates a stronger prediction of the corresponding sepsis status.")

            # Display detailed prediction probabilities
            st.subheader('Output DataFrame')
            st.write(output_df)
            # Add a download button for output_df
            csv = output_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download Output CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

            st.subheader('Sepsis Prediction Distribution')

            fig, ax = plt.subplots()
            bars = ax.bar(['Negative', 'Positive'], probabilities)
            ax.set_xlabel('Sepsis Status')
            ax.set_ylabel('Probability')
            ax.set_title('Sepsis Prediction Probabilities')
            ax.legend()

            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.annotate(f'{prob:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

            st.pyplot(fig)

            st.subheader('Feature Importance')

            # Print feature importance
            if hasattr(model, 'coef_'):
                feature_importances = model.coef_[0]
                feature_names = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance']

                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
                importance_df = importance_df.sort_values('Importance', ascending=False)
                fig, ax = plt.subplots()
                bars = ax.bar(importance_df['Feature'], importance_df['Importance'])
                ax.set_xlabel('Feature')
                ax.set_ylabel('Importance')
                ax.set_title('Feature Importance')
                ax.tick_params(axis='x', rotation=45)

                # Add data labels to the bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
                st.pyplot(fig)

            else:
                st.write('Feature importance is not available for this model.')

            st.subheader('Feature Importance Explanation')

            st.write("The figure you see is a plot of feature importance for linear regression prediction of sepsis. It displays the importance of each feature in predicting sepsis.")

            st.write("The x-axis of the plot represents the features, while the y-axis represents the importance of each feature. Features are ranked in order of importance, with the most important feature at the top.")

            st.write("In this particular case, the most important feature for predicting sepsis is PL (Blood Work Result 1). This is followed by PRG (Plasma Glucose), BD2 (Blood Work Result 4), and M11 (BMI). Other features are comparatively less important in predicting sepsis.")

            st.write("Feature importance is a measure of how much a feature contributes to the prediction of the target variable. In this case, the target variable is whether or not the patient has sepsis. Higher feature importance values indicate more significance for prediction.")

            st.write("The importance of each feature is calculated using the linear regression model.")

            #st.subheader('Sepsis Explanation')
            #st.markdown(f"{status_icon} {sepsis_explanation}")


if __name__ == '__main__':
    main()
