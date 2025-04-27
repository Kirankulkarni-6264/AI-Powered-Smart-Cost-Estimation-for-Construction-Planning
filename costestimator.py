import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from scipy.stats import zscore

#Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API Key not found! Set it as an environment variable")
    st.stop()

#Initialize LangChain LLM + Memory
memory = ConversationBufferMemory()
llm = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai_api_key)
conversation = ConversationChain(llm=llm, memory=memory)

#City Base Rates
city_base_rates = {
    'New York': 3500,
    'San Francisco': 3400,
    'Los Angeles': 3000,
    'Chicago': 2600,
    'Miami': 2500,
    'Dallas': 2400,
    'Houston': 2300,
    'Atlanta': 2200
}

#Load Dataset
@st.cache_data
def load_data():
    file_path = "realistic_construction_estimates.csv"
    df = pd.read_csv(file_path)
    df['Project_Type'] = df['Project_Type'].astype(str)
    df['Location'] = df['Location'].astype(str)
    return df

df_original = load_data()
st.title("AI-Powered Smart Construction Cost Estimation Chatbot")
#Show Raw Data
st.header("Raw Data Display")
if st.checkbox("Show Initial Raw Data"):
    st.dataframe(df_original.head(10))

#Data Quality Check
st.subheader("Data Quality Check")
if df_original.isnull().sum().sum() == 0:
    st.success("No missing values detected!")
else:
    st.error("Missing values found! Filling with median values.")
    df_original = df_original.fillna(df_original.median(numeric_only=True))

numeric_cols = df_original.select_dtypes(include=['float64', 'int64']).columns
z_scores = zscore(df_original[numeric_cols])
outliers = (abs(z_scores) > 3).any(axis=1)
if outliers.sum() == 0:
    st.success("No major outliers detected!")
else:
    st.warning(f"{outliers.sum()} outlier records detected. They are retained for analysis.")

#Show Data after Preprocessing
st.subheader("Processed Data Display")
st.dataframe(df_original.head(10))

# Create a copy for encoding
df = df_original.copy()

# Preprocessing
label_encoders = {}
for col in ['Project_Type', 'Location']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['Total_Estimate'])
y = df['Total_Estimate']

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Helper function to prepare input
def prepare_features(project_type, location, total_area, floors, basements):
    base_rate = city_base_rates.get(location, 2500)
    base_cost = total_area * base_rate

    material_cost = base_cost * 0.6 * (1 + 0.05 * (floors - 1) + 0.10 * basements)
    labor_cost = base_cost * 0.3 * (1 + 0.05 * (floors - 1) + 0.10 * basements)

    return pd.DataFrame({
        'Material_Cost': [material_cost],
        'Labor_Cost': [labor_cost],
        'Project_Type': [label_encoders['Project_Type'].transform([project_type])[0]],
        'Total_Area': [total_area],
        'Number_of_Floors': [floors],
        'Number_of_Basements': [basements],
        'Location': [label_encoders['Location'].transform([location])[0]]
    })

# Streamlit UI
st.title("AI-Powered Smart Construction Cost Estimation Chatbot")

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")

st.subheader("Correlation Heatmap")
st.write("The correlation heatmap shows the strength of relationships between numerical features in the dataset. Strong correlations help in understanding feature influence on project cost.")
fig3, ax3 = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

st.subheader("Location vs Total Estimated Cost")
st.write("This graph visualizes the total construction costs grouped by location, helping to identify cities with higher or lower project expenses.")
fig4, ax4 = plt.subplots()
sns.barplot(data=df_original, x="Location", y="Total_Estimate", estimator=sum, ci=None, ax=ax4)
ax4.set_xlabel("Location")
ax4.set_ylabel("Total Estimated Cost ($)")
ax4.set_title("Location-wise Total Construction Cost")
plt.xticks(rotation=45)
st.pyplot(fig4)

#Cost Estimation Interface
st.header("Cost Estimation Interface")

project_types = ['Residential', 'Commercial', 'Industrial', 'Institutional']
locations = list(city_base_rates.keys())

project_type = st.selectbox("Project Type", project_types)
location = st.selectbox("Location", locations)
total_area = st.number_input("Total Area (m²)", min_value=50.0, format="%.2f")
number_of_floors = st.number_input("Number of Floors", min_value=1, max_value=20, step=1)
number_of_basements = st.number_input("Number of Basements", min_value=0, max_value=5, step=1)

if st.button("Estimate Cost"):
    input_data = prepare_features(project_type, location, total_area, number_of_floors, number_of_basements)
    predicted_cost = model.predict(input_data)[0]

    st.success(f"Predicted Total Construction Cost: ${predicted_cost:,.0f}")

    st.session_state['base_project'] = {
        'project_type': project_type,
        'location': location,
        'total_area': total_area,
        'number_of_floors': number_of_floors,
        'number_of_basements': number_of_basements,
        'predicted_cost': predicted_cost
    }

#What-If Scenario Simulator
st.header("What-If Scenario Simulator")
what_if_change = st.text_input("Describe a project change (e.g., 'Increase area by 20%', 'Add 2 floors'):")

if st.button("Simulate What-If"):
    if 'base_project' not in st.session_state:
        st.warning("Please estimate a base project first.")
    elif what_if_change.strip() == "":
        st.warning("Please describe a change.")
    else:
        base = st.session_state['base_project']
        updated_area = base['total_area']
        updated_floors = base['number_of_floors']
        updated_basements = base['number_of_basements']

        change = what_if_change.lower()
        if 'increase area' in change:
            percent = int(''.join(filter(str.isdigit, change)))
            updated_area *= (1 + percent/100)
        if 'decrease area' in change:
            percent = int(''.join(filter(str.isdigit, change)))
            updated_area *= (1 - percent/100)
        if 'add' in change and 'floor' in change:
            num = int(''.join(filter(str.isdigit, change)))
            updated_floors += num
        if 'add' in change and 'basement' in change:
            num = int(''.join(filter(str.isdigit, change)))
            updated_basements += num

        input_data = prepare_features(base['project_type'], base['location'], updated_area, updated_floors, updated_basements)
        new_predicted_cost = model.predict(input_data)[0]

        st.success(f"Updated Total Estimated Cost after change: ${new_predicted_cost:,.0f}")

        what_if_prompt = f"""
        You are a construction expert.

        Base project:
        - Area: {base['total_area']} m²
        - Floors: {base['number_of_floors']}
        - Basements: {base['number_of_basements']}
        - Initial Cost: ${base['predicted_cost']:,.0f}

        Change:
        "{what_if_change}"

        After change:
        - Area: {updated_area:.2f} m²
        - Floors: {updated_floors}
        - Basements: {updated_basements}
        - New Cost: ${new_predicted_cost:,.0f}

        Please explain simply:
        - Why cost changed
        - Professional advice for the user
        """
        response = conversation.run(what_if_prompt)
        st.markdown(f"What-If Simulation Explanation:\n\n{response}")

# General Construction Chatbot Section
st.header("Construction Expert Chatbot")
user_question = st.text_input("Ask any construction-related question:")

if st.button("Ask AI"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        chat_prompt = f"""
        You are an expert construction consultant.
        Provide short, practical, professional answers.

        Question:
        {user_question}
        """
        response = conversation.run(chat_prompt)
        st.markdown(f"AI Response:\n\n{response}")