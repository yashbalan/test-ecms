import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import requests
import re
import os

# Streamlit configuration
st.set_page_config(layout="wide", page_title="Hopcharge Dashboard", page_icon=":bar_chart:")

# Function to clean license plates
def clean_license_plate(plate):
    match = re.match(r"([A-Z]+[0-9]+)(_R)$", plate)
    if match:
        return match.group(1)
    return plate

# Function to get data from the API
# Function to get JWT token from the API
def fetch_jwt_token():
    login_url = "https://2e855a4f93a0.api.hopcharge.com/admin/api/v1/login"
    payload = {
        "username": "admin",
        "password": "Hopadmin@2024#"
    }
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }
    response = requests.post(login_url, headers=headers, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        token = response.json().get('token')
        return token
    else:
        st.error("Failed to fetch JWT token")
        return None

    # Function to get data from the API


def fetch_data(url, token):
    headers = {
        'accept': 'application/json',
        'accept-language': 'en-US,en;q=0.9',
        'authorization': f'Bearer {token}'
    }
    response = requests.get(url, headers=headers)

    # Try to parse the response JSON
    response_json = response.json()
    if 'data' in response_json:
        return pd.json_normalize(response_json['data'])
    else:
        return pd.DataFrame()  # Return an empty DataFrame if 'data' key is not found

    # Fetch the JWT token


jwt_token = fetch_jwt_token()
        

# Function to get data from CSV files
def get_csv_files(directory_path):
    file_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    df_list = [pd.read_csv(file) for file in file_list]
    concatenated_df = pd.concat(df_list, ignore_index=True)
    return concatenated_df

# URLs for the APIs
url_bookings = "https://2e855a4f93a0.api.hopcharge.com/admin/api/v1/bookings/past?filter={\"chargedAt_lte\":\"2024-06-01\",\"chargedAt_gte\":\"2024-12-31\"}&range=[0,3000000]&sort=[\"created\",\"DESC\"]"
url_drivers = "https://2e855a4f93a0.api.hopcharge.com/admin/api/v1/drivers-shifts/export-list?filter={\"action\":\"exporter\",\"startedAt_lte\":\"2024-06-01\",\"endedAt_gte\":\"2024-12-31\"}"

# Fetch data from the APIs
past_bookings_df = fetch_data(url_bookings, jwt_token)
drivers_shifts_df = fetch_data(url_drivers, jwt_token)

# Load and clean CSV data
csv_directory_path = './data'
df_csv = get_csv_files(csv_directory_path)
df_csv = df_csv[df_csv['canceled'] != True]
df_csv['licensePlate'] = df_csv['licensePlate'].str.upper()
df_csv['licensePlate'] = df_csv['licensePlate'].str.replace('HR55AJ4OO3', 'HR55AJ4003')
df_csv['fromTime'] = pd.to_datetime(df_csv['fromTime'])
df_csv.rename(columns={'location.state': 'Customer Location City', 'fromTime': 'Actual Date'}, inplace=True)
df_csv['Actual Date'] = pd.to_datetime(df_csv['Actual Date'], format='mixed', errors='coerce')
df_csv = df_csv[df_csv['Actual Date'].dt.year > 2021]
df_csv['Actual Date'] = df_csv['Actual Date'].dt.date
df_csv['Customer Location City'].replace({'Haryana': 'Gurugram', 'Uttar Pradesh': 'Noida'}, inplace=True)
cities = ['Gurugram', 'Noida', 'Delhi']
df_csv = df_csv[df_csv['Customer Location City'].isin(cities)]

# Merge CSV data with EPOD data
df_epod = pd.read_csv('EPOD NUMBER.csv')
requiredcols = ['Actual Date', 'EPOD Name', 'Customer Location City']
merged_df_csv = pd.merge(df_csv, df_epod, on=["licensePlate"])
merged_df_csv = merged_df_csv[requiredcols]

# Clean and filter API data
if not past_bookings_df.empty and not drivers_shifts_df.empty:
    drivers_shifts_df['licensePlate'] = drivers_shifts_df['licensePlate'].apply(clean_license_plate)
    filtered_drivers_df = drivers_shifts_df[(drivers_shifts_df['donorVMode'] == 'FALSE') &
                                            (drivers_shifts_df['bookingStatus'] == 'completed')]

    merged_df_api = pd.merge(filtered_drivers_df, past_bookings_df[['uid', 'location.state']],
                             left_on='bookingUid', right_on='uid', how='left')

    merged_df_api['Actual Date'] = pd.to_datetime(merged_df_api['bookingFromTime'], errors='coerce')
    final_df_api = merged_df_api[['Actual Date', 'licensePlate', 'location.state', 'bookingUid', 'uid',
                                  'bookingFromTime', 'bookingStatus', 'customerUid', 'totalUnitsCharged']].rename(
        columns={'location.state': 'Customer Location City'})

    final_df_api = final_df_api.dropna(subset=['Actual Date']).drop_duplicates(
        subset=['uid', 'bookingUid', 'Actual Date'])

    final_df_api['licensePlate'] = final_df_api['licensePlate'].str.upper()
    replace_dict = {
        'HR551305': 'HR55AJ1305',
        'HR552932': 'HR55AJ2932',
        'HR551216': 'HR55AJ1216',
        'HR555061': 'HR55AN5061',
        'HR554745': 'HR55AR4745',
        'HR55AN1216': 'HR55AJ1216',
        'HR55AN8997': 'HR55AN8997'
    }
    final_df_api['licensePlate'] = final_df_api['licensePlate'].replace(replace_dict)
    final_df_api['Actual Date'] = pd.to_datetime(final_df_api['Actual Date'], errors='coerce')
    final_df_api = final_df_api[final_df_api['Actual Date'].dt.year > 2021]
    final_df_api['Actual Date'] = final_df_api['Actual Date'].dt.date

    final_df_api['Customer Location City'].replace({'Haryana': 'Gurugram', 'Uttar Pradesh': 'Noida'}, inplace=True)
    final_df_api = final_df_api[final_df_api['Customer Location City'].isin(cities)]

    merged_df_api = pd.merge(final_df_api, df_epod, on=["licensePlate"])
    merged_df_api = merged_df_api[requiredcols]

# Combine CSV and API data
combined_df = pd.concat([merged_df_csv, merged_df_api], ignore_index=True)

# Freeze start dates for specified EPODs
epod_start_dates = {
    "EPOD-005": "2024-01-22",
    "EPOD-007": "2024-03-15",
    "EPOD-010": "2024-05-10",
    "EPOD-011": "2023-11-10",
    "EPOD-012": "2024-03-29"
}

for epod, start_date in epod_start_dates.items():
    start_date = pd.to_datetime(start_date).date()
    combined_df = combined_df[~((combined_df['EPOD Name'] == epod) & (combined_df['Actual Date'] < start_date))]

# Function to format numbers in INR
def formatINR(number):
    s, *d = str(number).partition(".")
    r = ",".join([s[x - 2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return "".join([r] + d)

def check_credentials():
    st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=1, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)

    image = Image.open('LOGO HOPCHARGE-03.png')
    col2.image(image, use_column_width=True)
    col2.markdown(
        "<h2 style='text-align: center;'>ECMS Login</h2>", unsafe_allow_html=True)
    image = Image.open('roaming vans.png')
    col1.image(image, use_column_width=True)

    with col2:
        username = st.text_input("Username")
        password = st.text_input(
            "Password", type="password")
    flag = 0
    if username in st.secrets["username"] and password in st.secrets["password"]:
        index = st.secrets["username"].index(username)
        if st.secrets["password"][index] == password:
            st.session_state["logged_in"] = True
            flag = 1
        else:
            col2.warning("Invalid username or password.")
            flag = 0
    elif username not in st.secrets["username"] or password not in st.secrets["password"]:
        col2.warning("Invalid username or password.")
        flag = 0
    ans = [username, flag]
    return ans

def main_page(username):
    st.markdown(
        """
        <script>
        function refresh() {
            window.location.reload();
        }
        setTimeout(refresh, 120000);
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
            .appview-container .main .block-container {{
                padding-top: 1rem;
                padding-bottom: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    image = Image.open('LOGO HOPCHARGE-03.png')
    col1.image(image, use_column_width=True)

    st.markdown("<h2 style='text-align: left;'>EV Charging Management System</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        combined_df['Actual Date'] = pd.to_datetime(combined_df['Actual Date'], errors='coerce')
        min_date = combined_df['Actual Date'].min().date()
        max_date = combined_df['Actual Date'].max().date()
        start_date = st.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date,
                                   key="epod-date-start")
    with col2:
        end_date = st.date_input('End Date', min_value=min_date, max_value=max_date, value=max_date,
                                 key="epod-date-end")

    def get_epods_by_username(df, input_username):
        filtered_df = df[df['username'].str.contains(input_username, na=False)]
        epod_list = filtered_df['EPOD Name'].tolist()

        return epod_list

    epods = get_epods_by_username(df_epod, username)

    with col3:
        EPod = st.multiselect(label='Select The EPod', options=['All'] + epods, default='All')
    if 'All' in EPod:
        EPod = epods

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = combined_df[(combined_df['Actual Date'] >= start_date) & (combined_df['Actual Date'] <= end_date)]

    filtered_data = filtered_data[(filtered_data['EPOD Name'].isin(EPod))]
    filtered_data['Actual Date'] = pd.to_datetime(filtered_data['Actual Date'])
    final_df_count = filtered_data.groupby(['Actual Date', 'Customer Location City']).size().reset_index(
        name='Session Count')
    final_df_count['Actual Date'] = final_df_count['Actual Date'].dt.strftime('%d/%m/%y')

    sumcount = final_df_count['Session Count'].sum()
    col4.metric("Total Sessions of EPods", formatINR(sumcount))
    revenue = sumcount * 150
    revenue = formatINR(revenue)
    col5.metric("Total Revenue", f"\u20B9{revenue}")

    fig = px.bar(final_df_count, x='Actual Date', y='Session Count',
                 color_discrete_map={'Delhi': '#243465', 'Gurugram': ' #5366a0', 'Noida': '#919fc8'},
                 color='Customer Location City', text=final_df_count['Session Count'])
    total_counts = final_df_count.groupby('Actual Date')['Session Count'].sum().reset_index()

    for i, date in enumerate(total_counts['Actual Date']):
        fig.add_annotation(
            x=date,
            y=total_counts['Session Count'][i] + 0.9,
            text=str(total_counts['Session Count'][i]),
            showarrow=False,
            align='center',
            font_size=16,
            font=dict(color='black')
        )

    fig.update_layout(
        title='Session Count of All EPods till Date',
        xaxis_title='Date',
        yaxis_title='Session Count',
        xaxis_tickangle=-45,
        width=1200,
        legend_title='HSZs: ',
    )

    with col1:
        st.plotly_chart(fig, use_container_width=False)

    filtered_data = combined_df[combined_df['EPOD Name'].isin(EPod)]

    if len(EPod) > 1:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        filtered_data = filtered_data.sort_values('EPOD Name')
        for epod in filtered_data['EPOD Name'].unique():
            with col1:
                st.subheader(epod)
            filtered_data = combined_df[
                (combined_df['Actual Date'] >= start_date) & (combined_df['Actual Date'] <= end_date)]
            final_df_count = filtered_data[filtered_data['EPOD Name'] == epod].groupby(
                ['Actual Date', 'Customer Location City']).size().reset_index(name='Session Count')
            final_df_count['Actual Date'] = final_df_count['Actual Date'].dt.strftime('%d/%m/%y')
            final_df_count = final_df_count.sort_values('Actual Date', ascending=True)
            sumcount = final_df_count['Session Count'].sum()
            revenue = sumcount * 150
            revenue = formatINR(revenue)
            sumcount = formatINR(sumcount)
            col1.metric(f"Total Sessions by {epod}", sumcount)
            col1.metric("Total Revenue", f"\u20B9{revenue}")

            fig = px.bar(final_df_count, x='Actual Date', y='Session Count', color='Customer Location City',
                         color_discrete_map={'Delhi': '#243465', 'Gurugram': ' #5366a0', 'Noida': '#919fc8'},
                         text='Session Count')
            total_counts = final_df_count.groupby('Actual Date')['Session Count'].sum().reset_index()

            for i, date in enumerate(total_counts['Actual Date']):
                fig.add_annotation(
                    x=date,
                    y=total_counts['Session Count'][i] + 0.2,
                    text=str(total_counts['Session Count'][i]),
                    showarrow=False,
                    align='center',
                    font_size=18,
                    font=dict(color='black')
                )

            fig.update_xaxes(categoryorder='category ascending')
            fig.update_layout(
                title='Session Count by Date',
                xaxis_title='Date',
                yaxis_title=f'Session Count of {epod}',
                xaxis_tickangle=-45,
                width=1200,
                legend_title='HSZs: '
            )
            with col1:
                st.plotly_chart(fig)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if st.session_state.logged_in:
    main_page(st.session_state.username)
else:
    ans = check_credentials()
    if ans[1]:
        st.session_state.logged_in = True
        st.session_state.username = ans[0]
        st.experimental_rerun()
