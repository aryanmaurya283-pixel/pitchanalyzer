import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials missing. Check your .env file.")
    st.stop()
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title('Sign In / Sign Up')

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = {'name': '', 'email': ''}

mode = st.radio('Choose:', ['Login', 'Sign Up'])
email = st.text_input('Email')
password = st.text_input('Password', type='password')
name = st.text_input('Name (for Sign Up only)') if mode == 'Sign Up' else ''

if st.button(mode):
    if mode == 'Sign Up':
        res = sb.auth.sign_up({"email": email, "password": password, "options": {"data": {"name": name}}})
        if getattr(res, "user", None):
            st.success('Sign up successful! Please check your email to confirm.')
        else:
            st.error(getattr(res, "error", "Sign up failed."))
    else:
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        if res.user:
            st.session_state.logged_in = True
            st.session_state.user = {'name': res.user.user_metadata.get('name', ''), 'email': email}
            st.success('Login successful!')
            st.rerun()
        else:
            st.error('Login failed. Please check your credentials.') 