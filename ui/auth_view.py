import re
import streamlit as st
from core.auth import authenticate_user, create_user

def validate_password(password: str) -> tuple:
    """Validate password strength."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, "Valid"

def is_valid_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def show_auth_page():
    """Display user login and student registration page."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("📚 Edubridge")
        st.caption("AI-Powered Teaching Assistant")
        st.markdown("---")
        if not st.session_state.show_signup:
            st.subheader("🔐 Login")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.form_submit_button("🚀 Login", type="primary", use_container_width=True):
                        if username and password:
                            success, user_data = authenticate_user(username, password)
                            if success:
                                st.session_state.authenticated = True
                                st.session_state.user_data = user_data
                                st.success(f"✅ Welcome, {user_data['name']}!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid credentials")
                        else:
                            st.warning("⚠️ Enter username and password")
                with col_b:
                    if st.form_submit_button("📝 Student Signup", use_container_width=True):
                        st.session_state.show_signup = True
                        st.rerun()
            
            st.markdown("---")
            if st.button("🔑 Administrator Login", use_container_width=True):
                st.session_state.show_admin_login = True
                st.rerun()
            
            st.info("💡 **Students** can self-register. **Teachers** are created by admin.")
        else:
            st.subheader("📝 Student Registration")
            st.info("👨‍🎓 Self-registration is only for students. Teachers must be created by administrator.")
            
            with st.form("signup_form"):
                new_username = st.text_input("Username (min 3 chars)")
                new_password = st.text_input("Password (min 8 chars, mixed case, number, special char)", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                full_name = st.text_input("Full Name")
                email = st.text_input("Email")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.form_submit_button("✅ Sign Up", type="primary", use_container_width=True):
                        if not all([new_username, new_password, full_name, email]):
                            st.error("❌ All fields required")
                        elif new_password != confirm_password:
                            st.error("❌ Passwords don't match")
                        elif not is_valid_email(email):
                            st.error("❌ Invalid email format")
                        else:
                            valid, msg = validate_password(new_password)
                            if not valid:
                                st.error(f"❌ {msg}")
                            else:
                                success, message = create_user(new_username, new_password, "student", full_name, email)
                                if success:
                                    st.success(message)
                                    st.info("👉 You can now login")
                                    st.session_state.show_signup = False
                                else:
                                    st.error(message)
                with col_b:
                    if st.form_submit_button("⬅️ Back to Login", use_container_width=True):
                        st.session_state.show_signup = False
                        st.rerun()

def show_admin_login():
    """Display administrator login page."""
    st.markdown("<h1 style='text-align: center;'>📚 Administrator Login</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("admin_login"):
            st.warning("🔒 Administrator access only")
            username = st.text_input("Admin Username")
            password = st.text_input("Admin Password", type="password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.form_submit_button("🔓 Login as Admin", type="primary", use_container_width=True):
                    success, user_data = authenticate_user(username, password)
                    if success and user_data['role'] == 'admin':
                        st.session_state.authenticated = True
                        st.session_state.user_data = user_data
                        st.success("✅ Admin access granted")
                        st.rerun()
                    else:
                        st.error("❌ Invalid admin credentials")
            with col_b:
                if st.form_submit_button("⬅️ Back", use_container_width=True):
                    st.session_state.show_admin_login = False
                    st.rerun()
