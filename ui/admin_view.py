import streamlit as st
from core.auth import get_all_users, delete_user, create_user
from core.forum import get_forum_stats
from services.vector_service import get_available_documents
from services.video_service import get_available_videos
from ui.auth_view import is_valid_email, validate_password

def render_admin_panel():
    """Render the administrator management interface."""
    st.title("📚 Administrator Panel")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"**Logged in as:** {st.session_state.user_data['name']}")
    with col3:
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.rerun()
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["👥 All Users", "👨‍🏫 Create Teacher", "📊 Statistics"])
    
    with tab1:
        users = get_all_users()
        st.metric("Total Users", len(users))
        
        # Separate by role
        teachers = [u for u in users if u['role'] == 'teacher']
        students = [u for u in users if u['role'] == 'student']
        
        st.subheader("👨‍🏫 Teachers")
        for user in teachers:
            col_a, col_b = st.columns([0.8, 0.2])
            with col_a:
                st.write(f"**{user['name']}** (@{user['username']}) - {user['email']}")
            with col_b:
                if user['username'] not in ['admin', 'administrator']:
                    if st.button("🗑️", key=f"del_t_{user['username']}"):
                        success, msg = delete_user(user['username'])
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
        
        st.subheader("👨‍🎓 Students")
        for user in students:
            col_a, col_b = st.columns([0.8, 0.2])
            with col_a:
                st.write(f"**{user['name']}** (@{user['username']}) - {user['email']}")
            with col_b:
                if st.button("🗑️", key=f"del_s_{user['username']}"):
                    success, msg = delete_user(user['username'])
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    
    with tab2:
        st.subheader("➕ Create Teacher Account")
        st.info("🔒 Only administrators can create teacher accounts")
        
        with st.form("create_teacher_form"):
            t_username = st.text_input("Username*")
            t_email = st.text_input("Teacher Email*", placeholder="teacher@school.edu")
            t_name = st.text_input("Full Name*")
            t_password = st.text_input("Temporary Password*", type="password", 
                                      help="Min 8 chars with uppercase, lowercase, number, special char")
            
            if st.form_submit_button("✅ Create Teacher", type="primary"):
                if all([t_username, t_email, t_name, t_password]):
                    if not is_valid_email(t_email):
                        st.error("❌ Invalid email")
                    else:
                        valid, msg = validate_password(t_password)
                        if not valid:
                            st.error(f"❌ {msg}")
                        else:
                            success, message = create_user(t_username, t_password, "teacher", t_name, t_email)
                            if success:
                                st.success("✅ Teacher account created!")
                                st.info(f"📧 Notify {t_name} at {t_email} to change password on first login")
                            else:
                                st.error(message)
                else:
                    st.error("❌ Fill all fields")
    
    with tab3:
        forum_stats = get_forum_stats()
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Users", len(get_all_users()))
            st.metric("Teachers", len([u for u in get_all_users() if u['role'] == 'teacher']))
        with col_b:
            st.metric("Documents", len(get_available_documents()))
            st.metric("Videos", len(get_available_videos()))
        with col_c:
            st.metric("Forum Posts", forum_stats['total_posts'])
            st.metric("Pending Posts", forum_stats['pending_posts'])
