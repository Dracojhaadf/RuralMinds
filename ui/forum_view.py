from datetime import datetime
import streamlit as st
from core.forum import (
    get_forum_stats,
    get_all_posts,
    get_post_by_id,
    create_post,
    add_reply,
    upvote_post,
    delete_post,
    get_categories,
    search_posts
)
from services.vector_service import get_available_documents

def fmt_dt(iso):
    """Format ISO datetime string to readable string."""
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %I:%M %p")
    except Exception:
        return iso

def render_forum(is_teacher: bool):
    """Render the Discussion Forum tab."""
    st.markdown("## 💬 Discussion Forum")
    
    # Forum stats
    stats = get_forum_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Posts", stats['total_posts'])
    c2.metric("❓ Open", stats['open_posts'])
    c3.metric("✅ Answered", stats['answered_posts'])
    if is_teacher:
        c4.metric("🔔 Pending", stats['pending_posts'])
    else:
        c4.metric("💬 Replies", stats['total_replies'])
    
    st.markdown("---")
    
    # Forum tabs
    ft1, ft2, ft3 = st.tabs(["📋 All Posts", "➕ Create Post", "🔍 Search"])
    
    # ALL POSTS TAB
    with ft1:
        col_filter, col_sort = st.columns(2)
        with col_filter:
            filt = st.selectbox("Filter:", ["All", "Open", "Answered"], key="forum_filter")
        with col_sort:
            sort = st.selectbox("Sort by:", ["Recent", "Popular"], key="forum_sort")
        
        status_map = {"All": None, "Open": "open", "Answered": "answered"}
        sort_map = {"Recent": "recent", "Popular": "popular"}
        
        posts = get_all_posts(status_map[filt], None, sort_map[sort])
        
        if not posts:
            st.info("📭 No posts yet. Be the first to ask a question!")
        
        for p in posts:
            if p['status'] == 'open':
                status_emoji = "❓"
            elif p['status'] == 'answered':
                status_emoji = "✅"
            else:
                status_emoji = "🔒"
            
            with st.container():
                st.markdown(f"### {status_emoji} {p['title']}")
                
                col_meta1, col_meta2, col_meta3 = st.columns([2, 1, 1])
                with col_meta1:
                    role_badge = "👨‍🏫" if p['user_role'] == 'teacher' else "👨‍🎓"
                    st.caption(f"{role_badge} {p['username']} | 📅 {fmt_dt(p['created_at'])}")
                with col_meta2:
                    st.caption(f"📁 {p['category']}")
                with col_meta3:
                    st.caption(f"💬 {len(p['replies'])} replies")
                
                preview = p['content'][:200] + "..." if len(p['content']) > 200 else p['content']
                st.write(preview)
                
                col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 3])
                
                with col_a:
                    if st.button(f"👍 {p['upvotes']}", key=f"up_{p['id']}"):
                        upvote_post(p['id'])
                        st.rerun()
                
                with col_b:
                    if st.button("💬 View", key=f"view_{p['id']}", type="primary"):
                        st.session_state.viewing_post = p['id']
                        st.rerun()
                
                with col_c:
                    if is_teacher or p['username'] == st.session_state.user_data['username']:
                        if st.button("🗑️", key=f"del_{p['id']}"):
                            success, msg = delete_post(
                                p['id'], 
                                st.session_state.user_data['username'],
                                st.session_state.user_data['role']
                            )
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                
                st.markdown("---")
    
    # CREATE POST TAB
    with ft2:
        st.subheader("➕ Create New Post")
        
        with st.form("new_post_form", clear_on_submit=True):
            title = st.text_input("Title*", placeholder="Brief summary of your question")
            cat = st.selectbox("Category*", [c for c in get_categories() if c != "All"])
            content = st.text_area(
                "Question Details*", 
                height=200,
                placeholder="Describe your question in detail..."
            )
            
            docs = get_available_documents()
            if docs:
                related_doc = st.selectbox(
                    "Related Document (optional):", 
                    ["None"] + docs
                )
            else:
                related_doc = "None"
            
            col_submit, col_info = st.columns([1, 2])
            
            with col_submit:
                submit = st.form_submit_button("📤 Post Question", type="primary", use_container_width=True)
            
            with col_info:
                st.caption("*Required fields")
            
            if submit:
                if not title or not content:
                    st.error("❌ Title and content are required")
                elif len(title) < 5:
                    st.error("❌ Title must be at least 5 characters")
                elif len(content) < 10:
                    st.error("❌ Content must be at least 10 characters")
                else:
                    u = st.session_state.user_data
                    rel_doc = None if related_doc == "None" else related_doc
                    
                    success, msg, post_id = create_post(
                        u['username'], 
                        u['role'], 
                        title, 
                        content, 
                        cat,
                        rel_doc
                    )
                    
                    if success:
                        st.success(f"✅ {msg}")
                        st.info(f"📌 Post ID: {post_id}")
                        if 'viewing_post' in st.session_state:
                            del st.session_state.viewing_post
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
    
    # SEARCH TAB
    with ft3:
        st.subheader("🔍 Search Posts")
        
        search_query = st.text_input("Search for:", placeholder="Enter keywords...")
        
        if search_query:
            results = search_posts(search_query)
            st.write(f"**Found {len(results)} result(s)**")
            
            for r in results:
                with st.container():
                    status_emoji = "✅" if r['status'] == 'answered' else "❓"
                    st.markdown(f"#### {status_emoji} {r['title']}")
                    st.caption(f"By {r['username']} | {fmt_dt(r['created_at'])}")
                    st.write(r['content'][:150] + "...")
                    
                    if st.button("View Post", key=f"search_{r['id']}"):
                        st.session_state.viewing_post = r['id']
                        st.rerun()
                    
                    st.markdown("---")
        else:
            st.info("👆 Enter a search term to find posts")
    
    # VIEW SINGLE POST
    if 'viewing_post' in st.session_state:
        post = get_post_by_id(st.session_state.viewing_post)
        
        if post:
            st.markdown("---")
            st.markdown("---")
            
            status_emoji = "✅" if post['status'] == 'answered' else "❓"
            st.markdown(f"## {status_emoji} {post['title']}")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                role_badge = "👨‍🏫" if post['user_role'] == 'teacher' else "👨‍🎓"
                st.write(f"**Posted by:** {role_badge} {post['username']}")
            with col_m2:
                st.write(f"**Category:** {post['category']}")
            with col_m3:
                st.write(f"**Status:** {post['status'].title()}")
            
            st.caption(f"📅 Created: {fmt_dt(post['created_at'])} | Updated: {fmt_dt(post['updated_at'])}")
            
            st.info(post['content'])
            
            if post.get('related_document'):
                st.caption(f"📄 Related Document: {post['related_document']}")
            
            st.markdown("---")
            
            st.markdown(f"### 💬 {len(post['replies'])} Reply/Replies")
            
            if post['replies']:
                for reply in post['replies']:
                    if reply.get('is_answer'):
                        bg_color = "rgba(76, 175, 80, 0.1)"
                        border = "2px solid #4CAF50"
                        prefix = "✅ **Teacher's Answer**"
                    else:
                        bg_color = "rgba(50, 50, 50, 0.05)"
                        border = "1px solid #ddd"
                        prefix = ""
                    
                    role_badge = "👨‍🏫" if reply['user_role'] == 'teacher' else "👨‍🎓"
                    
                    st.markdown(f"""
                    <div style="background:{bg_color}; padding:15px; border-radius:8px; 
                                margin:10px 0; border:{border}">
                        <div style="margin-bottom:8px">
                            <strong>{prefix}</strong><br>
                            <small>{role_badge} {reply['username']} | {fmt_dt(reply['created_at'])}</small>
                        </div>
                        <p style="margin:0">{reply['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("💬 No replies yet. Be the first to respond!")
            
            st.markdown("---")
            
            st.subheader("📝 Add Your Reply")
            
            with st.form("reply_form"):
                reply_content = st.text_area(
                    "Your Reply*",
                    height=150,
                    placeholder="Write your reply here..."
                )
                
                mark_as_answer = False
                if is_teacher:
                    mark_as_answer = st.checkbox(
                        "✅ Mark this as the answer (closes the question)",
                        help="This will mark the post as 'answered'"
                    )
                
                col_reply, col_back = st.columns([1, 3])
                
                with col_reply:
                    if st.form_submit_button("💬 Post Reply", type="primary"):
                        if not reply_content:
                            st.error("❌ Reply cannot be empty")
                        elif len(reply_content) < 5:
                            st.error("❌ Reply must be at least 5 characters")
                        else:
                            u = st.session_state.user_data
                            success, msg = add_reply(
                                post['id'],
                                u['username'],
                                u['role'],
                                reply_content,
                                mark_as_answer
                            )
                            
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                
                with col_back:
                    if st.form_submit_button("⬅️ Back to Posts"):
                        del st.session_state.viewing_post
                        st.rerun()
        else:
            st.error("❌ Post not found")
            if st.button("⬅️ Back to Forum"):
                del st.session_state.viewing_post
                st.rerun()
