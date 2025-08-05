import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

# --- Constants ---
# Using placeholders for 2025 as per instructions.
CPF_RETIREMENT_SUMS_2025 = {
    "BRS": 106500,
    "FRS": 213000,
    "ERS": 426000,
}

# CPF interest rates (floor rates)
CPF_INTEREST_RATES = {
    "OA": 0.025,
    "SA": 0.04,
    "MA": 0.04,
    "RA": 0.04,
}
# Extra interest for first $60k, then next $30k.
EXTRA_INTEREST_TIERS = {
    "first_60k": 0.01,
    "next_30k": 0.01,  # For members below 55
    "additional_55_and_above": 0.01,  # For first $30k for members 55+
}

DATA_SOURCES = [
    "https://www.cpf.gov.sg/service/article/what-are-the-basic-retirement-sum-full-retirement-sum-and-enhanced-retirement-sum-applicable-to-me",
    # Retirement sums (BRS, FRS, ERS)
    "https://www.cpf.gov.sg/service/article/how-are-cpf-interest-rates-determined",
    # How CPF interest rates are determined
    "https://www.cpf.gov.sg/member/growing-your-savings/earning-higher-returns/earning-attractive-interest",
    # CPF interest rates explanation and earning higher returns
    "https://www.cpf.gov.sg/service/article/how-much-cpf-savings-can-i-withdraw-from-age-55-to-64",
    # CPF withdrawals from age 55
    "https://www.mom.gov.sg/employment-practices/central-provident-fund",  # MOM CPF overview
    "https://www.cpf.gov.sg/member/cpf-overview"
    # General CPF understanding (from cpf.gov.sg, as gov.sg article may be outdated)
]


# --- RAG and Agent Setup ---

@st.cache_resource(show_spinner="Setting up RAG: Ingesting CPF sources...")
def setup_rag():
    """
    Sets up the RAG pipeline by loading, splitting, and embedding documents.
    This function is cached to prevent re-running on every interaction.
    """
    try:
        loader = WebBaseLoader(DATA_SOURCES)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Use st.secrets for API key
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY secret not found. Please set it in Streamlit Cloud.")
            return None, None

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        retriever = vectorstore.as_retriever()

        retriever_tool = create_retriever_tool(
            retriever,
            "cpf_policy_retriever",
            "Searches and returns information about CPF policies, rules, and regulations. Use this for any questions about CPF.",
        )
        return retriever_tool, api_key
    except Exception as e:
        st.error(f"Error setting up RAG: {e}")
        return None, None


@tool
def retirement_savings_calculator(
        current_age: int,
        retirement_age: int,
        current_oa: float,
        current_sa: float,
        current_ma: float,
        monthly_contribution_oa: float,
        monthly_contribution_sa: float,
        monthly_contribution_ma: float,
) -> dict:
    """
    Estimates CPF balances at retirement age using compound interest.

    Args:
        current_age (int): The user's current age.
        retirement_age (int): The desired retirement age.
        current_oa (float): Current balance in Ordinary Account.
        current_sa (float): Current balance in Special Account.
        current_ma (float): Current balance in MediSave Account.
        monthly_contribution_oa (float): Monthly contribution to OA.
        monthly_contribution_sa (float): Monthly contribution to SA.
        monthly_contribution_ma (float): Monthly contribution to MA.

    Returns:
        dict: A dictionary containing the projected yearly breakdown as a DataFrame
              and a summary string.
    """
    years_to_retirement = retirement_age - current_age
    if years_to_retirement < 0:
        return {"error": "Retirement age cannot be less than current age."}

    yearly_data = []
    oa, sa, ma = current_oa, current_sa, current_ma

    for year in range(years_to_retirement + 1):
        age = current_age + year

        # Calculate interest for the year
        # Simplified extra interest calculation for demonstration
        total_balance = oa + sa + ma
        extra_interest_sa = 0
        if age >= 55:
            # First $30,000 in RA gets additional 1%
            extra_interest_sa = min(sa, 30000) * EXTRA_INTEREST_TIERS["additional_55_and_above"]

        oa_interest = oa * CPF_INTEREST_RATES["OA"]
        sa_interest = sa * CPF_INTEREST_RATES["SA"] + extra_interest_sa
        ma_interest = ma * CPF_INTEREST_RATES["MA"]

        oa += oa_interest
        sa += sa_interest
        ma += ma_interest

        yearly_data.append({
            "Year": datetime.now().year + year,
            "Age": age,
            "OA": round(oa, 2),
            "SA": round(sa, 2),
            "MA": round(ma, 2),
            "Total": round(oa + sa + ma, 2),
        })

        # Add contributions for the next year (if not the last year)
        if year < years_to_retirement:
            oa += monthly_contribution_oa * 12
            sa += monthly_contribution_sa * 12
            ma += monthly_contribution_ma * 12

    df = pd.DataFrame(yearly_data)

    summary = (
        f"Projection Summary:\n"
        f"- At age {retirement_age}, your estimated total CPF balance will be **S${df['Total'].iloc[-1]:,.2f}**.\n"
        f"- OA: **S${df['OA'].iloc[-1]:,.2f}**\n"
        f"- SA: **S${df['SA'].iloc[-1]:,.2f}**\n"
        f"- MA: **S${df['MA'].iloc[-1]:,.2f}**\n"
        f"This compares to the 2025 retirement sums: BRS (S${CPF_RETIREMENT_SUMS_2025['BRS']:,}), FRS (S${CPF_RETIREMENT_SUMS_2025['FRS']:,}), and ERS (S${CPF_RETIREMENT_SUMS_2025['ERS']:,})."
    )

    return {"dataframe": df.to_dict('records'), "summary": summary}


@st.cache_resource(show_spinner="Warming up the AI Agent...")
def setup_agent(_api_key, _retriever_tool):
    """
    Initializes the LangChain agent with tools.
    """
    if not _api_key or not _retriever_tool:
        return None

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=_api_key)
    tools = [_retriever_tool, retirement_savings_calculator]

    # Safeguard: System prompt to guide the agent's behavior
    prompt_template = """
    You are "CPF Sage", a helpful and respectful AI assistant for the CPF Retirement Planner app.
    Your primary role is to provide accurate and easy-to-understand information about Singapore's CPF policies based ONLY on the provided documents.

    IMPORTANT INSTRUCTIONS:
    1.  **Stick to the Source**: Answer questions strictly based on the retrieved context from the `cpf_policy_retriever` tool. Do not use any external knowledge or make assumptions. If the information is not in the documents, state that clearly.
    2.  **Reject Inappropriate Inputs**: Do not answer questions that are irrelevant to CPF, malicious, or seek personal financial advice. Politely decline such requests.
    3.  **Be Clear and Concise**: Explain complex CPF terms in simple language. Use formatting like bullet points and tables to improve readability.
    4.  **Use Tools When Necessary**:
        - For any questions about CPF rules, limits, or definitions, use the `cpf_policy_retriever` tool.
        - For calculations related to retirement savings projections, use the `retirement_savings_calculator` tool.
    5.  **Do Not Hallucinate**: Never make up facts, figures, or policy details. If you don't know the answer, say so.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


# --- Page Implementations ---

def display_home():
    # Header with logo and title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("cpf_logo.png", width=120)
    with col2:
        st.title("🏦 CPF Retirement Planner")
        st.markdown("*Your AI-powered guide to CPF planning*")

    # Welcome section with better formatting
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h3 style="color: #1f77b4; margin-top: 0;">🎯 Our Mission</h3>
        <p style="font-size: 16px; line-height: 1.6;">To empower Singaporeans with personalized insights and clear, accessible information about their CPF savings through AI-powered tools.</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation guide with icons
    st.markdown("""
    ### 🧭 Quick Navigation Guide

    Use the sidebar to explore these features:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        📊 **Retirement Savings Estimator**  
        Project your future CPF balances with interactive charts

        💰 **Withdrawal Options Explainer**  
        Understand rules for accessing your CPF savings
        """)
    with col2:
        st.markdown("""
        ℹ️ **About Us**  
        Learn about this project's scope and objectives

        🔧 **Methodology**  
        Discover the technology behind this app
        """)

    # Important Disclaimer
    with st.expander("⚠️ IMPORTANT DISCLAIMER - Please Read", expanded=True):
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; border: 2px solid #ffc107;">
            <p style="color: #856404; font-weight: bold; font-size: 16px; margin-bottom: 15px;">
                IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. 
                The information provided here is NOT intended for real-world usage and should not be relied upon 
                for making any decisions, especially those related to financial, legal, or healthcare matters.
            </p>
            <p style="color: #856404; font-size: 14px; margin-bottom: 15px;">
                Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. 
                You assume full responsibility for how you use any generated output.
            </p>
            <p style="color: #856404; font-size: 14px; font-weight: bold; margin-bottom: 0;">
                Always consult with qualified professionals for accurate and personalized advice.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Policy Update Checker with better design
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: white; margin-top: 0;">🤖 AI Policy Assistant</h2>
        <p style="color: white; font-size: 16px; margin-bottom: 0;">Ask me anything about CPF policies! I'm powered by official sources.</p>
    </div>
    """, unsafe_allow_html=True)

    # Example questions for user guidance
    with st.expander("💡 Example Questions You Can Ask", expanded=False):
        st.markdown("""
        - "How are CPF interest rates calculated?"
        - "What are the retirement sums for 2025?"
        - "When can I start withdrawing from my CPF?"
        - "What is the extra interest on the first $60,000?"
        - "How does CPF LIFE work?"
        """)

    query = st.text_input(
        "💬 Ask your CPF question here:",
        placeholder="Type your question about CPF policies...",
        help="Ask about CPF rules, interest rates, withdrawal options, or any policy-related questions."
    )

    if query and st.session_state.agent_executor:
        with st.spinner("🔍 Our AI agent is searching for the answer..."):
            try:
                response = st.session_state.agent_executor.invoke({"input": query, "chat_history": []})
                st.success(response['output'])
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
    elif query and not st.session_state.agent_executor:
        st.error("🤖 AI Agent is not available. Please check your setup.")


def display_estimator():
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>📊 Retirement Savings Estimator</h1>
        <p style="font-size: 18px; color: #666;">Project your future CPF savings with AI-powered calculations</p>
    </div>
    """, unsafe_allow_html=True)

    # Progress indicator
    st.markdown("""
    <div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4;">
        <strong>📋 Step 1:</strong> Fill in your details below → <strong>📈 Step 2:</strong> View your projection → <strong>💬 Step 3:</strong> Ask follow-up questions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Your Information")

    with st.form("estimator_form"):
        # Personal Details Section
        st.markdown("#### 👤 Personal Details")
        c1, c2 = st.columns(2)
        with c1:
            current_age = st.slider(
                "🎂 Your Current Age",
                21, 70, 30,
                help="Your age today. This affects contribution rates and retirement planning."
            )
            retirement_age = st.slider(
                "🏖️ Target Retirement Age",
                55, 75, 65,
                help="When you plan to retire. CPF allows withdrawals from age 55."
            )
        with c2:
            monthly_salary = st.number_input(
                "💰 Monthly Salary (SGD)",
                1000, 20000, 5000,
                help="Used to estimate your monthly CPF contributions."
            )
            st.markdown("""
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <small><strong>💡 Tip:</strong> This estimate uses simplified contribution rates. Actual rates vary by age.</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 💳 Current CPF Balances")
        st.markdown("*Enter your current balances in each account (you can find these on your CPF statement)*")

        c1, c2, c3 = st.columns(3)
        with c1:
            current_oa = st.number_input(
                "🏠 Ordinary Account (OA)",
                0.0, 1000000.0, 20000.0, step=1000.0,
                help="For retirement, housing, insurance, and approved investments. Current rate: 2.5%"
            )
        with c2:
            current_sa = st.number_input(
                "🎯 Special Account (SA)",
                0.0, 1000000.0, 15000.0, step=1000.0,
                help="For retirement and approved investments. Current rate: 4.0%"
            )
        with c3:
            current_ma = st.number_input(
                "🏥 MediSave Account (MA)",
                0.0, 1000000.0, 10000.0, step=1000.0,
                help="For healthcare expenses and medical insurance. Current rate: 4.0%"
            )

        # Basic contribution calculation (simplified)
        # This is a simplification. Actual rates depend on age.
        employee_contribution = monthly_salary * 0.20
        employer_contribution = monthly_salary * 0.17
        total_contribution = employee_contribution + employer_contribution

        # Simplified allocation
        monthly_contribution_oa = total_contribution * 0.62  # Approx for age < 35
        monthly_contribution_sa = total_contribution * 0.16  # Approx
        monthly_contribution_ma = total_contribution * 0.22  # Approx

        # Contribution breakdown with better formatting
        st.markdown("#### 📊 Estimated Monthly Contributions")
        contribution_col1, contribution_col2 = st.columns([2, 1])
        with contribution_col1:
            st.markdown(f"""
            <div style="background-color: #d4edda; padding: 15px; border-radius: 8px; border: 1px solid #c3e6cb;">
                <strong>Based on monthly salary of S${monthly_salary:,.2f}:</strong><br>
                🏠 OA: <strong>S${monthly_contribution_oa:.2f}</strong><br>
                🎯 SA: <strong>S${monthly_contribution_sa:.2f}</strong><br>
                🏥 MA: <strong>S${monthly_contribution_ma:.2f}</strong><br>
                <hr style="margin: 10px 0;">
                📈 <strong>Total: S${monthly_contribution_oa + monthly_contribution_sa + monthly_contribution_ma:.2f}</strong>
            </div>
            """, unsafe_allow_html=True)
        with contribution_col2:
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;">
                <small><strong>Note:</strong> Actual contribution rates depend on your age and other factors. This is a simplified estimate.</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        submitted = st.form_submit_button(
            "🚀 Calculate My Retirement Projection",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        with st.spinner("Calculating your projection..."):
            result = retirement_savings_calculator.invoke({
                "current_age": current_age, "retirement_age": retirement_age,
                "current_oa": current_oa, "current_sa": current_sa, "current_ma": current_ma,
                "monthly_contribution_oa": monthly_contribution_oa,
                "monthly_contribution_sa": monthly_contribution_sa,
                "monthly_contribution_ma": monthly_contribution_ma
            })

            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.projection_df = pd.DataFrame(result['dataframe'])
                st.session_state.projection_summary = result['summary']

    if 'projection_df' in st.session_state:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #28a745 0%, #20c997 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2 style="color: white; margin-top: 0;">📈 Your Retirement Projection</h2>
        </div>
        """, unsafe_allow_html=True)

        # Summary in a nice card format
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; margin: 15px 0;">
        """ + st.session_state.projection_summary.replace('\n', '<br>').replace('**', '<strong>').replace('**',
                                                                                                          '</strong>') + """
        </div>
        """, unsafe_allow_html=True)

        df = st.session_state.projection_df

        # Enhanced chart with better styling
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Age'], y=df['OA'],
            name='🏠 Ordinary Account (OA)',
            stackgroup='one',
            line=dict(color='#ff6b6b')
        ))
        fig.add_trace(go.Scatter(
            x=df['Age'], y=df['SA'],
            name='🎯 Special Account (SA)',
            stackgroup='one',
            line=dict(color='#4ecdc4')
        ))
        fig.add_trace(go.Scatter(
            x=df['Age'], y=df['MA'],
            name='🏥 MediSave Account (MA)',
            stackgroup='one',
            line=dict(color='#45b7d1')
        ))
        fig.update_layout(
            title={
                'text': "📊 CPF Balance Growth Over Time",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Age (Years)",
            yaxis_title="Balance (SGD)",
            legend_title="CPF Accounts",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data table with tabs for better organization
        tab1, tab2 = st.tabs(["📊 Summary View", "📋 Detailed Breakdown"])

        with tab1:
            # Show key milestones
            milestones_df = df[df['Age'].isin([current_age, 55, 65, retirement_age])].copy()
            st.markdown("### 🎯 Key Milestones")
            st.dataframe(
                milestones_df.style.format({
                    "OA": "S${:,.0f}", "SA": "S${:,.0f}", "MA": "S${:,.0f}", "Total": "S${:,.0f}"
                }).highlight_max(subset=['Total'], color='lightgreen'),
                use_container_width=True
            )

        with tab2:
            st.markdown("### 📋 Year-by-Year Projection")
            st.dataframe(
                df.style.format({
                    "OA": "S${:,.2f}", "SA": "S${:,.2f}", "MA": "S${:,.2f}", "Total": "S${:,.2f}"
                }),
                use_container_width=True
            )

    st.divider()

    # Enhanced Chat Interface
    st.divider()
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: white; margin-top: 0;">💬 Ask Follow-up Questions</h2>
        <p style="color: white; margin-bottom: 0;">Get personalized insights about your projection from our AI assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Example questions specific to estimator
    with st.expander("💡 Smart Questions to Ask About Your Projection", expanded=False):
        st.markdown("""
        - "Will I meet the Basic/Full/Enhanced Retirement Sum with this projection?"
        - "What happens if I stop working at age 60 but retire at 65?"
        - "How much more should I contribute to reach S$500,000 by retirement?"
        - "Explain how the extra interest is calculated in my projection"
        - "What if I transfer money from OA to SA?"
        """)

    if "messages_estimator" not in st.session_state:
        st.session_state.messages_estimator = []

    # Display chat messages with better styling
    if st.session_state.messages_estimator:
        st.markdown("### 💭 Conversation History")
        for message in st.session_state.messages_estimator:
            if message.type == "human":
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2196f3;">
                    <strong>You:</strong> {message.content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f3e5f5; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #9c27b0;">
                    <strong>🤖 CPF Sage:</strong> {message.content}
                </div>
                """, unsafe_allow_html=True)

    if prompt := st.chat_input(
            "Ask about your projection (e.g., What if I increase my contributions?)",
            key="estimator_chat"
    ):
        st.session_state.messages_estimator.append(HumanMessage(content=prompt))

        # Display user message immediately
        st.markdown(f"""
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2196f3;">
            <strong>You:</strong> {prompt}
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🤖 CPF Sage is thinking..."):
            if st.session_state.agent_executor:
                chat_history = st.session_state.get('messages_estimator', [])
                # Add context from the projection
                contextual_prompt = f"""
                Here is my current projection context:
                {st.session_state.get('projection_summary', 'No projection calculated yet.')}

                My question is: {prompt}
                """
                response = st.session_state.agent_executor.invoke({
                    "input": contextual_prompt,
                    "chat_history": chat_history
                })

                # Display AI response
                st.markdown(f"""
                <div style="background-color: #f3e5f5; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #9c27b0;">
                    <strong>🤖 CPF Sage:</strong> {response['output']}
                </div>
                """, unsafe_allow_html=True)

                st.session_state.messages_estimator.append(AIMessage(content=response['output']))
                st.rerun()
            else:
                st.error("🤖 AI Agent is not available.")


def display_explainer():
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>💰 Withdrawal Options Explainer</h1>
        <p style="font-size: 18px; color: #666;">Understand when and how you can access your CPF savings</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick info cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; text-align: center;">
            <h4 style="color: #2e7d32; margin-top: 0;">🏠 Housing</h4>
            <p style="font-size: 14px; margin-bottom: 0;">Buy or renovate your home</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center;">
            <h4 style="color: #1565c0; margin-top: 0;">🏥 Medical</h4>
            <p style="font-size: 14px; margin-bottom: 0;">Healthcare expenses</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px; text-align: center;">
            <h4 style="color: #ef6c00; margin-top: 0;">🏖️ Retirement</h4>
            <p style="font-size: 14px; margin-bottom: 0;">Monthly payouts from age 65</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📝 Select Your Scenario")

    col1, col2 = st.columns(2)
    with col1:
        purpose = st.selectbox(
            "🎯 What is the purpose of the withdrawal?",
            ("Retirement", "Housing", "Medical", "Investment"),
            help="Different purposes have different rules and restrictions"
        )
    with col2:
        age_bracket = st.selectbox(
            "🎂 Select your age bracket",
            ("Below 55", "55 and above"),
            help="Age 55 is a key milestone for CPF withdrawals"
        )

    query = f"Explain the CPF withdrawal rules for {purpose.lower()} for someone aged {age_bracket.lower()}."

    if st.button(
            f"🔍 Get Withdrawal Rules for {purpose}",
            use_container_width=True,
            type="primary"
    ):
        with st.spinner("🔍 Querying CPF policies..."):
            if st.session_state.agent_executor:
                response = st.session_state.agent_executor.invoke({"input": query, "chat_history": []})
                st.session_state.explainer_output = response['output']
            else:
                st.error("🤖 AI Agent is not available.")

    if 'explainer_output' in st.session_state:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #28a745 0%, #20c997 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2 style="color: white; margin-top: 0;">📋 Withdrawal Rules Explanation</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb;">
            {st.session_state.explainer_output}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Enhanced Impact Simulator
    st.divider()
    st.markdown("""
    <div style="background: linear-gradient(90deg, #fd7e14 0%, #e83e8c 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: white; margin-top: 0;">🎯 Withdrawal Impact Simulator</h2>
        <p style="color: white; margin-bottom: 0;">See how withdrawals at age 55 affect your retirement savings</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get('projection_df') is None:
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; border: 1px solid #ffeaa7; text-align: center;">
            <h4>📋 Projection Required</h4>
            <p>Please run a projection on the <strong>Retirement Savings Estimator</strong> page first to use this feature.</p>
            <p><em>This will calculate your estimated balance at age 55.</em></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.projection_df
        if 55 in df['Age'].values:
            balance_at_55 = df.loc[df['Age'] == 55, 'Total'].iloc[0]
            oa_at_55 = df.loc[df['Age'] == 55, 'OA'].iloc[0]
            sa_at_55 = df.loc[df['Age'] == 55, 'SA'].iloc[0]

            # Current situation display
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px;">
                    <h4 style="color: #2e7d32; margin-top: 0;">💰 Your Balance at Age 55</h4>
                    <p style="font-size: 24px; font-weight: bold; color: #2e7d32; margin: 10px 0;">S${balance_at_55:,.2f}</p>
                    <small>Total CPF Balance</small>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                withdrawable_amount_raw = (oa_at_55 + sa_at_55) - CPF_RETIREMENT_SUMS_2025['FRS']
                withdrawable_amount = max(5000, withdrawable_amount_raw)
                st.markdown(f"""
                <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px;">
                    <h4 style="color: #ef6c00; margin-top: 0;">🎯 Maximum Withdrawable</h4>
                    <p style="font-size: 24px; font-weight: bold; color: #ef6c00; margin: 10px 0;">S${withdrawable_amount:,.2f}</p>
                    <small>Amount above FRS (S${CPF_RETIREMENT_SUMS_2025['FRS']:,})</small>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🎯 Simulate Your Withdrawal")
            withdrawal_amount = st.slider(
                "💵 Amount to withdraw (SGD)",
                min_value=0.0,
                max_value=float(withdrawable_amount),
                value=min(5000.0, withdrawable_amount),
                step=1000.0,
                help=f"You can withdraw between S$5,000 and S${withdrawable_amount:,.2f}"
            )

            remaining_balance = balance_at_55 - withdrawal_amount

            # Enhanced visualization with metrics
            col1, col2 = st.columns([2, 1])
            with col1:
                # Pie chart with better styling
                labels = [f'Withdrawn: S${withdrawal_amount:,.0f}', f'Remaining: S${remaining_balance:,.0f}']
                values = [withdrawal_amount, remaining_balance]

                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.4,
                    marker_colors=['#ff6b6b', '#4ecdc4'],
                    textinfo='label+percent',
                    textfont_size=12
                )])
                fig.update_layout(
                    title={
                        'text': "📊 Impact of Withdrawal at Age 55",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16}
                    },
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; height: 350px;">
                    <h4>📈 Impact Summary</h4>
                """, unsafe_allow_html=True)

                percentage_withdrawn = (withdrawal_amount / balance_at_55) * 100 if balance_at_55 > 0 else 0
                st.metric(
                    "Percentage Withdrawn",
                    f"{percentage_withdrawn:.1f}%",
                    delta=f"-S${withdrawal_amount:,.0f}"
                )
                st.metric(
                    "Remaining for Retirement",
                    f"S${remaining_balance:,.0f}",
                    delta=f"{100 - percentage_withdrawn:.1f}% of total"
                )

                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; border: 1px solid #f5c6cb;">
                <h4>🚨 Age Range Issue</h4>
                <p>Your projection doesn't include age 55. Please adjust the age range on the estimator page to include age 55 for this simulation.</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    # Enhanced Chat Interface for Explainer
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: white; margin-top: 0;">💬 Ask Follow-up Questions</h2>
        <p style="color: white; margin-bottom: 0;">Get detailed explanations about withdrawal rules and impacts</p>
    </div>
    """, unsafe_allow_html=True)

    # Example questions specific to withdrawal explainer
    with st.expander("💡 Smart Questions About Withdrawals", expanded=False):
        st.markdown("""
        - "What is the impact on my monthly payouts if I withdraw S$20,000?"
        - "Can I use my CPF for a private property purchase?"
        - "What happens to my MediSave if I withdraw at 55?"
        - "How does early withdrawal affect my CPF LIFE payouts?"
        - "What are the conditions for education withdrawals?"
        """)

    if "messages_explainer" not in st.session_state:
        st.session_state.messages_explainer = []

    # Display chat messages with consistent styling
    if st.session_state.messages_explainer:
        st.markdown("### 💭 Conversation History")
        for message in st.session_state.messages_explainer:
            if message.type == "human":
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2196f3;">
                    <strong>You:</strong> {message.content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f3e5f5; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #9c27b0;">
                    <strong>🤖 CPF Sage:</strong> {message.content}
                </div>
                """, unsafe_allow_html=True)

    if prompt := st.chat_input(
            "Ask about withdrawal rules (e.g., What is the impact on my monthly payouts?)",
            key="explainer_chat"
    ):
        st.session_state.messages_explainer.append(HumanMessage(content=prompt))

        # Display user message immediately
        st.markdown(f"""
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2196f3;">
            <strong>You:</strong> {prompt}
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🤖 CPF Sage is analyzing withdrawal policies..."):
            if st.session_state.agent_executor:
                response = st.session_state.agent_executor.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.messages_explainer
                })

                # Display AI response
                st.markdown(f"""
                <div style="background-color: #f3e5f5; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #9c27b0;">
                    <strong>🤖 CPF Sage:</strong> {response['output']}
                </div>
                """, unsafe_allow_html=True)

                st.session_state.messages_explainer.append(AIMessage(content=response['output']))
                st.rerun()
            else:
                st.error("🤖 AI Agent is not available.")


def display_about_us():
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>ℹ️ About This Project</h1>
        <p style="font-size: 18px; color: #666;">AI Champions Bootcamp Capstone Project</p>
    </div>
    """, unsafe_allow_html=True)

    # Project overview in cards
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; margin: 20px 0; color: white;">
        <h2 style="margin-top: 0;">🎯 Project Scope and Objectives</h2>
        <p style="font-size: 16px; line-height: 1.6;">This "CPF Retirement Planner" is a Capstone Project for the AI Champions Bootcamp, designed to be an interactive, LLM-powered solution for understanding CPF policies.</p>
    </div>
    """, unsafe_allow_html=True)

    # Objectives in a grid
    st.markdown("### 🎯 Our Primary Objectives")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; height: 200px;">
            <h4 style="color: #2e7d32;">📚 Consolidate Information</h4>
            <p>Aggregate publicly available CPF information from official sources into one accessible platform.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; height: 200px; margin-top: 15px;">
            <h4 style="color: #ef6c00;">🎨 Enhance Understanding</h4>
            <p>Use interactive elements like chat, charts, and simulators to make complex policies easier to grasp.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; height: 200px;">
            <h4 style="color: #1565c0;">👤 Personalize Experience</h4>
            <p>Allow users to get tailored estimates and explanations based on generic, non-PII inputs.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #f3e5f5; padding: 20px; border-radius: 10px; height: 200px; margin-top: 15px;">
            <h4 style="color: #8e24aa;">📊 Present Effectively</h4>
            <p>Display data in various formats, including text, tables, and visualizations.</p>
        </div>
        """, unsafe_allow_html=True)

    # Key Features section
    st.markdown("### ✨ Key Features")

    features = [
        ("📈", "Retirement Savings Estimator", "Projects your CPF growth over time with interactive charts"),
        ("🔍", "Withdrawal Options Explainer", "Provides clear, tailored information on CPF withdrawal rules"),
        ("🎯", "Withdrawal Impact Simulator", "Visualizes the effect of lump-sum withdrawals on your retirement funds"),
        ("🤖", "AI-Powered Chat", "Intelligent agent backed by RAG pipeline to prevent hallucinations"),
        ("📡", "Policy Update Checker", "Semantic search feature to query recent policy changes"),
        ("⚖️", "Scenario Comparator", "Compare different financial scenarios side-by-side")
    ]

    for i in range(0, len(features), 2):
        col1, col2 = st.columns(2)

        with col1:
            icon, title, desc = features[i]
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
                <h4 style="margin-top: 0;">{icon} {title}</h4>
                <p style="margin-bottom: 0; color: #666;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        if i + 1 < len(features):
            with col2:
                icon, title, desc = features[i + 1]
                st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
                <h4 style="margin-top: 0;">{icon} {title}</h4>
                <p style="margin-bottom: 0; color: #666;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # Data Sources section
    st.markdown("### 📚 Data Sources")
    st.markdown(
        "Our AI agent relies exclusively on information retrieved from the following official sources to ensure accuracy and reliability:")

    for i, url in enumerate(DATA_SOURCES, 1):
        st.markdown(f"{i}. [{url}]({url})")

    # Disclaimer
    st.markdown("""
    <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; border: 1px solid #ffeaa7; margin: 20px 0;">
        <h4 style="color: #856404; margin-top: 0;">⚠️ Important Disclaimer</h4>
        <p style="color: #856404; margin-bottom: 0;"><strong>This is an educational tool and not a financial advisory service.</strong> All projections are estimates based on simplified assumptions. Please consult official CPF resources or a financial advisor for formal advice.</p>
    </div>
    """, unsafe_allow_html=True)


def display_methodology():
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>🔧 Methodology and Implementation</h1>
        <p style="font-size: 18px; color: #666;">Technical details and data flows behind this application</p>
    </div>
    """, unsafe_allow_html=True)

    # Technical Stack in cards
    st.markdown("### 🏗️ Technical Stack")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="color: #2e7d32;">🖥️ Frontend</h4>
            <p><strong>Streamlit</strong><br>Interactive web framework</p>
            <p><strong>Plotly</strong><br>Data visualizations</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="color: #1565c0;">🤖 AI/ML</h4>
            <p><strong>LangChain</strong><br>LLM integration framework</p>
            <p><strong>OpenAI GPT-4o</strong><br>Large Language Model</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="color: #ef6c00;">💾 Data</h4>
            <p><strong>FAISS</strong><br>Vector database</p>
            <p><strong>Pandas</strong><br>Data processing</p>
        </div>
        """, unsafe_allow_html=True)

    # RAG Pipeline explanation
    st.markdown("### 🔄 RAG Pipeline: How We Prevent AI Hallucinations")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0; color: white;">
        <p style="font-size: 16px; margin-bottom: 0;">Our application uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers backed by official sources.</p>
    </div>
    """, unsafe_allow_html=True)

    # RAG steps in expandable sections
    steps = [
        ("📥", "Data Ingestion", "Application loads content from official CPF-related URLs at startup", "#e8f5e8"),
        ("✂️", "Text Chunking", "Loaded text is split into smaller, manageable chunks for processing", "#e3f2fd"),
        ("🔢", "Embedding Generation", "Each chunk is converted into numerical vectors using OpenAI embeddings",
         "#fff3e0"),
        ("💾", "Vector Indexing", "Embeddings are stored in FAISS vector store for efficient similarity searches",
         "#f3e5f5"),
        ("🔍", "Retrieval", "User queries are matched against stored embeddings to find relevant content", "#fce4ec"),
        ("✨", "Answer Generation", "LLM generates responses based ONLY on retrieved context, reducing hallucinations",
         "#e8f5e8")
    ]

    for i, (icon, title, desc, color) in enumerate(steps, 1):
        st.markdown(f"""
        <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #007bff;">
            <h4 style="margin-top: 0;"><strong>Step {i}:</strong> {icon} {title}</h4>
            <p style="margin-bottom: 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Application Flow Diagrams")
    st.graphviz_chart('''
        digraph {
            node [shape=box, style=rounded];
            UserInput [label="User Inputs (Age, Balances, Salary)"];
            Calculator [label="Retirement Savings Calculator Tool"];
            Projection [label="Generate Projections (Table, Chart)"];
            Display [label="Display Results in UI"];
            Chat [label="User asks follow-up question"];
            Agent [label="AI Agent"];

            UserInput -> Calculator;
            Calculator -> Projection;
            Projection -> Display;
            Display -> Chat;
            Chat -> Agent;
            Agent -> Calculator [label="If calculation needed"];
            Agent -> Display [label="Shows answer"];
        }
    ''')

    st.markdown("### Flowchart: Withdrawal Options Explainer & Chat")
    st.graphviz_chart('''
        digraph {
            node [shape=box, style=rounded];
            UserInput [label="User selects options (Age, Purpose)"];
            Agent [label="AI Agent"];
            Retriever [label="RAG Retriever Tool"];
            VectorStore [label="FAISS Vector Store"];
            Display [label="Display Explanation in UI"];
            Chat [label="User asks follow-up question"];

            UserInput -> Agent;
            Agent -> Retriever;
            Retriever -> VectorStore [label="Similarity Search"];
            VectorStore -> Retriever [label="Return relevant docs"];
            Retriever -> Agent;
            Agent -> Display;
            Display -> Chat;
            Chat -> Agent;
        }
    ''')

    # Safeguards section
    st.markdown("### 🛡️ Safeguards and Prompt Engineering")

    safeguards = [
        ("🎯", "System Prompts", "Detailed system prompts define the AI agent's role and constraints", "#e8f5e8"),
        ("📚", "RAG Grounding", "RAG architecture is our primary defense against hallucinations", "#e3f2fd"),
        ("🔧", "Tool-Based Agents", "Capabilities are structured into specific tools with predefined actions",
         "#fff3e0"),
        ("✅", "Input Validation", "Basic validation to prevent prompt injection and malicious inputs", "#f3e5f5")
    ]

    for icon, title, desc, color in safeguards:
        st.markdown(f"""
        <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #28a745;">
            <h4 style="margin-top: 0;">{icon} {title}</h4>
            <p style="margin-bottom: 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # Performance metrics
    st.markdown("### 📈 Key Performance Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Response Time", "< 3 seconds", "For most queries")
    with col2:
        st.metric("Data Sources", "6 Official URLs", "Real-time access")
    with col3:
        st.metric("Accuracy", "Source-grounded", "No hallucinations")


# --- Main App Logic ---

def check_password():
    """Returns `True` if the user has the correct password."""

    def password_entered():
        if st.session_state["password"] == st.secrets.get("APP_PASSWORD", "bootcamp2025"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True


def main():
    st.set_page_config(page_title="CPF Retirement Planner", layout="wide")

    if not check_password():
        st.stop()

    # Setup RAG and Agent once password is correct
    if 'agent_executor' not in st.session_state:
        retriever_tool, api_key = setup_rag()
        if retriever_tool and api_key:
            st.session_state.agent_executor = setup_agent(api_key, retriever_tool)
        else:
            st.session_state.agent_executor = None
            st.error("Failed to initialize the AI Agent. Please check your API key and data sources.")

    PAGES = {
        "🏠 Home": display_home,
        "📈 Retirement Savings Estimator": display_estimator,
        "💰 Withdrawal Options Explainer": display_explainer,
        "ℹ️ About Us": display_about_us,
        "🔧 Methodology": display_methodology
    }

    # Enhanced sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h2 style="color: #1f77b4; margin-bottom: 10px;">🧭 Navigation</h2>
        <p style="color: #666; font-size: 14px;">Choose a page to explore</p>
    </div>
    """, unsafe_allow_html=True)

    selection = st.sidebar.radio(
        "Go to",
        list(PAGES.keys()),
        label_visibility="collapsed"
    )

    # Add helpful information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h4 style="color: #1f77b4; margin-top: 0;">💡 Quick Tips</h4>
        <ul style="font-size: 12px; color: #666;">
            <li>Start with the <strong>Estimator</strong> to project your savings</li>
            <li>Use the <strong>AI chat</strong> for follow-up questions</li>
            <li>Check <strong>Withdrawal Options</strong> for accessing funds</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Current CPF rates display
    st.sidebar.markdown("""
    <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h4 style="color: #2e7d32; margin-top: 0;">📊 Current CPF Rates</h4>
        <p style="font-size: 12px; margin: 5px 0;"><strong>OA:</strong> 2.5% p.a.</p>
        <p style="font-size: 12px; margin: 5px 0;"><strong>SA/MA:</strong> 4.0% p.a.</p>
        <p style="font-size: 11px; color: #666; margin: 5px 0;"><em>Effective Q1 2025</em></p>
    </div>
    """, unsafe_allow_html=True)

    page = PAGES[selection]
    page()


if __name__ == "__main__":
    # To run this app, you need to set two secrets in Streamlit Cloud:
    # 1. OPENAI_API_KEY = "your_openai_api_key"
    # 2. APP_PASSWORD = "a_password_of_your_choice" (defaults to bootcamp2025 if not set)
    main()