# 🏦 CPF Retirement Planner

An AI-powered interactive web application for understanding Singapore's CPF (Central Provident Fund) policies and planning retirement savings. Built as a capstone project for the AI Champions Bootcamp.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Testing Questions](#testing-questions)
- [Security & Privacy](#security--privacy)
- [Limitations & Disclaimers](#limitations--disclaimers)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The CPF Retirement Planner is an educational prototype that demonstrates the power of AI in making complex financial information accessible and interactive. It combines:

- **Retrieval-Augmented Generation (RAG)** to provide accurate, source-grounded responses
- **Interactive visualizations** for retirement planning
- **AI-powered chat assistant** for personalized guidance
- **Real-time calculations** based on current CPF policies

### 🎯 Project Objectives

1. **Information Consolidation**: Aggregate publicly available CPF information from official sources
2. **Enhanced Understanding**: Use interactive elements to make complex policies accessible
3. **Personalized Experience**: Provide tailored estimates based on user inputs
4. **Effective Presentation**: Display data through various formats (text, charts, tables)

## ✨ Features

### 📊 Retirement Savings Estimator
- **Interactive Projections**: Calculate future CPF balances with compound interest
- **Visual Charts**: Plotly-powered graphs showing account growth over time
- **Milestone Tracking**: Key ages (55, 65, retirement age) with balance projections
- **Contribution Analysis**: Breakdown of monthly contributions across OA, SA, and MA

### 💰 Withdrawal Options Explainer
- **Scenario-Based Guidance**: Tailored explanations based on age and purpose
- **Impact Simulator**: Visualize how withdrawals affect retirement savings
- **Rule Clarification**: Clear explanations of CPF withdrawal policies
- **Interactive Chat**: Ask follow-up questions about withdrawal scenarios

### 🤖 AI-Powered Chat Assistant
- **RAG-Enhanced Responses**: Grounded in official CPF documentation
- **Context-Aware**: Remembers conversation history and projection context
- **Multi-Tool Integration**: Combines policy retrieval with calculation tools
- **Hallucination Prevention**: Strict adherence to source materials

### 📈 Interactive Visualizations
- **Stacked Area Charts**: Show CPF account growth over time
- **Pie Charts**: Visualize withdrawal impact on total savings
- **Data Tables**: Detailed year-by-year breakdowns
- **Metrics Dashboard**: Key performance indicators

## 🏗️ Technology Stack

### Frontend
- **Streamlit**: Interactive web framework for rapid prototyping
- **Plotly**: Advanced data visualization library
- **HTML/CSS**: Custom styling for enhanced user experience

### AI/ML Backend
- **LangChain**: LLM integration and agent framework
- **OpenAI GPT-4o**: Large Language Model for natural language processing
- **FAISS**: Vector database for efficient similarity search
- **OpenAI Embeddings**: Text embedding for semantic search

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **BeautifulSoup**: Web scraping for data ingestion

### Deployment
- **Streamlit Cloud**: Hosting platform
- **Environment Variables**: Secure API key management

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Internet connection for data ingestion

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI_Champion_Bootcamp_Assignment
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.streamlit/secrets.toml` file:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key_here"
   APP_PASSWORD = "your_chosen_password"
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. **Push to GitHub**: Ensure your repository is on GitHub
2. **Connect to Streamlit Cloud**: Link your GitHub repository
3. **Configure Secrets**: Add the following secrets in Streamlit Cloud:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `APP_PASSWORD`: Password for app access (defaults to "bootcamp2025")

## 📖 Usage Guide

### 🏠 Home Page
- **AI Policy Assistant**: Ask general questions about CPF policies
- **Quick Navigation**: Access all features from the sidebar
- **Example Questions**: Get started with suggested queries

### 📊 Retirement Savings Estimator
1. **Enter Personal Details**: Age, retirement age, monthly salary
2. **Input Current Balances**: OA, SA, and MA account balances
3. **Review Projections**: View charts and detailed breakdowns
4. **Ask Follow-up Questions**: Use the chat interface for personalized insights

### 💰 Withdrawal Options Explainer
1. **Select Scenario**: Choose withdrawal purpose and age bracket
2. **Get Rules**: Receive tailored policy explanations
3. **Simulate Impact**: Visualize withdrawal effects on retirement savings
4. **Explore Options**: Chat with AI about different scenarios

### 🤖 AI Chat Features
- **Policy Questions**: "How are CPF interest rates calculated?"
- **Calculation Requests**: "What if I increase my monthly contributions?"
- **Scenario Analysis**: "How does early withdrawal affect my payouts?"
- **Rule Clarification**: "Can I use CPF for private property?"

## 🏛️ Architecture

### RAG Pipeline
```
User Query → Embedding → Vector Search → Context Retrieval → LLM Response
```

### Data Flow
1. **Data Ingestion**: Web scraping from official CPF sources
2. **Text Processing**: Chunking and embedding generation
3. **Vector Storage**: FAISS index for similarity search
4. **Query Processing**: Semantic search and context retrieval
5. **Response Generation**: LLM-based answer with source grounding

### Agent Architecture
- **Tools**: RAG retriever and calculation tools
- **Memory**: Conversation history management
- **Safeguards**: System prompts and input validation
- **Error Handling**: Graceful degradation for API failures

## 📚 Data Sources

The application relies exclusively on official CPF documentation:

1. [CPF Retirement Sum Scheme](https://www.cpf.gov.sg/member/faq/growing-your-savings/retirement-sum-scheme/what-is-the-basic-full-and-enhanced-retirement-sum)
2. [CPF Interest Rates](https://www.cpf.gov.sg/member/faq/growing-your-savings/cpf-interest-rates/how-are-cpf-interest-rates-determined)
3. [CPF Interest Rates Guide](https://growbeansprout.com/cpf-interest-rates)
4. [CPF Withdrawals from Age 55](https://www.cpf.gov.sg/member/faq/retirement-income/cpf-withdrawals-from-55/how-much-cpf-savings-can-i-withdraw-from-age-55)
5. [MOM CPF Information](https://www.mom.gov.sg/employment-practices/cpf-and-other-statutory-matters/central-provident-fund)
6. [Government CPF Overview](https://www.gov.sg/article/understanding-cpf)

## 🧪 Testing Questions

### 📊 Retirement Savings Estimator Testing

#### Basic Functionality
1. **Input Validation**
   - What happens when you enter a negative age?
   - How does the app handle retirement age less than current age?
   - What if you input zero balances in all accounts?
   - Test with very high salary values (>$50,000/month)

2. **Calculation Accuracy**
   - Verify compound interest calculations for a 5-year projection
   - Test with different age ranges (25-30, 45-50, 60-65)
   - Check if interest rates match current CPF rates (OA: 2.5%, SA/MA: 4%)
   - Validate contribution allocations across accounts

3. **Edge Cases**
   - What happens with age 55+ projections?
   - Test maximum age limits (70+ years)
   - Verify behavior with minimal contribution amounts
   - Check handling of decimal inputs

#### Visualization Testing
4. **Chart Functionality**
   - Do charts update correctly when inputs change?
   - Test chart responsiveness on different screen sizes
   - Verify hover information displays correctly
   - Check if milestone markers appear at correct ages

5. **Data Table Testing**
   - Verify year-by-year breakdown accuracy
   - Test table sorting and filtering (if applicable)
   - Check formatting of currency values
   - Validate milestone table shows correct key ages

### 💰 Withdrawal Options Explainer Testing

#### Policy Retrieval
6. **Scenario Coverage**
   - Test all withdrawal purposes (Retirement, Housing, Medical, Investment)
   - Verify age bracket responses (Below 55, 55 and above)
   - Check if explanations are appropriate for selected scenarios
   - Test edge cases like age 54 vs age 55

7. **Content Accuracy**
   - Verify policy information matches official sources
   - Check if withdrawal limits are correctly stated
   - Test explanation clarity and completeness
   - Validate rule citations and references

#### Impact Simulator
8. **Simulation Functionality**
   - Test withdrawal amount slider limits
   - Verify pie chart updates with different withdrawal amounts
   - Check percentage calculations accuracy
   - Test with zero withdrawal amount

9. **Integration Testing**
   - Verify simulator works with projection data
   - Test behavior when no projection exists
   - Check error handling for missing data
   - Validate age 55 requirement for simulation

### 🤖 AI Chat Assistant Testing

#### General Functionality
10. **Basic Chat Operations**
    - Test simple policy questions
    - Verify conversation history maintenance
    - Check response generation time
    - Test chat input validation

11. **RAG Effectiveness**
    - Ask questions not covered in source documents
    - Test if AI admits when information is not available
    - Verify responses are grounded in official sources
    - Check for hallucination prevention

#### Tool Integration
12. **Calculator Tool Usage**
    - Ask for retirement projections via chat
    - Test calculation requests with different parameters
    - Verify tool output integration in responses
    - Check error handling for invalid inputs

13. **Context Awareness**
    - Ask follow-up questions about projections
    - Test if AI remembers previous calculations
    - Verify contextual responses based on user data
    - Check conversation continuity

### 🔍 Policy-Specific Testing

#### Interest Rate Questions
14. **Rate Information**
    - "What are the current CPF interest rates?"
    - "How is extra interest calculated?"
    - "What is the floor rate for CPF accounts?"
    - "How often do CPF rates change?"

#### Retirement Sum Questions
15. **Sum Calculations**
    - "What are the retirement sums for 2025?"
    - "How do BRS, FRS, and ERS differ?"
    - "When are retirement sums reviewed?"
    - "What happens if I don't meet the FRS?"

#### Withdrawal Rules
16. **Access Conditions**
    - "When can I start withdrawing from CPF?"
    - "What are the conditions for housing withdrawals?"
    - "Can I use CPF for private property?"
    - "What are the medical withdrawal rules?"

#### CPF LIFE Questions
17. **Annuity Information**
    - "How does CPF LIFE work?"
    - "What are the different CPF LIFE plans?"
    - "When do CPF LIFE payouts start?"
    - "Can I opt out of CPF LIFE?"

### 🎯 Advanced Testing Scenarios

#### Complex Calculations
18. **Multi-Scenario Analysis**
    - "What if I transfer money from OA to SA?"
    - "How much more should I contribute to reach $500k?"
    - "What's the impact of stopping work at 60?"
    - "Compare scenarios with different retirement ages"

#### Policy Updates
19. **Current Information**
    - "What are the latest CPF changes?"
    - "When was the last interest rate review?"
    - "Are there any upcoming policy changes?"
    - "How do recent changes affect my planning?"

#### Error Handling
20. **Robustness Testing**
    - Test with network connectivity issues
    - Verify API failure handling
    - Check timeout scenarios
    - Test with invalid API keys

### 📱 User Experience Testing

#### Interface Testing
21. **Navigation**
    - Test all sidebar navigation options
    - Verify page transitions
    - Check responsive design on mobile
    - Test form submission and validation

22. **Accessibility**
    - Test keyboard navigation
    - Verify color contrast compliance
    - Check screen reader compatibility
    - Test with different font sizes

#### Performance Testing
23. **Load Times**
    - Measure initial page load time
    - Test calculation response time
    - Verify chart rendering speed
    - Check chat response latency

24. **Scalability**
    - Test with multiple concurrent users
    - Verify memory usage patterns
    - Check API rate limiting
    - Test with large datasets

### 🔒 Security Testing

#### Input Validation
25. **Security Measures**
    - Test for SQL injection attempts
    - Verify XSS prevention
    - Check input sanitization
    - Test password protection

26. **Data Privacy**
    - Verify no PII is stored
    - Check session management
    - Test data retention policies
    - Verify secure API communication

## 🔒 Security & Privacy

### Data Protection
- **No PII Storage**: Application does not store personal information
- **Session-Based**: All data is temporary and session-scoped
- **Secure API**: OpenAI API communication is encrypted
- **Password Protection**: Basic access control for demo purposes

### Input Validation
- **Type Checking**: All inputs are validated for correct data types
- **Range Validation**: Age and financial inputs have reasonable limits
- **Sanitization**: User inputs are sanitized to prevent injection attacks
- **Error Handling**: Graceful error messages without exposing system details

## ⚠️ Limitations & Disclaimers

### Educational Purpose
This application is developed for educational purposes as part of the AI Champions Bootcamp. It is not intended for real-world financial planning or decision-making.

### Accuracy Limitations
- **Simplified Calculations**: Uses simplified contribution rates and assumptions
- **Policy Updates**: May not reflect the latest CPF policy changes
- **Individual Variations**: Does not account for personal circumstances
- **Professional Advice**: Not a substitute for qualified financial advice

### Technical Limitations
- **API Dependencies**: Requires OpenAI API access and internet connectivity
- **Performance**: Response times may vary based on API availability
- **Data Sources**: Limited to publicly available CPF information
- **Scalability**: Designed for demonstration, not production use

## 🤝 Contributing

This is an educational project, but contributions are welcome:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for any changes
- Ensure all tests pass before submitting

## 📄 License

This project is developed for educational purposes as part of the AI Champions Bootcamp. Please refer to the bootcamp guidelines for usage permissions.

---

## 🎓 About the AI Champions Bootcamp

This project was developed as a capstone assignment for the AI Champions Bootcamp, demonstrating practical application of AI technologies in creating user-friendly financial planning tools.

**Disclaimer**: This is an educational prototype and should not be used for actual financial planning. Always consult with qualified professionals for financial advice.