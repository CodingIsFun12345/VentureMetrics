import os
import io
from flask import Flask, render_template, request, jsonify
from groq import Groq
import pandas as pd
import PyPDF2

# Initialize Flask app, pointing to the templates folder in the root directory
app = Flask(__name__, template_folder='../templates')

# Initialize Groq client. Ensure GROQ_API_KEY is set in your environment variables.
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

def extract_text_from_file(file):
    """Helper function to parse text from various file formats."""
    filename = file.filename.lower()
    try:
        if filename.endswith('.pdf'):
            pdf = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        elif filename.endswith('.csv'):
            df = pd.read_csv(file)
            return df.to_string()
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
            return df.to_string()
        else:
            # Fallback for standard text files
            return file.read().decode('utf-8')
    except Exception as e:
        return f"[Error parsing file: {str(e)}]"

def review_content_for_brand_alignment(draft_content, company_desc, company_focus, context_type):
    """
    Acts as an internal Brand Guardian to review and refine AI-generated drafts 
    ensuring they do not dilute the company's core identity before sending to the user.
    """
    review_prompt = f"""
    You are a Chief Brand Officer and Strategic Guardian.

    Company Background: {company_desc}
    Core Focus & Business Model: {company_focus}

    You are reviewing a drafted {context_type}. 
    CRITICAL TASK: Evaluate the drafted content below to ensure it aligns perfectly with the company's brand identity. 
    - If the brand is luxury, exclusive, highly specialized, or strictly in-person, you MUST REMOVE or REWRITE any suggestions that push mass-market, cheap, or misaligned digital/DTC strategies.
    - Ensure the tone and strategic recommendations strictly elevate the brand rather than dilute it.
    
    Drafted Content:
    {draft_content}

    Output the final, polished version of this content. Keep the exact same structural formatting (Markdown, headings, tables) as the draft, but ensure the strategic alignment is flawless.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": review_prompt}],
            model="openai/gpt-oss-120b", 
            max_tokens=4096
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        # If the review pass fails for any reason, return the original draft so the app doesn't break
        print(f"Brand Guardian Review Failed: {e}")
        return draft_content

# ==========================================
# PAGE ROUTES
# ==========================================

@app.route('/')
def home():
    """Renders the landing page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Renders the main dashboard for central data ingestion."""
    return render_template('dashboard.html')

@app.route('/expansion')
def expansion():
    """Renders the strategic expansion tool."""
    return render_template('expansion.html')

@app.route('/competitor')
def competitor():
    """Renders the competitor positioning tool."""
    return render_template('competitor.html')

# ==========================================
# API ROUTES
# ==========================================

@app.route('/api/extract_context', methods=['POST'])
def extract_context():
    """Endpoint to extract text from a file and return it so the frontend can store it across pages."""
    file = request.files.get('file')
    if file and file.filename != '':
        text = extract_text_from_file(file)
        return jsonify({"success": True, "data_context": text})
    return jsonify({"success": False, "error": "No file data provided."})

@app.route('/api/generate_strategy', methods=['POST'])
def generate_strategy():
    """
    Unified endpoint for handling strategy generation based on the active module.
    Runs the initial draft generation followed by the Brand Guardian review pass.
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No JSON data provided."}), 400

    # Extract global context
    module = data.get('module')
    company_desc = data.get('company_desc', 'Not provided')
    company_focus = data.get('company_focus', 'Not provided')
    data_context = data.get('data_context', 'No file data provided.')

    try:
        # ---------------------------------------------------------
        # MODULE: EXPANSION
        # ---------------------------------------------------------
        if module == 'expansion':
            target = data.get('target', 'Unknown Target')
            goal = data.get('goal', 'Unknown Goal')
            
            prompt = f"""
            You are a Senior Operational & Financial Strategist.

            Company Background: {company_desc}
            Core Focus & Business Model: {company_focus}
            Financial Data Context: {data_context[:6000]}
            
            Target Market/Region: {target}
            Strategic Objective: {goal}

            CRITICAL FINANCIAL RULE: Analyze the provided financial data. EVERY recommendation MUST BE strictly financially feasible for this company.

            Format your output strictly as a highly polished, professional **BUSINESS CASE PRESENTATION DOCUMENT**. 
            Use clear Markdown headings, bullet points, and perfectly structured tables.
            
            # Strategic Expansion Blueprint: {target}
            
            ## 1. Executive Summary & Market Feasibility
            Provide a corporate overview of the business's capacity to enter {target} and achieve the objective: {goal}.

            ## 2. Operational Blueprint & Milestones
            Breakdown of deployment into structured phases (Phase 1: Setup/Pre-launch, Phase 2: Launch, Phase 3: Scaling). Use bullet points for readability.

            ## 3. Granular Budget & Capital Allocation
            A detailed cost breakdown table demonstrating exactly how funds are deployed safely without breaking liquidity.
            | Expense Category | Description | Estimated Allocation | Timeline |
            | --- | --- | --- | --- |

            ## 4. Key Performance Indicators (KPIs) & Risk Guardrails
            Metrics to track monthly to verify ROI and financial safety.
            """
            context_type = "Strategic Expansion Blueprint"

        # ---------------------------------------------------------
        # MODULE: COMPETITOR
        # ---------------------------------------------------------
        elif module == 'competitor':
            competitor_name = data.get('competitor', 'Unknown Competitor')
            concern = data.get('concern', 'Unknown Concern')
            
            prompt = f"""
            You are a Lead Market Research Analyst and Competitive Strategy Consultant.

            Company Description: {company_desc}
            Core Business Focus & Model: {company_focus}
            Financial & Operational Context: {data_context[:6000]}
            
            Target Competitor / Market Segment: {competitor_name}
            Primary Strategic Concern: {concern}

            Perform a thorough Competitor Positioning and Market Dynamics Analysis.
            Format your output strictly as a highly polished, professional **PRESENTATION DOCUMENT** in Markdown.

            # Competitor Intelligence Report: {competitor_name}
            
            ## 1. Key Strategic Takeaways
            * **Core Threat:** Deep analysis of the concern: {concern}.
            * **Top Market Opportunities:** High-leverage gaps the competitor is leaving open.
            * **Core Competitive Edge:** Your key operational/financial advantage over them.

            ## 2. Competitive Positioning Matrix
            Construct a professional Markdown table comparing this business against {competitor_name}:
            | Category | {competitor_name} | Our Business | Strategic Counter-Positioning |
            | --- | --- | --- | --- |

            ## 3. Strategic Transformation & Differentiation Roadmap
            Provide actionable steps for how this business can defend against the primary concern and capture market share. Use subheadings and bullet points for:
            * Product & Service Differentiation
            * Pricing & Packaging Strategy
            * Go-To-Market Execution
            """
            context_type = "Competitor Positioning & Market Research Report"
            
        else:
            return jsonify({"success": False, "error": "Invalid module type specified."}), 400

        # ==========================================
        # EXECUTE TWO-PASS GENERATION
        # ==========================================
        
        # Pass 1: Generate Draft
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b", 
            max_tokens=4096
        )
        draft_text = chat_completion.choices[0].message.content
        
        # Pass 2: Review for Brand Alignment
        final_text = review_content_for_brand_alignment(
            draft_content=draft_text, 
            company_desc=company_desc, 
            company_focus=company_focus, 
            context_type=context_type
        )
        
        return jsonify({
            "success": True, 
            "content": final_text
        })
        
    except Exception as e:
        print(f"API Error in generate_strategy: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)