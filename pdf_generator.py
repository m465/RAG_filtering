import os
import textwrap
from fpdf import FPDF

# ==========================================
# 1. Configuration & Content Definitions
# ==========================================

BASE_DIR = "all_docs"

# We define the specific text content for each document here.
# This ensures the text is semantically relevant to the category.
DATASET_CONFIG = {
    "SOPs": [
        {
            "filename": "SOP_01_Equipment_Safety.pdf",
            "title": "Standard Operating Procedure: Heavy Machinery Safety",
            "page1": """
            1. Purpose
            The purpose of this SOP is to define the safety guidelines for operating the X-200 Hydraulic Press. 
            All personnel must be trained and certified before operating this machinery.
            
            2. Scope
            This procedure applies to all production floor employees and maintenance staff.
            
            3. Prerequisites
            - Wear Personal Protective Equipment (PPE): Helmets, Gloves, Safety Goggles.
            - Ensure the emergency stop button is functional.
            - Verify that the voltage regulator is set to 240V.
            """,
            "page2": """
            4. Emergency Procedures
            In the event of a malfunction, immediately press the RED E-STOP button located on the main panel.
            Do not attempt to retrieve jammed materials with bare hands. Use the provided safety tongs.
            
            5. Reporting
            Any incidents, near-misses, or equipment damage must be reported to the Floor Manager within 15 minutes.
            Failure to report incidents is a violation of company safety policy.
            """
        },
        {
            "filename": "SOP_02_Quality_Control.pdf",
            "title": "SOP: Quality Assurance Inspection Process",
            "page1": """
            1. Introduction
            This document outlines the steps for the final inspection of manufactured widgets.
            
            2. Inspection Criteria
            - Surface Finish: Must be smooth with no visible scratches exceeding 1mm.
            - Dimensions: Tolerance levels must be within +/- 0.05mm of the schematic.
            - Durability: Random samples must pass the stress test of 50kg pressure.
            """,
            "page2": """
            3. Rejection Protocols
            If a batch has a defect rate higher than 3%, the entire lot must be quarantined.
            Fill out Form Q-99 and tag the crate with a red 'HOLD' label.
            
            4. Calibration
            All calipers and micrometers must be calibrated at the start of every shift using the standard gauge blocks.
            """
        },
        {
            "filename": "SOP_03_Shift_Handover.pdf",
            "title": "SOP: Shift Handover Protocols",
            "page1": """
            1. Objective
            To ensure a seamless transition between the Morning, Afternoon, and Night shifts.
            
            2. Checklist
            - Clean the workstation.
            - Complete the digital logbook entry.
            - Inform the incoming operator of any pending maintenance tasks or active alerts.
            """,
            "page2": """
            3. Accountabilities
            The outgoing operator remains responsible for the station until the incoming operator physically signs into the console.
            
            4. Tool Inventory
            Conduct a quick tool audit. Missing tools must be flagged immediately to prevent Foreign Object Debris (FOD) hazards.
            """
        }
    ],
    "HR_Manual": [
        {
            "filename": "HR_01_Code_of_Conduct.pdf",
            "title": "Employee Code of Conduct",
            "page1": """
            1. Professional Behavior
            Employees are expected to treat colleagues, clients, and partners with respect. 
            Discrimination or harassment based on race, gender, religion, or age will not be tolerated.
            
            2. Conflict of Interest
            Employees must not engage in outside business activities that compete with the Company.
            """,
            "page2": """
            3. Dress Code
            Business casual is the standard attire. For client-facing meetings, formal business attire is required.
            
            4. Confidentiality
            Disclosing company trade secrets or client data to unauthorized third parties is grounds for immediate termination.
            """
        },
        {
            "filename": "HR_02_Leave_Policy.pdf",
            "title": "Leave and Benefits Policy",
            "page1": """
            1. Annual Leave
            Full-time employees are entitled to 20 days of paid annual leave per calendar year.
            Leave requests must be submitted via the HR portal at least 2 weeks in advance.
            
            2. Sick Leave
            Employees receive 10 days of sick leave. A medical certificate is required for absences exceeding 3 consecutive days.
            """,
            "page2": """
            3. Maternity/Paternity Leave
            Maternity leave is granted for 26 weeks. Paternity leave is granted for 4 weeks.
            
            4. Carry Over
            Up to 5 days of unused annual leave can be carried over to the first quarter of the following year.
            """
        }
    ],
    "Technical_Specifications": [
        {
            "filename": "Tech_01_Server_Arch.pdf",
            "title": "System Architecture v2.0",
            "page1": """
            1. Overview
            The system utilizes a microservices architecture running on Kubernetes clusters.
            
            2. Components
            - Load Balancer: Nginx Ingress Controller.
            - Backend: Python FastAPI services.
            - Frontend: React.js SPA served via CDN.
            """,
            "page2": """
            3. Database Schema
            Primary storage is PostgreSQL 14. 
            - User Table: UUID (PK), Email (Unique), Hash.
            - Transaction Table: ID (PK), Amount, Timestamp, User_FK.
            
            4. Redis Caching
            Session data is stored in Redis with a TTL of 30 minutes to reduce database load.
            """
        },
        {
            "filename": "Tech_02_API_Docs.pdf",
            "title": "API Documentation: External Integration",
            "page1": """
            1. Authentication
            All API requests must include the 'Authorization: Bearer <token>' header.
            Tokens are obtained via the OAuth2 /token endpoint.
            
            2. Rate Limiting
            Clients are limited to 100 requests per minute per IP address.
            """,
            "page2": """
            3. Endpoints
            - GET /api/v1/status: Returns service health.
            - POST /api/v1/orders: Creates a new order. Payload must be JSON.
            
            4. Error Codes
            - 401: Unauthorized.
            - 429: Too Many Requests.
            - 500: Internal Server Error.
            """
        }
    ],
    "Finance_Policy": [
        {
            "filename": "Fin_01_Expense_Reimbursement.pdf",
            "title": "Global Expense Reimbursement Policy",
            "page1": """
            1. Reimbursable Expenses
            - Business travel flights (Economy class).
            - Client dinners (capped at $100 per head).
            - Office supplies purchased for remote work.
            
            2. Submission Process
            Expenses must be uploaded to Concur by the 25th of the month.
            """,
            "page2": """
            3. Non-Reimbursable Items
            - Personal entertainment (movies, alcohol exceeding limits).
            - Parking fines or traffic violations.
            
            4. Approval Workflow
            Expenses under $500 are auto-approved. Expenses over $500 require VP approval.
            """
        },
        {
            "filename": "Fin_02_Procurement.pdf",
            "title": "Procurement and Vendor Management",
            "page1": """
            1. Vendor Selection
            At least three quotes must be obtained for any purchase exceeding $5,000.
            
            2. Contract Signing
            Only Department Heads have the authority to sign vendor contracts.
            """,
            "page2": """
            3. Payment Terms
            Standard payment terms are Net-60 days from invoice receipt.
            
            4. Ethics
            Accepting gifts or kickbacks from vendors is strictly prohibited and constitutes gross misconduct.
            """
        }
    ],
    "Legal_Contracts": [
        {
            "filename": "Legal_01_NDA.pdf",
            "title": "Non-Disclosure Agreement (Template)",
            "page1": """
            1. Definition of Confidential Information
            Includes all technical data, trade secrets, financial plans, and customer lists.
            
            2. Obligations
            The receiving party agrees to hold information in strict confidence and not to disclose it to third parties.
            """,
            "page2": """
            3. Term
            This agreement is effective for 5 years from the date of signing.
            
            4. Remedies
            The company reserves the right to seek injunctive relief in the event of a breach.
            """
        },
        {
            "filename": "Legal_02_Terms_of_Service.pdf",
            "title": "Platform Terms of Service",
            "page1": """
            1. Acceptance of Terms
            By accessing our platform, you agree to be bound by these terms.
            
            2. User Accounts
            You are responsible for maintaining the security of your password. 
            We are not liable for any loss resulting from unauthorized access.
            """,
            "page2": """
            3. Termination
            We reserve the right to suspend accounts that violate our usage policy (e.g., spamming, scraping data).
            
            4. Liability Limitation
            To the maximum extent permitted by law, the company is not liable for indirect damages.
            """
        }
    ]
}

# ==========================================
# 2. PDF Generation Class
# ==========================================

class ReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Acme Corp Internal Documents - Confidential', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_content_page(self, title, body_text):
        self.add_page()
        
        # Chapter Title
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
        
        # Body Text
        self.set_font('Arial', '', 12)
        # Clean up indentation from the multi-line strings above
        clean_text = textwrap.dedent(body_text).strip()
        self.multi_cell(0, 8, clean_text)

# ==========================================
# 3. Main Execution
# ==========================================

def create_dataset():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        print(f"Created base folder: {BASE_DIR}")

    for category, docs in DATASET_CONFIG.items():
        # Create Category Subfolder
        cat_path = os.path.join(BASE_DIR, category)
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        
        print(f"\nProcessing Category: {category}")
        
        for doc in docs:
            pdf = ReportPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Page 1
            pdf.add_content_page(f"{doc['title']} - Part 1", doc['page1'])
            
            # Page 2 (Adding specific content to make it 2 pages minimum)
            pdf.add_content_page(f"{doc['title']} - Part 2", doc['page2'])
            
            # Save file
            file_path = os.path.join(cat_path, doc['filename'])
            pdf.output(file_path)
            print(f"  -> Generated: {doc['filename']}")

    print("\n---------------------------------------------------")
    print(f"Done! RAG Dataset created at: {os.path.abspath(BASE_DIR)}")
    print("---------------------------------------------------")

if __name__ == "__main__":
    create_dataset()