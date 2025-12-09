import os
import random
import datetime
from fpdf import FPDF

# Configuration
OUTPUT_FOLDER = "all_docs/HR_Manual"
FILENAME = "Global_Employee_Handbook_v2025.pdf"
FILE_PATH = os.path.join(OUTPUT_FOLDER, FILENAME)

# The "Needle" - Unique Identifier for Keyword Search
UNIQUE_CLAUSE_ID = "CLAUSE-882-OMEGA"
UNIQUE_TOPIC = "Grandfathered Lunar Travel Allowance"

# ---------------------------------------------------------
# TEXT GENERATION HELPERS
# ---------------------------------------------------------
def get_corporate_filler(topic):
    """Generates realistic-sounding corporate speak based on a topic."""
    starters = [
        "The company is committed to", "Employees are expected to", 
        "It is the responsibility of management to", "Adherence to this policy ensures",
        "We strive to maintain a standard of", "In accordance with global compliance,"
    ]
    
    actions = [
        "facilitate a productive work environment", "uphold the highest ethical standards",
        "optimize operational efficiency", "foster a culture of inclusivity",
        "mitigate potential risks", "streamline communication channels",
        "ensure mutual respect and cooperation", "maximize stakeholder value"
    ]
    
    closers = [
        "in all professional interactions.", "during standard business hours.",
        "as outlined in the quarterly review.", "subject to managerial discretion.",
        "aligned with our core values.", "to prevent any conflict of interest."
    ]
    
    # Generate a paragraph of text
    text = ""
    for _ in range(15): # 15 sentences per block
        s = random.choice(starters)
        a = random.choice(actions)
        c = random.choice(closers)
        text += f"{s} {topic} to {a} {c} "
    
    return text

# ---------------------------------------------------------
# PDF CLASS
# ---------------------------------------------------------
class HandbookPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 10)
        self.set_text_color(100, 100, 100) # Gray
        self.cell(0, 10, 'Acme Corp - Global Employee Handbook (CONFIDENTIAL)', 0, 1, 'R')
        self.line(10, 20, 200, 20)
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, f'Page {self.page_no()} | Ref: HR-DOC-2025-FULL', 0, 0, 'C')

def create_long_handbook():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"Generating 30-Page Handbook: {FILE_PATH}...")

    pdf = HandbookPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # ---------------------------------------------------------
    # METADATA & TITLE PAGE (Page 1)
    # ---------------------------------------------------------
    pdf.set_title("Global Employee Handbook 2025")
    pdf.set_author("Acme HR Department")
    pdf.set_subject("HR Policies and Procedures")
    pdf.set_keywords(f"HR, policy, conduct, benefits, {UNIQUE_CLAUSE_ID}") # Metadata Injection

    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.ln(50)
    pdf.cell(0, 20, "GLOBAL EMPLOYEE HANDBOOK", ln=True, align='C')
    pdf.set_font("Arial", '', 16)
    pdf.cell(0, 10, "Policies, Procedures, and Benefits", ln=True, align='C')
    pdf.ln(20)
    pdf.set_font("Courier", '', 12)
    pdf.cell(0, 10, f"Version: 2025.1.0", ln=True, align='C')
    pdf.cell(0, 10, f"Generated: {datetime.date.today()}", ln=True, align='C')
    pdf.cell(0, 10, f"ID: {UNIQUE_CLAUSE_ID} (Internal Ref)", ln=True, align='C')

    # ---------------------------------------------------------
    # GENERATE CONTENT PAGES (Pages 2 - 26)
    # ---------------------------------------------------------
    
    # Define topics to simulate a real handbook structure
    topics = [
        ("Code of Conduct", 4),        # 4 pages
        ("Workplace Safety", 3),       # 3 pages
        ("IT & Data Security", 3),     # 3 pages
        ("Anti-Harassment", 2),        # 2 pages
        ("Leave & Holidays", 4),       # 4 pages
        ("Standard Compensation", 5),  # 5 pages (The Haystack!)
        ("Termination Policy", 3)      # 3 pages
    ]

    current_page = 2
    
    for section_title, page_count in topics:
        # Section Title Page
        pdf.add_page()
        pdf.set_font("Arial", 'B', 18)
        pdf.set_text_color(0, 0, 128) # Navy Blue
        pdf.cell(0, 10, f"Section: {section_title}", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(0, 0, 0)

        # Fill the pages for this section
        for i in range(page_count):
            if i > 0: pdf.add_page() # New page for subsequent filler
            
            # Sub-header
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"{section_title} - Subsection {i+1}.0", ln=True)
            pdf.ln(2)
            
            # Body Text (Filler)
            pdf.set_font("Arial", '', 11)
            content = get_corporate_filler(section_title.lower())
            
            # Add some visual structure (bullet points)
            pdf.multi_cell(0, 8, content)
            pdf.ln(5)
            pdf.cell(10, 8, "-", align='R')
            pdf.cell(0, 8, "Compliance is mandatory.", ln=True)
            pdf.cell(10, 8, "-", align='R')
            pdf.cell(0, 8, "Exceptions require written approval.", ln=True)
            pdf.ln(10)
            
            # More filler to ensure page is full
            pdf.multi_cell(0, 8, get_corporate_filler(f"standard {section_title} protocols"))
            
            print(f"  -> Generated Page {pdf.page_no()}: {section_title}")

    # ---------------------------------------------------------
    # THE NEEDLE (Page 27-28 approx)
    # ---------------------------------------------------------
    # We deliberately place this deep in the "Standard Compensation" or "Termination" area
    # but give it a very specific ID.
    
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(200, 0, 0) # Dark Red
    pdf.cell(0, 10, "Appendix Z: Special Grandfathered Clauses", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 8, "The following clauses apply only to employees hired prior to 1999 who opted into the legacy retention scheme.")
    pdf.ln(10)
    
    # THE NEEDLE TEXT
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Policy ID: {UNIQUE_CLAUSE_ID}", ln=True)
    pdf.set_font("Arial", '', 11)
    
    needle_text = (
        f"Subject: {UNIQUE_TOPIC}. "
        "Unlike standard travel allowances discussed in Section 5, this specific provision grants eligible "
        "employees a reimbursement of 500,000 Credits for lunar transport expenses. "
        "To claim this, one must submit Form L-99 physically to the basement archives. "
        "This is strictly separate from the Annual Merit Bonus."
    )
    pdf.multi_cell(0, 8, needle_text)
    
    pdf.ln(10)
    pdf.multi_cell(0, 8, get_corporate_filler("legacy contract administration"))

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------
    pdf.output(FILE_PATH)
    print(f"\nSUCCESS: Created {FILENAME} ({pdf.page_no()} Pages).")
    print(f"The 'Needle' is located in Appendix Z.")
    print(f"Keyword to test: {UNIQUE_CLAUSE_ID}")

if __name__ == "__main__":
    create_long_handbook()