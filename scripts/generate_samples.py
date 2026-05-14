"""
Generate synthetic legal sample documents for testing and demo.
Creates realistic messy legal-style text files and PDFs.

Run: python scripts/generate_samples.py
"""
import sys
import os
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_inputs"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


SAMPLE_CONTRACT = """\
SERVICES AGREEMENT

This Services Agreement ("Agreement") is entered into as of January 15, 2024
by and between:

Westbrook Capital Partners LLC, a Delaware limited liability company
("Client"), and

Meridian Consulting Group Inc., a New York corporation ("Service Provider").

CASE NO.: CV-2024-00891-NYC

WHEREAS, Client desires to retain Service Provider to perform certain
consulting services as described herein; and

WHEREAS, Service Provider desires to provide such services subject to the
terms and conditions set forth in this Agreement;

NOW THEREFORE, in consideration of the mutual covenants and agreements
hereinafter set forth, the parties agree as follows:

1. SERVICES
Service Provider shall perform management consulting and strategic advisory
services (the "Services") as outlined in Exhibit A attached hereto and
incorporated herein by reference. Services shall commence on February 1, 2024
and continue through December 31, 2024 unless earlier terminated.

2. COMPENSATION
Client shall pay Service Provider a monthly retainer fee of $45,000.00
(forty-five thousand dollars), payable within 30 days of invoice. In addition,
Client agrees to reimburse Service Provider for all reasonable out-of-pocket
expenses not to exceed $5,000.00 per month without prior written approval.
Total contract value: $540,000.00.

3. CONFIDENTIALITY
Each party acknowledges that it may receive Confidential Information of the
other party. "Confidential Information" means any information disclosed that
is designated as confidential or that reasonably should be understood to be
confidential. This obligation of confidentiality shall survive termination of
this Agreement for a period of three (3) years.

4. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws
of the State of New York, without regard to its conflict of law provisions.
Any disputes arising hereunder shall be resolved by binding arbitration in
New York County, New York.

5. TERMINATION
Either party may terminate this Agreement upon thirty (30) days prior written
notice to the other party. Client may terminate immediately for cause upon
written notice if Service Provider materially breaches this Agreement and
fails to cure such breach within fifteen (15) days after receipt of notice.

6. LIMITATION OF LIABILITY
In no event shall either party be liable for any indirect, incidental, special,
or consequential damages. Service Provider's total liability under this
Agreement shall not exceed the total fees paid in the three (3) months
preceding the claim.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date
first written above.

WESTBROOK CAPITAL PARTNERS LLC          MERIDIAN CONSULTING GROUP INC.

By: _____________________________       By: _____________________________
Name: Jonathan R. Westbrook             Name: Sarah Chen
Title: Managing Partner                 Title: Chief Executive Officer
Date: January 15, 2024                  Date: January 15, 2024

Counsel for Client:                     Counsel for Service Provider:
Michael D. Harrison, Esq.               Patricia L. Nguyen, Esq.
Bar No. 4521890                         Bar No. 3819045
Harrison & Associates LLP               Nguyen Law Group PC
350 Park Avenue, New York, NY 10022     200 Vesey Street, New York, NY 10281
"""


SAMPLE_NOTICE = """\
                                                        VIA CERTIFIED MAIL
                                              Return Receipt Requested

Date: March 8, 2024

TO:   Redstone Development Corp.
      ATTN: Gregory T. Redstone, President
      450 Lexington Avenue, Suite 2100
      New York, NY 10017

FROM: Pearson Specter Litt LLP
      On behalf of: BlueSky Properties LLC

RE:   NOTICE OF DEFAULT AND DEMAND TO CURE
      Lease Agreement dated April 1, 2022
      Premises: 780 Fifth Avenue, Floors 12–14, New York, NY 10019
      Docket Reference: PSL-2024-0308

Dear Mr. Redstone:

This firm represents BlueSky Properties LLC ("Landlord") in connection with
the above-referenced Lease Agreement ("Lease"). You are hereby notified that
Redstone Development Corp. ("Tenant") is in material default under the Lease.

NATURE OF DEFAULT

1. Failure to Pay Rent: Tenant has failed to pay base rent for the months of
   January 2024 ($125,000.00), February 2024 ($125,000.00), and March 2024
   ($125,000.00), for a total of $375,000.00 in unpaid rent currently due
   and owing.

2. Failure to Maintain Insurance: Tenant has allowed its commercial general
   liability insurance policy to lapse as of February 15, 2024, in direct
   violation of Section 12.3 of the Lease.

3. Unauthorized Alterations: Tenant has made structural alterations to the
   12th floor premises without obtaining Landlord's prior written consent,
   in violation of Section 8.1 of the Lease.

DEMAND TO CURE

Pursuant to Section 19.2 of the Lease and New York Real Property Law § 235,
Tenant has FIFTEEN (15) DAYS from the date of this notice (i.e., by
March 23, 2024) to cure each of the above defaults by:

   (a) Paying all outstanding rent in the amount of $375,000.00;
   (b) Providing evidence of reinstatement of the required insurance; and
   (c) Removing all unauthorized alterations and restoring the premises.

FAILURE TO CURE

If Tenant fails to cure all defaults within the time specified, Landlord will
exercise all rights and remedies available under the Lease, at law, and in
equity, including but not limited to commencing eviction proceedings and
seeking all damages incurred by Landlord.

This notice is without prejudice to any other rights or remedies Landlord
may have, all of which are expressly reserved.

Sincerely,

Harvey R. Specter
Pearson Specter Litt LLP
1221 Avenue of the Americas
New York, NY 10020
Tel: (212) 555-0100
harvey.specter@psl-law.com
"""


SAMPLE_COMPLAINT = """\
UNITED STATES DISTRICT COURT
SOUTHERN DISTRICT OF NEW YORK
-------------------------------------------------- x

ARMADA HOLDINGS INC.,

                               Plaintiff,          Case No. 1:24-cv-02847-JGK

              -against-

PINNACLE ASSET MANAGEMENT LLC and
DEREK C. MORRISON,

                               Defendants.
-------------------------------------------------- x

                        COMPLAINT FOR DAMAGES

        Plaintiff Armada Holdings Inc. ("Armada" or "Plaintiff"), by and through
its undersigned counsel, Pearson Specter Litt LLP, hereby brings this action
against Defendants Pinnacle Asset Management LLC ("Pinnacle") and Derek C.
Morrison ("Morrison") (collectively, "Defendants"), and alleges as follows:

                              NATURE OF THE ACTION

        1.  This action arises from Defendants' fraudulent scheme to misappropriate
approximately $8,200,000.00 (eight million two hundred thousand dollars) in
investment capital entrusted by Plaintiff to Pinnacle pursuant to an Investment
Management Agreement dated June 1, 2022 (the "IMA").

        2.  Defendants misrepresented the nature and performance of investment
positions, fabricated account statements, and diverted client funds for personal
use. When confronted with evidence of the fraud in November 2023, Defendants
ceased communications and absconded with the remaining funds.

                                     PARTIES

        3.  Plaintiff Armada Holdings Inc. is a Delaware corporation with its
principal place of business at 100 Wall Street, New York, NY 10005.

        4.  Defendant Pinnacle Asset Management LLC is a New York limited liability
company registered to conduct business in New York, with offices at 601 Lexington
Avenue, New York, NY 10022.

        5.  Defendant Derek C. Morrison is an individual residing at 72 Park
Avenue, New York, NY 10016, and is the sole managing member of Pinnacle.

                             JURISDICTION AND VENUE

        6.  This Court has subject matter jurisdiction pursuant to 28 U.S.C.
§ 1332 based on diversity of citizenship. The amount in controversy exceeds
$75,000, exclusive of interest and costs.

                          FIRST CAUSE OF ACTION — FRAUD

        7.  Plaintiff incorporates by reference all preceding paragraphs.

        8.  Defendants knowingly made material misrepresentations regarding the
investment performance and the safeguarding of Plaintiff's funds, with intent
to induce Plaintiff's reliance.

        9.  Plaintiff reasonably relied on Defendants' misrepresentations and
sustained damages in the amount of not less than $8,200,000.00.

                    SECOND CAUSE OF ACTION — BREACH OF FIDUCIARY DUTY

        10. Pinnacle owed Plaintiff a fiduciary duty as its investment manager.

        11. Defendants breached that duty by commingling client funds, making
unauthorized trades, and failing to act in Plaintiff's best interest.

                                  PRAYER FOR RELIEF

        WHEREFORE, Plaintiff demands judgment against Defendants as follows:

        (a)  Compensatory damages in an amount to be determined at trial but no
             less than $8,200,000.00;

        (b)  Punitive damages;

        (c)  Pre- and post-judgment interest;

        (d)  Attorneys' fees and costs; and

        (e)  Such other and further relief as the Court deems just and proper.

Dated:  April 12, 2024
        New York, New York

                                          Respectfully submitted,

                                          PEARSON SPECTER LITT LLP

                                          By: /s/ Michael D. Harrison
                                              Michael D. Harrison, Esq.
                                              Bar No. 4521890
                                              1221 Avenue of the Americas
                                              New York, NY 10020
                                              Tel: (212) 555-0100

                                          Attorneys for Plaintiff Armada Holdings Inc.
"""


SAMPLE_AFFIDAVIT = """\
SUPREME COURT OF THE STATE OF NEW YORK
COUNTY OF NEW YORK
-------------------------------------------------------- x

In the Matter of the Application of
THORNFIELD ESTATE MANAGEMENT LLC,

                              Petitioner,               Index No.: 650892/2024

              -against-

CITY OF NEW YORK DEPARTMENT OF
BUILDINGS,

                              Respondent.
-------------------------------------------------------- x

                    AFFIDAVIT OF ROBERT J. CALHOUN

STATE OF NEW YORK  )
                   ) ss.:
COUNTY OF NEW YORK )

        ROBERT J. CALHOUN, being duly sworn, deposes and says:

        1.  I am a licensed professional engineer in the State of New York
(License No. PE-087234) and a principal at Calhoun Structural Engineering PLLC.
I make this affidavit based upon my personal knowledge and review of the
records and plans relating to the premises located at 455 West 23rd Street,
New York, New York (the "Building").

        2.  I was retained by Thornfield Estate Management LLC ("Petitioner")
on January 22, 2024, to evaluate the structural integrity of the Building
following the issuance of Emergency Repair Order No. 2024-EMG-004418 by the
New York City Department of Buildings ("DOB") on January 19, 2024.

        3.  On January 24, 2024, I conducted a thorough physical inspection of
all accessible areas of the Building, including the cellar, first through
sixth floors, roof, and all exterior facades.

        4.  Based upon my inspection and review of original construction
drawings dated 1962 and subsequent renovation drawings from 1998, I find
that the structural elements of the Building are sound and not in imminent
danger of collapse.

        5.  The condition cited in the DOB emergency order — specifically,
cracking in the northeast parapet wall — is cosmetic in nature and does not
affect the structural integrity of the Building. I have enclosed herewith
as Exhibit A a photographic record of my inspection findings.

        6.  I respectfully submit that the Emergency Repair Order should be
vacated and that Petitioner should be permitted to address the parapet
condition through normal repair procedures under a standard work permit.

        Sworn to before me this
        15th day of February, 2024

        ___________________________
        Notary Public

        ROBERT J. CALHOUN, P.E.
"""


def write_sample(filename: str, content: str) -> None:
    path = SAMPLE_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"  ✓ Written: {path}")


def generate_pdf_from_text(filename: str, content: str) -> None:
    """Generate a simple PDF using reportlab or fpdf2 if available, else skip."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch

        path = SAMPLE_DIR / filename
        doc = SimpleDocTemplate(str(path), pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        styles = getSampleStyleSheet()
        story = []
        for line in content.split('\n'):
            if line.strip():
                story.append(Paragraph(line.replace('&', '&amp;'), styles['Normal']))
                story.append(Spacer(1, 4))
            else:
                story.append(Spacer(1, 8))
        doc.build(story)
        print(f"  ✓ PDF written: {path}")
    except ImportError:
        print(f"  ℹ Skipping PDF generation (reportlab not installed). Use text files instead.")


def main():
    print(f"Generating sample documents in: {SAMPLE_DIR}")
    print()

    write_sample("01_services_agreement.txt", SAMPLE_CONTRACT)
    write_sample("02_notice_of_default.txt", SAMPLE_NOTICE)
    write_sample("03_complaint_fraud.txt", SAMPLE_COMPLAINT)
    write_sample("04_affidavit_calhoun.txt", SAMPLE_AFFIDAVIT)

    # Try to generate PDFs
    generate_pdf_from_text("01_services_agreement.pdf", SAMPLE_CONTRACT)
    generate_pdf_from_text("02_notice_of_default.pdf", SAMPLE_NOTICE)

    print()
    print(f"✅ Sample documents ready in: {SAMPLE_DIR}")
    print("   Run: python main.py --demo")


if __name__ == "__main__":
    main()
