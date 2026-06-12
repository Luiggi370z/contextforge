#!/usr/bin/env python3
"""Generate the overlapping reference-corpus PDFs used by the eval harness.

These are *real* PDFs with a text layer (PyMuPDF ``insert_text`` / tables /
footers), multi-page, structured with titles, sections, bullet lists and
tables — not one blob of text per page. ``PdfLoader`` extracts them per page
via ``get_text("text")``.

The whole point of the set is **vocabulary overlap**: the verbose PDFs
(handbook, benefits, security guidelines, expense + travel, onboarding) restate
the same topics the precise ``policy_*.md`` files own — "approval", "VPN",
"MFA", "encryption", "reimbursement", "leave", "limit", "incident" — so a
dense-only retriever is tempted by the wrong document and hybrid + rerank have
real lift to demonstrate. They are deliberate *distractors*, not the canonical
answers (those stay in the short ``.md`` policies).

Run from the repo root:

    uv run python scripts/make_corpus_pdfs.py
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

CORPUS_DIR = (
    Path(__file__).resolve().parent.parent / "backend" / "app" / "eval" / "corpus"
)

# Page geometry (A4 portrait) and typography.
PAGE_W, PAGE_H = 595.276, 841.89
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 64.0, 64.0, 68.0, 64.0

FONT, FONT_B = "helv", "hebo"
SIZE_TITLE, SIZE_H1, SIZE_H2, SIZE_BODY, SIZE_SMALL = 22.0, 14.0, 11.0, 10.5, 9.0

INK = (0.12, 0.12, 0.16)
GRAY = (0.42, 0.44, 0.5)
RULE = (0.7, 0.72, 0.78)
ACCENT = (0.13, 0.3, 0.55)
FILL = (0.93, 0.94, 0.97)


class Pdf:
    """Minimal cursor-based flow layout over PyMuPDF with auto pagination."""

    def __init__(self, title: str, subtitle: str) -> None:
        self.title_text = title
        self.doc = fitz.open()
        self.doc.set_metadata(
            {"title": title, "author": "Northwind Labs People & Security"}
        )
        self.page = self._add_page()
        self.y = MARGIN_T
        self._cover(title, subtitle)

    # --- page + footer plumbing -------------------------------------------------
    def _add_page(self) -> fitz.Page:
        return self.doc.new_page(width=PAGE_W, height=PAGE_H)

    def _new_page(self) -> None:
        self.page = self._add_page()
        self.y = MARGIN_T

    def _ensure(self, needed: float) -> None:
        if self.y + needed > PAGE_H - MARGIN_B:
            self._new_page()

    @property
    def _max_w(self) -> float:
        return PAGE_W - MARGIN_L - MARGIN_R

    # --- text primitives --------------------------------------------------------
    def _wrap(self, text: str, font: str, size: float, max_w: float) -> list[str]:
        lines: list[str] = []
        current = ""
        for word in text.split():
            trial = word if not current else f"{current} {word}"
            if fitz.get_text_length(trial, fontname=font, fontsize=size) <= max_w:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    def _line(self, text: str, *, font: str, size: float, color, x: float) -> None:
        leading = size * 1.42
        self._ensure(leading)
        self.page.insert_text(
            (x, self.y + size), text, fontname=font, fontsize=size, color=color
        )
        self.y += leading

    # --- public block API -------------------------------------------------------
    def _cover(self, title: str, subtitle: str) -> None:
        self.y = MARGIN_T + 18
        for ln in self._wrap(title, FONT_B, SIZE_TITLE, self._max_w):
            self._line(ln, font=FONT_B, size=SIZE_TITLE, color=ACCENT, x=MARGIN_L)
        self.y += 2
        self._line(subtitle, font=FONT, size=SIZE_SMALL, color=GRAY, x=MARGIN_L)
        self.y += 6
        self.page.draw_line(
            (MARGIN_L, self.y), (PAGE_W - MARGIN_R, self.y), color=RULE, width=1.1
        )
        self.y += 16

    def h1(self, text: str) -> None:
        self._ensure(SIZE_H1 * 1.42 + SIZE_BODY * 3)  # keep heading with some body
        self.y += 6
        self._line(text, font=FONT_B, size=SIZE_H1, color=INK, x=MARGIN_L)
        self.page.draw_line(
            (MARGIN_L, self.y), (PAGE_W - MARGIN_R, self.y), color=RULE, width=0.7
        )
        self.y += 8

    def h2(self, text: str) -> None:
        self._ensure(SIZE_H2 * 1.42 + SIZE_BODY * 2)
        self.y += 3
        self._line(text, font=FONT_B, size=SIZE_H2, color=ACCENT, x=MARGIN_L)
        self.y += 1

    def p(self, text: str) -> None:
        for ln in self._wrap(text, FONT, SIZE_BODY, self._max_w):
            self._line(ln, font=FONT, size=SIZE_BODY, color=INK, x=MARGIN_L)
        self.y += 6

    def bullets(self, items: list[str]) -> None:
        indent = 16.0
        for item in items:
            wrapped = self._wrap(item, FONT, SIZE_BODY, self._max_w - indent)
            self._ensure(SIZE_BODY * 1.42)
            self.page.insert_text(
                (MARGIN_L + 2, self.y + SIZE_BODY),
                "•",
                fontname=FONT,
                fontsize=SIZE_BODY,
                color=ACCENT,
            )
            for ln in wrapped:
                self._line(
                    ln, font=FONT, size=SIZE_BODY, color=INK, x=MARGIN_L + indent
                )
        self.y += 6

    def table(
        self, headers: list[str], rows: list[list[str]], widths: list[float]
    ) -> None:
        total = sum(widths)
        scale = self._max_w / total
        cols = [w * scale for w in widths]
        row_h = SIZE_BODY * 1.7

        def draw_row(cells: list[str], *, bold: bool, fill: bool) -> None:
            self._ensure(row_h)
            top = self.y
            if fill:
                self.page.draw_rect(
                    fitz.Rect(MARGIN_L, top, MARGIN_L + self._max_w, top + row_h),
                    color=FILL,
                    fill=FILL,
                )
            x = MARGIN_L
            for cell, w in zip(cells, cols, strict=True):
                clipped = self._wrap(cell, FONT_B if bold else FONT, SIZE_SMALL, w - 8)[
                    0
                ]
                self.page.insert_text(
                    (x + 4, top + row_h - 6),
                    clipped,
                    fontname=FONT_B if bold else FONT,
                    fontsize=SIZE_SMALL,
                    color=INK,
                )
                x += w
            self.page.draw_line(
                (MARGIN_L, top + row_h),
                (MARGIN_L + self._max_w, top + row_h),
                color=RULE,
                width=0.5,
            )
            self.y += row_h

        draw_row(headers, bold=True, fill=True)
        for row in rows:
            draw_row(row, bold=False, fill=False)
        self.y += 8

    # --- finalize ---------------------------------------------------------------
    def save(self, path: Path) -> int:
        total = self.doc.page_count
        for index in range(total):
            page = self.doc.load_page(index)
            footer = f"{self.title_text}    ·    Page {index + 1} of {total}"
            tw = fitz.get_text_length(footer, fontname=FONT, fontsize=SIZE_SMALL)
            page.draw_line(
                (MARGIN_L, PAGE_H - MARGIN_B + 14),
                (PAGE_W - MARGIN_R, PAGE_H - MARGIN_B + 14),
                color=RULE,
                width=0.5,
            )
            page.insert_text(
                ((PAGE_W - tw) / 2, PAGE_H - MARGIN_B + 28),
                footer,
                fontname=FONT,
                fontsize=SIZE_SMALL,
                color=GRAY,
            )
        self.doc.save(str(path), deflate=True, garbage=4)
        self.doc.close()
        return total


# ---------------------------------------------------------------------------
# Document content. Deliberately overlapping vocabulary with the canonical
# policy_*.md files and with each other.
# ---------------------------------------------------------------------------


def build_employee_handbook() -> Pdf:
    d = Pdf("Northwind Labs Employee Handbook", "Revision 7 · Effective 1 January 2099")
    d.p(
        "This handbook is a high-level orientation to how we work at Northwind Labs. It "
        "summarizes our values, the way we manage time off, our working arrangements, how "
        "expenses are reimbursed, and the security expectations that apply to everyone. Where a "
        "topic has a dedicated policy, this handbook only paraphrases it; the dedicated policy "
        "document is always authoritative if the two ever disagree."
    )
    d.h1("1. Working at Northwind")
    d.p(
        "We are a remote-friendly company built around small, autonomous teams. Most decisions "
        "are made by the people closest to the work, with their manager's support. We value "
        "written communication, reproducible work, and treating one another with respect."
    )
    d.bullets(
        [
            "Default to transparency: write things down so colleagues in other time zones can "
            "follow along without a meeting.",
            "Ask for forgiveness, not permission, on reversible decisions; ask for approval "
            "first on anything costly or hard to undo.",
            "Protect focus time. Core collaboration hours exist so the rest of the day is yours.",
        ]
    )
    d.h1("2. Time Off and Leave")
    d.p(
        "Northwind offers paid time off, company holidays, sick leave, and several extended "
        "leave programs. Paid time off accrues over the year and a limited amount may roll over; "
        "the exact accrual rate, the rollover cap, and the notice you must give are defined in "
        "the Paid Time Off Policy. Extended programs such as parental, bereavement, and "
        "sabbatical leave are described in the Benefits Guide."
    )
    d.h2("Requesting time off")
    d.p(
        "Submit time-off requests through the people portal and give your team reasonable "
        "notice so coverage can be arranged. Emergencies are understood and the notice "
        "expectation is waived in those cases. Your manager approves the request."
    )
    d.h1("3. Working Arrangements")
    d.p(
        "Employees may work remotely for part of the week, subject to manager approval and the "
        "needs of the team. We keep a band of core hours so synchronous collaboration is "
        "predictable across time zones; outside those hours you manage your own schedule. When "
        "working remotely you must connect to internal systems through the company-approved VPN. "
        "The specifics — how many days, the exact core-hours window, and the approval path — "
        "live in the Remote Work Policy."
    )
    d.h1("4. Expenses and Reimbursement")
    d.p(
        "Reasonable business expenses are reimbursed. Travel, lodging, meals, and home-office "
        "equipment follow per-category limits and an approval threshold that scales with the "
        "amount. Keep itemized receipts and submit them promptly. The Travel and Expense Policy "
        "defines the limits, the per-diem rates, and who must approve a given amount."
    )
    d.h1("5. Compensation and Benefits")
    d.p(
        "Compensation is reviewed annually. Benefits include medical, dental, and vision "
        "coverage, a retirement plan with an employer match, an annual learning budget, and "
        "wellness support. Enrollment windows and the precise amounts are in the Benefits Guide."
    )
    d.h1("6. Information Security")
    d.p(
        "Security is everyone's responsibility. Company laptops must use full-disk encryption "
        "and lock automatically. Access to sensitive and production systems requires "
        "multi-factor authentication and is granted on a least-privilege basis. If you suspect a "
        "security incident, report it immediately through the security hotline or ticket queue. "
        "The Security Policy and the Information Security Guidelines give the full detail, "
        "including the reporting timeline and the approval required for production access."
    )
    d.h1("7. Communication and Meetings")
    d.p(
        "We bias toward asynchronous, written communication. Decisions, proposals, and status "
        "updates are written down and shared in the open so colleagues across time zones can "
        "catch up without a meeting. Meetings are for discussion that genuinely needs to be live; "
        "they have an agenda and a note-taker, and the notes are posted afterward."
    )
    d.bullets(
        [
            "Keep recurring meetings inside core collaboration hours so no one is forced online "
            "outside their working day.",
            "Use threads, not direct messages, for anything a teammate might later need to find.",
            "Record important live sessions so people who could not attend are not penalized.",
        ]
    )
    d.h1("8. Performance and Growth")
    d.p(
        "Performance is reviewed twice a year through a lightweight process focused on impact, "
        "collaboration, and growth rather than on hours logged. Your manager gives continuous "
        "feedback; the formal review simply summarizes it. Promotions follow from sustained "
        "impact at the next level and are calibrated across teams to keep the bar consistent. "
        "Your annual learning budget exists to support this growth — see the Benefits Guide."
    )
    d.h1("9. Conduct and Respect")
    d.p(
        "Everyone is entitled to a workplace free of harassment and discrimination. We expect "
        "professional, inclusive behavior in every channel — chat, video, documents, and "
        "in-person events alike. Concerns can be raised with your manager, the People team, or "
        "through the confidential reporting line, and they will be handled discreetly and "
        "without retaliation."
    )
    d.h2("Acknowledgement")
    d.p(
        "By continuing employment you acknowledge that you have read this handbook and agree to "
        "follow the policies it references. Questions go to your manager or the People team."
    )
    return d


def build_benefits_handbook() -> Pdf:
    d = Pdf(
        "2099 Benefits Guide", "Northwind Labs · People Operations · Plan year 2099"
    )
    d.p(
        "This guide explains the benefits available to full-time employees: health insurance, "
        "retirement, paid leave programs, and learning and wellness support. It complements the "
        "Paid Time Off Policy — that policy governs your day-to-day PTO accrual and rollover, "
        "while this guide covers the longer, program-based forms of leave."
    )
    d.h1("Health and Insurance")
    d.p(
        "Northwind offers medical, dental, and vision plans. The company pays the majority of "
        "the premium; your share depends on the tier you select and whether you cover "
        "dependents. Coverage begins on the first of the month after your start date."
    )
    d.table(
        ["Plan", "Employee monthly", "Family monthly", "Deductible"],
        [
            ["Core PPO", "$48", "$190", "$1,500"],
            ["Plus PPO", "$92", "$310", "$750"],
            ["High-deductible + HSA", "$0", "$120", "$3,000"],
        ],
        [2.2, 1.6, 1.6, 1.4],
    )
    d.h1("Retirement")
    d.p(
        "Employees may contribute to the 401(k) plan from their first paycheck. Northwind matches "
        "100% of the first 4% of eligible pay you contribute; the match vests immediately. You "
        "can change your contribution rate at any time through the benefits portal."
    )
    d.h1("Paid Leave Programs")
    d.p(
        "Beyond standard paid time off, Northwind provides several dedicated leave programs. "
        "These are separate from your PTO balance and do not draw it down."
    )
    d.bullets(
        [
            "Parental leave: 16 weeks of fully paid leave for the birth or adoption of a child, "
            "available to all new parents within the first year.",
            "Bereavement leave: up to 5 paid days for the loss of an immediate family member.",
            "Jury duty: paid in full for the duration of service.",
            "Sabbatical: after five continuous years, a four-week paid sabbatical, scheduled with "
            "manager approval so the team has coverage.",
        ]
    )
    d.h1("Learning and Wellness")
    d.p(
        "Every employee receives an annual learning budget of $1,500 for courses, books, "
        "conferences, and certifications relevant to their role. Unused learning budget does not "
        "carry into the next year. A separate wellness stipend of $50 per month may be used for "
        "fitness, mental-health apps, or ergonomic accessories."
    )
    d.h2("Using the learning budget")
    d.p(
        "Purchases under $200 need no pre-approval; above that, get your manager's approval "
        "first. Submit the receipt through the same expense workflow used for business expenses, "
        "tagged as Learning, and you will be reimbursed on the next cycle."
    )
    d.h1("Reimbursements and Commuter Benefits")
    d.p(
        "Tuition reimbursement of up to $5,000 per year is available for approved degree programs "
        "with a passing grade. Commuter benefits let you set aside pre-tax dollars for transit "
        "and parking. Both are requested through the benefits portal; reimbursement timelines "
        "match the standard expense policy."
    )
    d.h1("Time Off and Holidays")
    d.p(
        "This guide does not set your day-to-day paid-time-off accrual or its rollover cap — "
        "those belong to the Paid Time Off Policy — but it is worth knowing how the pieces fit "
        "together. PTO covers vacation and personal days and accrues across the year. Sick leave "
        "is tracked separately and is not capped the way PTO rollover is. Company holidays are "
        "in addition to PTO, and the leave programs above sit on top of all of it."
    )
    d.table(
        ["Program", "Length", "Pay", "Approval"],
        [
            ["Parental leave", "16 weeks", "100% paid", "People Ops"],
            ["Bereavement", "Up to 5 days", "100% paid", "Manager"],
            ["Sabbatical", "4 weeks / 5 yrs", "100% paid", "Manager"],
            ["Jury duty", "As required", "100% paid", "Notify manager"],
        ],
        [1.8, 1.6, 1.4, 1.6],
    )
    d.h1("Wellness and Support")
    d.p(
        "Beyond the monthly wellness stipend, Northwind provides a confidential employee "
        "assistance program offering counseling and financial and legal guidance at no cost, "
        "and a subsidized membership to a telehealth provider. These services are independent of "
        "your medical plan and do not affect your deductible."
    )
    d.h1("Enrollment and Contacts")
    d.p(
        "New hires enroll within 30 days of their start date. Outside of that window, changes "
        "require a qualifying life event such as marriage, a new dependent, or a change in a "
        "spouse's coverage. Open enrollment for the following plan year runs each November. "
        "Questions about any benefit go to People Operations through the benefits portal or the "
        "#benefits channel; for a claim dispute, contact the carrier directly using the number "
        "on your insurance card and copy People Operations so we can help escalate."
    )
    return d


def build_security_guidelines() -> Pdf:
    d = Pdf(
        "Information Security Guidelines",
        "Northwind Labs · Security Team · Classification: Internal",
    )
    d.p(
        "These guidelines expand on the Security Policy with practical detail for every "
        "employee. Where a number here differs from the Security Policy, the Security Policy "
        "governs — these guidelines exist to explain how to comply, not to set the rule."
    )
    d.h1("1. Device Security")
    d.p(
        "All company laptops and workstations must run full-disk encryption, which is enforced "
        "by device management at setup. Screens must lock automatically after a short idle "
        "period, and operating systems and browsers must stay on supported, patched versions. "
        "Do not store the only copy of sensitive data on a local disk."
    )
    d.bullets(
        [
            "Full-disk encryption is mandatory and verified centrally.",
            "Automatic screen lock and a strong device passcode are required.",
            "Install security updates within seven days of release.",
        ]
    )
    d.h1("2. Authentication and Access")
    d.p(
        "Sign in to company systems through single sign-on. Multi-factor authentication is "
        "required for SSO and for any access to sensitive or production systems. Use the "
        "company password manager for service credentials; never reuse a personal password for "
        "work. Access follows least privilege: you receive only the access your role needs."
    )
    d.h2("Production access")
    d.p(
        "Access to production systems is privileged. It requires multi-factor authentication and "
        "explicit approval from the security team before it is granted, and it is reviewed "
        "periodically. Standing production access is avoided in favor of time-bound, "
        "just-in-time grants."
    )
    d.h1("3. Network and Remote Access")
    d.p(
        "When working away from the office, connect to internal systems through the "
        "company-approved VPN. Treat public and untrusted Wi-Fi as hostile: the VPN must be on "
        "before you reach any internal service. Keep your home router firmware current and its "
        "admin password changed from the default. The VPN requirement here is the same one "
        "referenced by the Remote Work Policy."
    )
    d.h1("4. Incident Response")
    d.p(
        "If you see anything suspicious — a phishing attempt, a lost device, unexpected access, "
        "or a possible data exposure — report it immediately. Reporting early is always the "
        "right call; the security team would far rather triage a false alarm than learn of a "
        "real incident late."
    )
    d.bullets(
        [
            "Report through the security hotline or the security ticket queue.",
            "Do not attempt to investigate or remediate on your own; preserve evidence.",
            "The on-call security engineer acknowledges and begins triage as soon as a report "
            "arrives.",
        ]
    )
    d.h1("5. Data Handling")
    d.p(
        "Classify data as Public, Internal, Confidential, or Restricted and handle it according "
        "to its class. Share Confidential and Restricted data only with those who need it, and "
        "only over approved, encrypted channels. Delete data you no longer need in line with the "
        "retention schedule."
    )
    d.table(
        ["Class", "Examples", "Sharing", "At rest"],
        [
            ["Public", "Marketing, docs site", "Anyone", "No restriction"],
            ["Internal", "Org charts, plans", "All employees", "Encrypted"],
            [
                "Confidential",
                "Customer data, code",
                "Need-to-know",
                "Encrypted + access log",
            ],
            ["Restricted", "Secrets, keys, PII", "Named approvers", "Vaulted + MFA"],
        ],
        [1.3, 2.0, 1.6, 1.8],
    )
    d.h1("6. Email, Phishing, and Social Engineering")
    d.p(
        "Most real incidents start with a convincing message. Treat unexpected requests for "
        "credentials, payments, gift cards, or urgent secrecy as suspicious, even when they "
        "appear to come from a leader. Verify out of band through a known channel before acting. "
        "Multi-factor authentication blocks most credential theft, but a prompt you did not "
        "initiate is itself a warning sign — deny it and report it."
    )
    d.bullets(
        [
            "Never approve an MFA prompt you did not start; report unsolicited prompts.",
            "Hover before you click; check the real sender address, not the display name.",
            "Finance verifies any payment or banking-detail change by a second channel.",
        ]
    )
    d.h1("7. Vendors and Third-Party Access")
    d.p(
        "Vendors that touch company or customer data go through a security review before access "
        "is granted, sign a data-protection agreement, and receive the minimum access needed for "
        "the shortest necessary time. Their access is logged and reviewed on the same schedule "
        "as employee production access, and it is revoked promptly when the engagement ends."
    )
    return d


def build_expense_policy() -> Pdf:
    d = Pdf(
        "Travel and Expense Policy", "Northwind Labs · Finance · Effective 1 March 2099"
    )
    d.p(
        "This policy defines which business expenses Northwind reimburses, the limits that "
        "apply, and who must approve them. It applies to all employees and contractors who incur "
        "expenses on the company's behalf, including remote employees claiming a home-office "
        "stipend."
    )
    d.h1("1. General Principles")
    d.bullets(
        [
            "Spend the company's money as if it were your own: choose the reasonable option.",
            "Keep an itemized receipt for every expense; a card statement line is not a receipt.",
            "Submit expenses within 30 days of incurring them so the books close cleanly.",
            "Personal, entertainment-only, and out-of-policy expenses are not reimbursed.",
        ]
    )
    d.h1("2. Approval Authority")
    d.p(
        "Approval scales with the amount. Smaller expenses are auto-approved once a receipt is "
        "attached; larger amounts route to your manager and then to Finance."
    )
    d.table(
        ["Amount", "Approver", "Pre-approval"],
        [
            ["Up to $100", "Auto-approved with receipt", "Not required"],
            ["$100 - $1,000", "Direct manager", "Required before purchase"],
            ["$1,000 - $5,000", "Manager + Finance", "Required before purchase"],
            [
                "Over $5,000",
                "Department head + Finance",
                "Required, with justification",
            ],
        ],
        [1.4, 2.4, 1.8],
    )
    d.h1("3. Travel")
    d.p(
        "Book travel through the company tool when possible. Economy airfare is the standard; "
        "premium cabins need department-head approval. Lodging should be a standard business "
        "room near the work location. Meals on travel days are covered by a per-diem rather than "
        "itemized receipts."
    )
    d.table(
        ["Region", "Per-diem (meals)", "Lodging cap / night"],
        [
            ["Domestic, standard city", "$65", "$220"],
            ["Domestic, high-cost city", "$80", "$320"],
            ["International", "$90", "$300"],
        ],
        [2.0, 1.6, 1.8],
    )
    d.h1("4. Home Office and Equipment")
    d.p(
        "Remote employees may claim a one-time home-office setup stipend of up to $750 for a "
        "desk, chair, monitor, and peripherals, with manager approval. Ongoing internet costs "
        "are not separately reimbursed. Company laptops are provided by IT and are not part of "
        "the stipend. Equipment over the stipend follows the approval table above."
    )
    d.h1("5. Meals and Entertainment")
    d.p(
        "On non-travel days, team meals and client meals are reimbursable within reason and with "
        "a clear business purpose recorded on the receipt. Alcohol is reimbursable only as a "
        "modest part of a client meal, never as a standalone expense. Recurring individual meals "
        "and groceries are personal and are not reimbursed."
    )
    d.table(
        ["Occasion", "Limit per person", "Approval"],
        [
            ["Team meal (in office)", "$30", "Auto with receipt"],
            ["Client meal", "$80", "Manager"],
            ["Offsite / team event", "$120", "Manager + Finance"],
        ],
        [2.0, 1.6, 1.8],
    )
    d.h1("6. Corporate Cards and Personal Funds")
    d.p(
        "Employees who incur frequent expenses may be issued a corporate card; charges still "
        "require receipts and follow the same approval table. If you pay out of pocket, you are "
        "reimbursed on the next payroll run after approval. Cash advances are not provided; use "
        "the corporate card or seek pre-approval for large planned costs."
    )
    d.h1("7. Non-Reimbursable Expenses")
    d.bullets(
        [
            "Traffic fines, parking tickets, and late fees of any kind.",
            "Personal entertainment, in-room movies, and minibar charges.",
            "Travel upgrades or premium cabins without department-head approval.",
            "Home internet and personal phone plans (covered by the home-office stipend rules).",
        ]
    )
    d.h1("8. Submitting Expenses")
    d.p(
        "Enter expenses in the expense tool, attach itemized receipts, and tag the category. "
        "Approved reimbursements are paid with the next payroll run, typically within ten "
        "business days of approval. A rejected expense comes back with a reason; correct it and "
        "resubmit. Questions about a category or a limit go to Finance through the #finance "
        "channel before you spend, not after."
    )
    return d


def build_onboarding_guide() -> Pdf:
    d = Pdf(
        "New Hire Onboarding Guide",
        "Northwind Labs · People Operations · First 30 days",
    )
    d.p(
        "Welcome to Northwind Labs. This guide walks you through your first month. It points to "
        "the policies that own the details — time off, remote work, security, expenses, and "
        "benefits — so treat it as a checklist and a map, not the final word on any single rule."
    )
    d.h1("Day One")
    d.bullets(
        [
            "Collect your company laptop from IT; confirm full-disk encryption and automatic "
            "screen lock are enabled before you sign in to anything.",
            "Set up single sign-on and enroll in multi-factor authentication — you will need MFA "
            "for nearly every system.",
            "Install and test the company-approved VPN; you must be on it to reach internal "
            "systems when working remotely.",
            "Meet your manager and review your team's core collaboration hours.",
        ]
    )
    d.h1("Week One")
    d.bullets(
        [
            "Enroll in benefits within your 30-day window: medical, dental, vision, and the "
            "401(k) with its employer match.",
            "Read the Paid Time Off Policy so you know how PTO accrues, how much can roll over, "
            "and the notice your team expects.",
            "Read the Remote Work Policy for how many remote days are allowed and the approval "
            "path with your manager.",
            "Skim the Travel and Expense Policy before your first purchase so you submit receipts "
            "correctly and route approvals.",
        ]
    )
    d.h1("First 30 Days")
    d.p(
        "By the end of your first month you should have completed security awareness training, "
        "submitted any home-office stipend request with manager approval, set your 401(k) "
        "contribution, and had a first check-in with your manager. If anything about a policy is "
        "unclear, ask in the relevant channel — #people, #security, or #finance — rather than "
        "guessing."
    )
    d.h1("Your Onboarding Buddy and Manager")
    d.p(
        "Every new hire is paired with an onboarding buddy — an experienced teammate, separate "
        "from your manager, who answers the small questions that are awkward to escalate. Your "
        "manager owns your ramp plan and your first goals; your buddy helps you navigate the "
        "tools, the channels, and the unwritten norms. Expect a short daily check-in with your "
        "buddy in week one and a weekly one-on-one with your manager from the start."
    )
    d.h1("Tools You Will Use")
    d.table(
        ["Tool", "Purpose", "Set up by"],
        [
            ["SSO + MFA", "Sign in to everything", "Day one, with IT"],
            ["VPN client", "Reach internal systems", "Day one, with IT"],
            ["People portal", "Time off, benefits", "Week one"],
            ["Expense tool", "Receipts and reimbursement", "Before first purchase"],
            ["Password manager", "Service credentials", "Day one"],
        ],
        [1.4, 2.0, 1.6],
    )
    d.h1("Common First-Week Questions")
    d.p(
        "How much time off do I get, and can it roll over? How many days a week can I work "
        "remotely? Who approves my expenses? What do I do if I think I clicked a phishing link? "
        "Each of these has a precise answer in a dedicated policy — this guide deliberately does "
        "not restate the numbers, because policies change and the guide should not drift out of "
        "sync. Always confirm the figure in the owning policy below."
    )
    d.h1("Where the Rules Actually Live")
    d.bullets(
        [
            "Paid time off, rollover, and notice: Paid Time Off Policy.",
            "Remote days, core hours, VPN: Remote Work Policy.",
            "Encryption, MFA, production access, incident reporting: Security Policy and the "
            "Information Security Guidelines.",
            "Limits, per-diems, and approvals: Travel and Expense Policy.",
            "Insurance, retirement, leave programs, and the learning budget: Benefits Guide.",
        ]
    )
    d.h1("A Note on Asking Questions")
    d.p(
        "No question is too small in your first month. The fastest way to ramp is to ask early "
        "and in the open, so the answer helps the next new hire too. If you are unsure which "
        "channel to use, ask your buddy — routing you to the right place is exactly what they are "
        "there for."
    )
    return d


def main() -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "employee_handbook.pdf": build_employee_handbook,
        "benefits_handbook.pdf": build_benefits_handbook,
        "security_guidelines.pdf": build_security_guidelines,
        "expense_policy.pdf": build_expense_policy,
        "onboarding_guide.pdf": build_onboarding_guide,
    }
    for filename, builder in builders.items():
        pages = builder().save(CORPUS_DIR / filename)
        print(f"wrote {filename:28s} {pages} page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
