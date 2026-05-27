# cover_generator.py — DIU Cover Page Generator (PDF, Lab Report + Assignment)

import os
import zipfile
import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

NAVY  = HexColor('#003366')
BLUE  = HexColor('#0054A6')
GREEN = HexColor('#008000')
GRAY  = HexColor('#666666')
BLACK = HexColor('#000000')


def _draw_border(c, doc):
    c.saveState()
    c.setStrokeColor(BLACK)
    c.setLineWidth(1.5)
    m = 18
    w, h = A4
    c.rect(m, m, w - 2*m, h - 2*m)
    c.restoreState()


def _s(name, font='Times-Roman', size=12, leading=None, align=TA_LEFT,
       color=BLACK, space_before=0, space_after=4):
    return ParagraphStyle(
        name, fontName=font, fontSize=size,
        leading=leading or size*1.35,
        alignment=align, textColor=color,
        spaceBefore=space_before, spaceAfter=space_after
    )


def _row(label, value, size=12):
    return Paragraph(
        f'<b>{label}:</b> {value}',
        _s(f'r_{label}', size=size, space_before=1, space_after=4)
    )


def _heading(text, color=BLUE, size=13):
    return Paragraph(
        f'<u><b>{text}</b></u>',
        _s(f'h_{text}', font='Helvetica-Bold', size=size,
           align=TA_CENTER, color=color, space_before=6, space_after=8)
    )


def create_single_cover(info: dict, filepath: str):
    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        topMargin=0.55*inch, bottomMargin=0.55*inch,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
    )
    story = []
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diu_logo.png')
    doc_type = info.get('doc_type', 'Lab Report')
    is_assignment = 'assignment' in doc_type.lower()

    # ── Header: Logo + University name ───────────────────────────────────────
    logo_cell = ''
    if os.path.exists(logo_path):
        logo_cell = RLImage(logo_path, width=1.1*inch, height=1.1*inch)

    uni1 = Paragraph(
        '<font name="Helvetica-Bold" size="22" color="#003366">Daffodil </font>'
        '<font name="Helvetica-Bold" size="20" color="#003366">International</font>',
        _s('u1', align=TA_LEFT, space_before=8, space_after=0)
    )
    uni2 = Paragraph(
        '<font name="Helvetica-Bold" size="26" color="#008000">University</font>',
        _s('u2', align=TA_LEFT, space_before=0, space_after=0)
    )
    header = Table([[logo_cell, [uni1, uni2]]], colWidths=[1.3*inch, 4.5*inch])
    header.setStyle(TableStyle([
        ('VALIGN',        (0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',   (0,0),(-1,-1),0),
        ('RIGHTPADDING',  (0,0),(-1,-1),6),
        ('TOPPADDING',    (0,0),(-1,-1),0),
        ('BOTTOMPADDING', (0,0),(-1,-1),0),
    ]))
    story.append(header)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width='100%', thickness=1.2, color=BLACK))
    story.append(Spacer(1, 10))

    # ── Document type ─────────────────────────────────────────────────────────
    story.append(_heading(f'{doc_type} Submission', size=14))

    # ── Course / topic info ───────────────────────────────────────────────────
    story.append(_row('Course Code', info.get('course_code', '')))
    story.append(_row('Course Name', info.get('course_title', '').upper()))

    if is_assignment:
        topic = info.get('experiment_name', '').strip()
        story.append(_row('Topic Name', topic.upper() if topic else ''))
    else:
        exp_no = info.get('experiment_no', '')
        if exp_no:
            story.append(_row('Experiment No', exp_no))
        exp_name = info.get('experiment_name', '').strip()
        story.append(_row('Experiment Name', exp_name.upper() if exp_name else ''))

    story.append(Spacer(1, 14))

    # ── Submitted To ──────────────────────────────────────────────────────────
    teacher = info.get('teacher_name', '').strip()
    if teacher:
        story.append(_heading('Submitted To'))
        story.append(_row('Name', teacher.upper()))
        story.append(_row('Designation', info.get('teacher_designation', 'Lecturer')))
        story.append(_row('Department', info.get('teacher_dept', info.get('department', 'CSE'))))
        story.append(Paragraph('<b>Daffodil International University</b>',
                               _s('diu1', size=12, space_before=1, space_after=4)))
        story.append(Spacer(1, 14))

    # ── Submitted By ──────────────────────────────────────────────────────────
    story.append(_heading('Submitted By'))
    story.append(_row('Name', info.get('student_name', '').upper()))
    story.append(_row('ID', info.get('student_id', '')))
    story.append(_row('Section', info.get('section', '')))
    if info.get('semester'):
        story.append(_row('Semester', info.get('semester', '')))
    story.append(_row('Department', info.get('department', 'CSE')))
    story.append(Paragraph('<b>Daffodil International University</b>',
                           _s('diu2', size=12, space_before=1, space_after=4)))
    story.append(Spacer(1, 18))

    # ── Submission Date box ───────────────────────────────────────────────────
    date_str = info.get('date', datetime.now().strftime('%d/%m/%Y'))
    dp = Paragraph(
        f'<font color="#0054A6"><b>Submission Date: {date_str}</b></font>',
        _s('dp', font='Helvetica-Bold', size=12, align=TA_CENTER)
    )
    date_tbl = Table([[dp]], colWidths=[3.2*inch])
    date_tbl.setStyle(TableStyle([
        ('BOX',           (0,0),(-1,-1), 1.5, HexColor('#1F6BAE')),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING',   (0,0),(-1,-1), 10),
        ('RIGHTPADDING',  (0,0),(-1,-1), 10),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
    ]))
    wrapper = Table([[date_tbl]], colWidths=[doc.width])
    wrapper.setStyle(TableStyle([
        ('ALIGN',         (0,0),(-1,-1),'CENTER'),
        ('LEFTPADDING',   (0,0),(-1,-1),0),
        ('RIGHTPADDING',  (0,0),(-1,-1),0),
    ]))
    story.append(wrapper)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY))
    story.append(Paragraph('www.diucoverpage.com',
                           _s('foot', size=8, align=TA_CENTER, color=GRAY, space_before=3)))

    doc.build(story, onFirstPage=_draw_border, onLaterPages=_draw_border)
    return filepath


def batch_create_covers(student_info: dict, experiments: list) -> str:
    tmp_dir = tempfile.mkdtemp(prefix='aria_covers_')
    generated = []
    try:
        for i, exp in enumerate(experiments, 1):
            info = dict(student_info)
            if isinstance(exp, dict):
                info['experiment_name'] = exp.get('name', f'Experiment {i}')
                info['experiment_no']   = exp.get('no', str(i))
            else:
                info['experiment_name'] = str(exp)
                info['experiment_no']   = str(i)

            safe = ''.join(c if c.isalnum() or c in ' _-' else '_'
                           for c in info['experiment_name']).strip()[:50]
            fp = os.path.join(tmp_dir, f'{i:02d}_{safe}.pdf')
            create_single_cover(info, fp)
            generated.append(fp)

        zip_path = os.path.join(os.getcwd(), 'Aria_CoverPages.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fp in generated:
                zf.write(fp, os.path.basename(fp))
        return zip_path
    finally:
        for fp in generated:
            try: os.remove(fp)
            except: pass
        try: os.rmdir(tmp_dir)
        except: pass
