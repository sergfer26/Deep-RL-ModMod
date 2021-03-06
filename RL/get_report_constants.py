from reportlab.platypus import Table
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import TableStyle
from constanst import CONSTANTS,INPUTS,CONTROLS,OTHER_CONSTANTS

style = TableStyle([
    ('BACKGROUND', (0,0), (3,0), colors.blue),
    ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),

    ('ALIGN',(0,0),(-1,-1),'CENTER'),

    ('FONTNAME', (0,0), (-1,0), 'Courier-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 14),

    ('BOTTOMPADDING', (0,0), (-1,0), 12),

    ('BACKGROUND',(0,1),(-1,-1),colors.white),
])

ts = TableStyle(
    [
    ('BOX',(0,0),(-1,-1),2,colors.black),

    ('LINEBEFORE',(2,1),(2,-1),2,colors.red),
    ('LINEABOVE',(0,2),(-1,2),2,colors.green),

    ('GRID',(0,0),(-1,-1),2,colors.black),
    ]
)


def drawMyRuler(pdf):
    pdf.drawString(100,810, 'x100')
    pdf.drawString(200,810, 'x200')
    pdf.drawString(300,810, 'x300')
    pdf.drawString(400,810, 'x400')
    pdf.drawString(500,810, 'x500')

    pdf.drawString(10,100, 'y100')
    pdf.drawString(10,200, 'y200')
    pdf.drawString(10,300, 'y300')
    pdf.drawString(10,400, 'y400')
    pdf.drawString(10,500, 'y500')
    pdf.drawString(10,600, 'y600')
    pdf.drawString(10,700, 'y700')
    pdf.drawString(10,800, 'y800')





def dic_to_list(data,reg):
    lista = list()
    for k, v in data.items():   # iter on both keys and values
        if reg != None:
            if k.startswith(reg):
                lista.append([k, str(v.val),v.units,v.ok])
        else:
            lista.append([k, str(v.val),v.units,v.ok])
    lista.insert(0, ['Nombre', 'Valor','Unidades','Info'])
    return lista

def add_table(pdf,data,reg,x,y):
    data = dic_to_list(data,reg)
    table = Table(data)
    table.setStyle(style)
    table.setStyle(ts)
    table.wrapOn(pdf,400,100)
    table.drawOn(pdf, x, y)

def add_text(pdf,textLines,x,y):
    text = pdf.beginText(x, y)
    text.setFont("Courier", 18)
    text.setFillColor(colors.black)
    for line in textLines:
        text.textLine(line)
    pdf.drawText(text)

#def dic_to_list(data):
#    lista = [[k, str(v.val),v.units,v.ok] for k, v in list(data.items())]
#    lista.insert(0, ['Nombre', 'Valor','Unidades','Info'])
#    return lista



def create_report():
    fileName = 'Reporte_constantes.pdf'
    documentTitle = 'Document title!'
    title = 'Reporte de constantes'
    subTitle = ''
    pdf = canvas.Canvas(fileName)
    pdf.setTitle(documentTitle)
    pdf.drawCentredString(300, 800, title)
    pdf.setFillColorRGB(0, 0, 255)
    pdf.setFont("Courier-Bold", 26)
    pdf.drawCentredString(290,720, subTitle)
    #drawMyRuler(pdf)
    x = 50
    add_text(pdf,['Alpha'],x, 750)
    add_table(pdf,CONSTANTS,'alpha',x,540)
    add_text(pdf,['Beta'],x, 510)
    add_table(pdf,CONSTANTS,'beta',x,405)
    add_text(pdf,['Gamma'],x, 370)
    add_table(pdf,CONSTANTS,'gamma',x,200)
    pdf.showPage()
    add_text(pdf,['Delta'],x, 770)
    add_table(pdf,CONSTANTS,'delta',x,590)
    add_text(pdf,['Epsilon'],x, 545)
    add_table(pdf,CONSTANTS,'epsi',x,390)
    add_text(pdf,['Eta'],x, 350)
    add_table(pdf,CONSTANTS,'eta',x,40)
    pdf.showPage()
    add_text(pdf,['Lambda'],x, 770)
    add_table(pdf,CONSTANTS,'lamb',x,590)
    add_text(pdf,['Rho'],x, 545)
    add_table(pdf,CONSTANTS,'rho',x,420)
    add_text(pdf,['Tau'],x, 390)
    add_table(pdf,CONSTANTS,'tau',x,290)
    add_text(pdf,['Nu'],x, 260)
    add_table(pdf,CONSTANTS,'nu',x,60)
    pdf.showPage()
    add_text(pdf,['Phi'],x, 770)
    add_table(pdf,CONSTANTS,'phi',x,570)
    add_text(pdf,['Psi'],x, 535)
    add_table(pdf,CONSTANTS,'psi',50,430)
    add_text(pdf,['Omega'],x, 390)
    add_table(pdf,CONSTANTS,'omega',x,290)
    pdf.showPage()
    add_text(pdf,['Inputs'],x, 780)
    add_table(pdf,INPUTS,None,x,490)
    add_text(pdf,['Controles'],x, 450)
    add_table(pdf,CONTROLS,None,x,215)
    add_text(pdf,['Otras'],x, 180)
    add_table(pdf,OTHER_CONSTANTS,None,x,40)
    pdf.save() 

create_report()
