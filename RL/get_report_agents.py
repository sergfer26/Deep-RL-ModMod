
from reportlab.platypus import Table
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import TableStyle

from params import PARAMS_ENV, PARAMS_TRAIN
from ddpg.params import PARAMS_UTILS, PARAMS_DDPG






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





def dic_to_list(data):
    lista = [[v, str(k)] for v, k in list(data.items())]
    lista.insert(0, ['Parámetro', 'Valor'])
    return lista

def add_table(pdf,data,x,y):
    data = dic_to_list(data)
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

def add_image(PATH,pdf,name,x,y,width = 500,height=500):
    pdf.drawInlineImage(PATH + name , x,y, width = width, height=height,preserveAspectRatio=True)

def create_report(PATH):
    fileName = '/Reporte_agentes.pdf'
    fileName = PATH + fileName
    documentTitle = 'Document title!'
    title = 'Comparación de Agentes'
    subTitle = ''
    pdf = canvas.Canvas(fileName)

    pdf.setTitle(documentTitle)
    pdf.drawCentredString(300, 800, title)
# RGB - Red Green and Blue
    pdf.setFillColorRGB(0, 0, 255)
    pdf.setFont("Courier-Bold", 26)
    pdf.drawCentredString(290,720, subTitle)
    #drawMyRuler(pdf)
    add_image(PATH,pdf,'/actions_above_umbral.png',10, 350,600,600)
    add_image(PATH,pdf,'/mean_actions.png',10,10,600,600)
    pdf.showPage()
    #Siguiente pagina
    add_image(PATH,pdf,'/var_actions.png',10, 350,600,600)
    add_image(PATH,pdf,'/scores_productions.png',10,0,600,600)

    pdf.save()  
    
