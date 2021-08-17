from reportlab.pdfgen import canvas
import os,sys
from pathlib import Path
p = Path(os.getcwd())
sys.path.append(str(p))
from params import params_control

from base import add_table,add_text,add_image


def report_control_expert(PATH):
    fileName = '/reports/Experto.pdf'
    fileName = PATH + fileName
    documentTitle = 'Document title!'
    title = 'Reporte de Constantes'
    subTitle = ''
    pdf = canvas.Canvas(fileName)
    pdf.setTitle(documentTitle)
    pdf.drawCentredString(300, 800, title)
    pdf.setFillColorRGB(0, 0, 255)
    pdf.setFont("Courier-Bold", 26)
    pdf.drawCentredString(290,720, subTitle)
    add_text(pdf,['Constantes del control experto'],100, 750)
    add_table(pdf,params_control.PARAMS_CONTROL,100,250)
    pdf.showPage()
    add_image('results_ddpg/expert_control/images/violin_actions.png',pdf,'',10, 350,600,600)
    add_image('results_ddpg/expert_control/images/violin_rewards.png',pdf,'',10,0,600,600)
    pdf.save()  


if __name__ == "__main__":
    report_control_expert('results_ddpg/expert_control')