from reportlab.pdfgen import canvas
from base import add_table,add_text,add_image

from params import PARAMS_ENV, PARAMS_TRAIN
from ddpg.params import PARAMS_UTILS, PARAMS_DDPG


def create_report(PATH, time = 0):
    fileName = '/reports/Reporte.pdf'
    fileName = PATH + fileName
    documentTitle = 'Document title!'
    title = 'Reporte de Entrenamiento'
    subTitle = ''
    pdf = canvas.Canvas(fileName)

    pdf.setTitle(documentTitle)
    pdf.drawCentredString(300, 800, title)
    pdf.setFillColorRGB(0, 0, 255)
    pdf.setFont("Courier-Bold", 26)
    pdf.drawCentredString(290,720, subTitle)

    add_text(pdf,['Parámetros del Entrenamiento'],100, 750)
    add_table(pdf,PARAMS_TRAIN,100,610)
    

    add_text(pdf,['Parámetros del Ambiente'],100, 560)
    add_table(pdf,PARAMS_ENV,100,410)
    
    
    add_text(pdf,['Parámetros de DDPG'],100, 360)
    add_table(pdf,PARAMS_DDPG,100,210)
     

    add_text(pdf,['Parámetros del Ruido'],100, 170)
    add_table(pdf,PARAMS_UTILS,100,30)

    pdf.showPage()
    add_image(PATH,pdf,'/images/reward.png',10, 350,600,600)
    add_image(PATH,pdf,'/images/sim_climate_inputs.png',30,10)
    pdf.showPage()
    #Siguiente pagina
    add_image(PATH,pdf,'/images/sim_rh_par.png',30, 350,550,550)
    add_image(PATH,pdf,'/images/sim_climate.png',30,0,550,550)

    pdf.showPage()
    add_image(PATH,pdf,'/images/sim_prod.png',30, 350,550,550)
    add_image(PATH,pdf,'/images/sim_actions.png',30,0,550,550)
    pdf.showPage()
    add_image(PATH,pdf,'/images/violin_actions.png',30, 350,550,550)
    add_image(PATH,pdf,'/images/violin_rewards.png',30,0,550,550)
    cadena = 'Tiempo de ejecucion: '
    cadena += str(round((time/(60**2)),2)) + ' Horas'
    add_text(pdf,[cadena],30,60)
    pdf.save()  

    
