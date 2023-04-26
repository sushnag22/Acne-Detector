from flask import Flask, render_template, url_for, request
import io, json
from PIL import Image
from roboflow import Roboflow
rf = Roboflow(api_key="490txGCcR4mjEOmyVx97")
project = rf.workspace().project("acne-detection-v2")
model = project.version(1).model

app = Flask(__name__)

# confidence and overlap
max_size = (416, 416)
confidence = 20
overlap = 30
resultInJsonFf = None
resultInJsonLc = None
resultInJsonRc = None
ffCount, lcCount, rcCount = {}, {}, {}
res = 'None'

def generateReport(res, totalScore, ftemp, ltemp, rtemp,fn, fpu, fpa, fc, ln, lpu, lpa, lc, rn, rpu, rpa, rc):
    from fpdf import FPDF

    # Define the PDF document
    pdf = FPDF(format='A4')

    # Add a page
    pdf.add_page()

    pdf.set_font('Arial', 'B', 16)

    pdf.set_fill_color(255, 255, 255)
    pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 10, str('Acne Severity Analysis Report') , 0, 1, 'C')

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(10,10, str('Date:'), ln=1)
    pdf.cell(10,10, str('Patient Name:'), ln=1)
    pdf.cell(10,10, str('Gender:                        Age:'), ln=1)
    pdf.ln()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, str('Result: ' + res) , 0, 1, 'C')
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, str('Global Score: {}'.format(totalScore)), 0, 1, 'C')

    pdf.set_text_color(0,0,0)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(10, 10, str('Face Region'), ln=1)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(10, 10, str('      Frontal Face'), ln=1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 10, str('              nodule {0}, pustule {1}, papule {2}, comedone {3}'.format(fn, fpu, fpa,fc)), ln=1)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(10, 10, str('              Local Score: {}'.format(ftemp)), ln=1)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(10, 10, str('      Left Cheek'), ln=1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 10, str('              nodule {0}, pustule {1}, papule {2}, comedone {3}'.format(ln, lpu, lpa, lc)), ln=1)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(10, 10, str('              Local Score: {}'.format(ltemp)), ln=1)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(10, 10, str('      Right Cheek'), ln=1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 10, str('              nodule {0}, pustule {1}, papule {2}, comedone {3}'.format(rn, rpu, rpa, rc)), ln=1)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(10, 10, str('              Local Score: {}'.format(rtemp)), ln=1)


    pdf.image('static/images/result_of_upload_front_face.jpg', x=15, y=200,w=50)
    pdf.image('static/images/result_of_upload_left_cheek.jpg', x=75, y=200,w=50)
    pdf.image('static/images/result_of_upload_right_cheek.jpg', x=135, y=200,w=50)


    # Save the PDF document
    pdf.output('acne_report.pdf', 'F')


@app.route('/upload_front_face', methods=['POST'])
def upload_front_face():
    image_file = request.files['image']
    image_data = image_file.read()
    image = Image.open(io.BytesIO(image_data))
    image.thumbnail(max_size)
    image.save('static/images/upload_front_face.jpg')
    # infer on a local image
    global resultInJsonFf
    resultInJsonFf = model.predict('static/images/upload_front_face.jpg', confidence=confidence, overlap=overlap).json()
    # visualize your prediction
    model.predict('static/images/upload_front_face.jpg', confidence=confidence, overlap=overlap).save('static/images/result_of_upload_front_face.jpg')
    # parsed json data
    parsed_data = resultInJsonFf
    global ffCount
    # loop through each object in the "predictions" list
    for obj in parsed_data["predictions"]:
        # get the class of the object
        obj_class = obj["class"]
        # check if the class is already in the dictionary
        if obj_class in ffCount:
            # if it is, increment the count
            ffCount[obj_class] += 1
        else:
            # if it isn't, set the count to 1
            ffCount[obj_class] = 1
    css_file = url_for('static', filename='css/style.css')
    return render_template('left_cheek.html', css_file=css_file)

@app.route('/upload_left_cheek', methods=['POST'])
def upload_left_cheek():
    image_file = request.files['image']
    image_data = image_file.read()
    image = Image.open(io.BytesIO(image_data))
    image.thumbnail(max_size)
    image.save('static/images/upload_left_cheek.jpg')
    # infer on a local image
    global resultInJsonLc
    resultInJsonLc = model.predict('static/images/upload_left_cheek.jpg', confidence=confidence, overlap=overlap).json()
    # visualize your prediction
    model.predict('static/images/upload_left_cheek.jpg', confidence=confidence, overlap=overlap).save('static/images/result_of_upload_left_cheek.jpg')
    parsed_data = resultInJsonLc
    global lcCount
    # loop through each object in the "predictions" list
    for obj in parsed_data["predictions"]:
        # get the class of the object
        obj_class = obj["class"]
        # check if the class is already in the dictionary
        if obj_class in lcCount:
            # if it is, increment the count
            lcCount[obj_class] += 1
        else:
            # if it isn't, set the count to 1
            lcCount[obj_class] = 1
    css_file = url_for('static', filename='css/style.css')
    return render_template('right_cheek.html', css_file=css_file)

@app.route('/upload_right_cheek', methods=['POST'])
def upload_right_cheek():
    image_file = request.files['image']
    image_data = image_file.read()
    image = Image.open(io.BytesIO(image_data))
    image.thumbnail(max_size)
    image.save('static/images/upload_right_cheek.jpg')
    # infer on a local image
    global resultInJsonRc
    resultInJsonRc = model.predict('static/images/upload_right_cheek.jpg', confidence=confidence, overlap=overlap).json()
    # visualize your prediction
    model.predict('static/images/upload_right_cheek.jpg', confidence=confidence, overlap=overlap).save('static/images/result_of_upload_right_cheek.jpg')
    parsed_data = resultInJsonRc
    global rcCount
    # loop through each object in the "predictions" list
    for obj in parsed_data["predictions"]:
        # get the class of the object
        obj_class = obj["class"]
        # check if the class is already in the dictionary
        if obj_class in rcCount:
            # if it is, increment the count
            rcCount[obj_class] += 1
        else:
            # if it isn't, set the count to 1
            rcCount[obj_class] = 1
    
    # Final severity analysis based on GAGS
    totalScore = 0
    ftemp, ltemp, rtemp = 0, 0, 0
    for pred_class, count in ffCount.items():
        if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0') and count >= 1:
            if count >= 10:
                ftemp += 8 + 4 + 4
            else: ftemp += 8 + 4
            break
        if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2') and count >= 1:
            if count >= 10:
                ftemp += 6 + 3 + 3
            else: ftemp += 6 + 3
            break
        if (pred_class == "papule" or pred_class == "papules" or pred_class == '1') and count >= 1:
            if count >= 10:
                ftemp += 4 + 2 + 2
            else: ftemp += 4 + 2
            break
        if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3') and count >= 1:
            if count >= 10:
                ftemp += 2 + 1 + 1
            else: ftemp += 2 + 1
            break
        
    for pred_class, count in lcCount.items():
        if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0') and count >= 1:
            ltemp += 8
            break
        if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2') and count >= 1:
            ltemp += 6
            break
        if (pred_class == "papule" or pred_class == "papules" or pred_class == '1') and count >= 1:
            ltemp += 4
            break
        if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3') and count >= 1:
            ltemp += 2
            break

    for pred_class, count in rcCount.items():
        if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0') and count >= 1:
            rtemp += 8
            break
        if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2') and count >= 1:
            rtemp += 6
            break
        if (pred_class == "papule" or pred_class == "papules" or pred_class == '1') and count >= 1:
            rtemp += 4
            break
        if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3') and count >= 1:
            rtemp += 2
            break
    
    totalScore = ftemp+ltemp+rtemp
    if totalScore >= 1 and totalScore <= 18:
        res = 'Mild'
    elif totalScore >= 19 and totalScore <= 30:
        res ='Moderate'
    elif totalScore >= 31:
        res = 'Severe'
    else: res = 'None'
    
    print(totalScore, ftemp,rtemp, ltemp)

    # individual scores for each face part
    fn, fpu, fpa, fc, ln, lpu, lpa, lc, rn, rpu, rpa, rc = 0, 0,0,0,0,0,0,0,0,0,0,0 
    for pred_class, count in ffCount.items():
        if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0'): fn = count;
        if (pred_class == "papule" or pred_class == "papules" or pred_class == '1'): fpa = count;
        if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2'): fpu = count;
        if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3'): fc = count;
    for pred_class, count in lcCount.items():
        if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0'): ln = count;
        if (pred_class == "papule" or pred_class == "papules" or pred_class == '1'): lpa = count;
        if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2'): lpu = count;
        if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3'): lc = count;
    for pred_class, count in rcCount.items():
        if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0'): rn = count;
        if (pred_class == "papule" or pred_class == "papules" or pred_class == '1'): rpa = count;
        if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2'): rpu = count;
        if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3'): rc = count;

    generateReport(res, totalScore, ftemp, ltemp, rtemp,fn, fpu, fpa, fc, ln, lpu, lpa, lc, rn, rpu, rpa, rc)
    css_file = url_for('static', filename='css/style.css')
    return render_template('result.html', css_file=css_file, res = res, globalScore=totalScore, lsff=ftemp, lslc=ltemp, lsrc=rtemp, fn=fn, fpu=fpu, fpa=fpa, fc=fc, ln=ln, lpu=lpu, lpa=lpa, lc=lc, rn=rn, rpu=rpu, rpa=rpa, rc=rc)

@app.route('/')
def index():
    css_file = url_for('static', filename='css/style.css')
    return render_template('index.html', css_file=css_file)

@app.route('/instructions')
def instructions():
    css_file = url_for('static', filename='css/style.css')
    return render_template('instructions.html', css_file=css_file)

@app.route('/download')
def download():
    filename = 'acne_report.pdf'
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run()
