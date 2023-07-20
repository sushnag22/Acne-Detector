from flask import Flask, render_template, url_for, request, send_file
import io, json
from PIL import Image
from roboflow import Roboflow

# acne model
rf = Roboflow(api_key="490txGCcR4mjEOmyVx97")
project = rf.workspace().project("acne-detection-v2")
model = project.version(2).model

# face model
rf2 = Roboflow(api_key="490txGCcR4mjEOmyVx97")
project = rf2.workspace().project("facial-region-detection")
model2 = project.version(1).model

app = Flask(__name__)

# confidence and overlap
max_size = (416, 416)
confidence = 20
overlap = 30
resultInJsonFf = None
resultInJsonLc = None
resultInJsonRc = None
ffCount_forehead, ffCount_nose, ffCount_chin = {}, {}, {}
ffCount, lcCount, rcCount = {}, {}, {}
res = 'None'

patientName = ''

def generateReport(res, totalScore, ftemp, ltemp, rtemp,fn, fpu, fpa, fc,nn, npu, npa, nc, cn, cpu, cpa, cc, ln, lpu, lpa, lc, rn, rpu, rpa, rc):
    from fpdf import FPDF
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

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
    pdf.cell(10,10, str('Date & Time: {}').format(dt_string), ln=1)
    pdf.cell(10,10, str('Patient Registration Number: {0}'.format(patientRegNo)), ln=1)
    pdf.cell(10,10, str('Patient Name: {0}'.format(patientName)), ln=1)
    pdf.cell(10,10, str('Patient Age: {0}'.format(patientAge)), ln=1)
    pdf.cell(10,10, str('Patient Sex: {0}'.format(patientSex)), ln=1)
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
    pdf.cell(10, 10, str('              Forehead: nodule {0}, pustule {1}, papule {2}, comedone {3}'.format(fn, fpu, fpa,fc)), ln=1)
    pdf.cell(10, 10, str('              Nose: nodule {0}, pustule {1}, papule {2}, comedone {3}'.format(nn, npu, npa,nc)), ln=1)
    pdf.cell(10, 10, str('              Chin: nodule {0}, pustule {1}, papule {2}, comedone {3}'.format(cn, cpu, cpa,cc)), ln=1)
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


    pdf.image('static/images/result_of_upload_front_face.jpg', x=15, y=210,w=50)
    pdf.image('static/images/result_of_upload_left_cheek.jpg', x=75, y=210,w=50)
    pdf.image('static/images/result_of_upload_right_cheek.jpg', x=135, y=210,w=50)


    # Save the PDF document
    pdf.output('acne_report.pdf', 'F')


@app.route('/upload_front_face', methods=['POST'])
def upload_front_face():
    image_file = request.files['image']
    global patientRegNo
    patientRegNo = request.form['patientregno']
    print(patientRegNo)
    global patientName
    patientName = request.form['patientname']
    print(patientName)
    global patientAge
    patientAge = request.form['patientage']
    print(patientAge)
    global patientSex
    patientSex = request.form['patientsex']
    print(patientSex)
    image_data = image_file.read()
    image = Image.open(io.BytesIO(image_data))
    image.thumbnail(max_size)
    image.save('static/images/upload_front_face.jpg')
    # infer on a local image
    global resultInJsonFf
    resultInJsonFf = model.predict('static/images/upload_front_face.jpg', confidence=confidence, overlap=overlap).json()
    # visualize your prediction
    model.predict('static/images/upload_front_face.jpg', confidence=confidence, overlap=overlap).save('static/images/result_of_upload_front_face.jpg')
    data = model2.predict("static/images/upload_front_face.jpg", confidence=40, overlap=30).json()
    # parsed json data
    parsed_data = resultInJsonFf
    global ffCount
    fx0,fx1, fy0, fy1, nx0,nx1, ny0, ny1, cx0,cx1, cy0, cy1 = 0,0,0,0,0,0,0,0,0,0,0,0
    fconfi, nconfi, cconfi = 0, 0, 0
    # loop through each object in the "predictions" list
    for prediction in data['predictions']:
		# img = cv2.imread("compressed_image.jpg", cv2.IMREAD_COLOR)
        if prediction['class'] == 'Forehead' and prediction['confidence'] > fconfi:
            fconfi = prediction['confidence']
            # forehead_coordinates = (prediction['x'], prediction['y'], prediction['width'], prediction['height'])
            fx0 = prediction['x'] - prediction['width'] / 2
            fx1 = prediction['x'] + prediction['width'] / 2
            fy0 = prediction['y'] - prediction['height'] / 2
            fy1 = prediction['y'] + prediction['height'] / 2

        elif prediction['class'] == 'Nose' and prediction['confidence'] > nconfi:
            nconfi = prediction['confidence']
            # nose_coordinates = (prediction['x'], prediction['y'], prediction['width'], prediction['height'])
            nx0 = prediction['x'] - prediction['width'] / 2
            nx1 = prediction['x'] + prediction['width'] / 2
            ny0 = prediction['y'] - prediction['height'] / 2
            ny1 = prediction['y'] + prediction['height'] / 2

        elif prediction['class'] == 'Chin' and prediction['confidence'] > cconfi:
            cconfi = prediction['confidence']
            # chin_coordinates = (prediction['x'], prediction['y'], prediction['width'], prediction['height'])
            cx0 = prediction['x'] - prediction['width'] / 2
            cx1 = prediction['x'] + prediction['width'] / 2
            cy0 = prediction['y'] - prediction['height'] / 2
            cy1 = prediction['y'] + prediction['height'] / 2
        
    global ffCount_forehead
    global ffCount_nose
    global ffCount_chin

    for obj in parsed_data["predictions"]:
        obj_class = obj["class"]
        if (fx0 <= obj['x'] and fy0 <= obj['y']) and (fx1 >= obj['x'] and fy1 >= obj['y']):
            if obj_class in ffCount_forehead:
                ffCount_forehead[obj_class] += 1
            else:
                ffCount_forehead[obj_class] = 1
        elif (nx0 <= obj['x'] and ny0 <= obj['y']) and (nx1 >= obj['x'] and ny1 >= obj['y']):
            if obj_class in ffCount_nose:
                ffCount_nose[obj_class] += 1
            else:
                ffCount_nose[obj_class] = 1
        elif (cx0 <= obj['x'] and cy0 <= obj['y']) and (cx1 >= obj['x'] and cy1 >= obj['y']):
            if obj_class in ffCount_chin:
                ffCount_chin[obj_class] += 1
            else:
                ffCount_chin[obj_class] = 1
        if obj_class in ffCount:
            ffCount[obj_class] += 1
        else:
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
    data = model2.predict("static/images/upload_left_cheek.jpg", confidence=40, overlap=30).json()
    parsed_data = resultInJsonLc
    global lcCount
	# validation it is left lateral image
    lcc, rcc = 0, 0
    lcwidth, lcheight, rcwidth, rcheight = 0, 0, 0, 0
    for prediction in data['predictions']:
        if prediction['class'] == 'Left-cheek' and prediction['confidence'] > lcc:
            lcc = prediction['confidence']
            lcwidth, lcheight = prediction['width'], prediction['height']
        if prediction['class'] == 'Right-cheek' and prediction['confidence'] > rcc:
            rcc = prediction['confidence']
            rcwidth, rcheight = prediction['width'], prediction['height']
            
    if (int(lcwidth+lcheight)) <= 0:
        css_file = url_for('static', filename='css/style.css')
        return render_template('left_cheek.html', css_file=css_file, improper_image_lc = 1)

    if (rcwidth > lcwidth and rcheight > lcheight) and rcc > lcc:
        css_file = url_for('static', filename='css/style.css')
        return render_template('left_cheek.html', css_file=css_file, improper_image_lc = 1)

    lcx0, lcy0, lcx1, lcy1 = 0, 0, 0, 0
    lcconfi = 0
    for prediction in data['predictions']:
        if prediction['class'] == 'Left-cheek' and prediction['confidence'] > lcconfi:
            lcconfi = prediction['confidence']
            # leftcheek_coordinates = (prediction['x'], prediction['y'], prediction['width'], prediction['height'])
            lcx0 = prediction['x'] - prediction['width'] / 2
            lcx1 = prediction['x'] + prediction['width'] / 2
            lcy0 = prediction['y'] - prediction['height'] / 2
            lcy1 = prediction['y'] + prediction['height'] / 2


    for obj in parsed_data["predictions"]:
        obj_class = obj["class"]
        if (lcx0 <= obj['x'] and lcy0 <= obj['y']) and (lcx1 >= obj['x'] and lcy1 >= obj['y']):
            if obj_class in lcCount:
                lcCount[obj_class] += 1
            else:
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
    datarc = model2.predict("static/images/upload_right_cheek.jpg", confidence=40, overlap=30).json()
    parsed_data = resultInJsonRc
    global rcCount
	# validation it is right lateral image
    lcc, rcc = 0, 0
    lcwidth, lcheight, rcwidth, rcheight = 0, 0, 0, 0
    for prediction in datarc['predictions']:
        if prediction['class'] == 'Right-cheek' and prediction['confidence'] > rcc:
            rcc = prediction['confidence']
            rcwidth, rcheight = prediction['width'], prediction['height']
        if prediction['class'] == 'Left-cheek' and prediction['confidence'] > lcc:
            lcc = prediction['confidence']
            lcwidth, lcheight = prediction['width'], prediction['height']
    
    if (int(rcwidth+rcheight)) <= 0:
        css_file = url_for('static', filename='css/style.css')
        return render_template('right_cheek.html', css_file=css_file, improper_image_rc = 1)
    
    if ((lcwidth > rcwidth and lcheight > rcheight) and lcc > rcc):
        css_file = url_for('static', filename='css/style.css')
        return render_template('right_cheek.html', css_file=css_file, improper_image_rc = 1)

    rcx0, rcy0, rcx1, rcy1 = 0, 0, 0, 0
    rcconfi = 0
    for prediction in datarc['predictions']:
        if prediction['class'] == 'Right-cheek' and prediction['confidence'] > rcconfi:
            rcconfi = prediction['confidence']
            # rightcheek_coordinates = (prediction['x'], prediction['y'], prediction['width'], prediction['height'])
            rcx0 = prediction['x'] - prediction['width'] / 2
            rcx1 = prediction['x'] + prediction['width'] / 2
            rcy0 = prediction['y'] - prediction['height'] / 2
            rcy1 = prediction['y'] + prediction['height'] / 2

    for obj in parsed_data["predictions"]:
        obj_class = obj["class"]
        if (rcx0 <= obj['x'] and rcy0 <= obj['y']) and (rcx1 >= obj['x'] and rcy1 >= obj['y']):
            if obj_class in rcCount:
                rcCount[obj_class] += 1
            else:
                rcCount[obj_class] = 1
    
    # Final severity analysis based on GAGS
    totalScore = 0
    ftemp, ltemp, rtemp = 0, 0, 0

    for pred_class, count in ffCount_forehead.items():
        if ('nodule' in ffCount_forehead and ffCount_forehead['nodule'] >= 1) or ('nodules' in ffCount_forehead and ffCount_forehead['nodules'] >= 1) or ('0' in ffCount_forehead and ffCount_forehead['0'] >= 1) :
            if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0') and count >= 1:
                ftemp += 8
                break
            continue
        elif ('pustule' in ffCount_forehead and ffCount_forehead['pustule'] >= 1) or ('pustules' in ffCount_forehead and ffCount_forehead['pustules'] >= 1) or ('2' in ffCount_forehead and ffCount_forehead['2'] >= 1) :
            if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2') and count >= 1:
                ftemp += 6
                break
            continue
        elif ('papule' in ffCount_forehead and ffCount_forehead['papule'] >= 1) or ('papules' in ffCount_forehead and ffCount_forehead['papules'] >= 1) or ('1' in ffCount_forehead and ffCount_forehead['1'] >= 1) :
            if (pred_class == "papule" or pred_class == "papules" or pred_class == '1') and count >= 1:
                ftemp += 4
                break
            continue
        elif ('comedone' in ffCount_forehead and ffCount_forehead['comedone'] >= 1) or ('comedones' in ffCount_forehead and ffCount_forehead['comedones'] >= 1) or ('3' in ffCount_forehead and ffCount_forehead['3'] >= 1) :
            if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3') and count >= 1:
                ftemp += 2
                break
            continue

    for pred_class, count in ffCount_nose.items():
        if ('nodule' in ffCount_nose and ffCount_nose['nodule'] >= 1) or ('nodules' in ffCount_nose and ffCount_nose['nodules'] >= 1) or ('0' in ffCount_nose and ffCount_nose['0'] >= 1) :
            if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0') and count >= 1:
                ftemp += 4
                break
            continue
        elif ('pustule' in ffCount_nose and ffCount_nose['pustule'] >= 1) or ('pustules' in ffCount_nose and ffCount_nose['pustules'] >= 1) or ('2' in ffCount_nose and ffCount_nose['2'] >= 1) :
            if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2') and count >= 1:
                ftemp += 3
                break
            continue
        elif ('papule' in ffCount_nose and ffCount_nose['papule'] >= 1) or ('papules' in ffCount_nose and ffCount_nose['papules'] >= 1) or ('1' in ffCount_nose and ffCount_nose['1'] >= 1) :
            if (pred_class == "papule" or pred_class == "papules" or pred_class == '1') and count >= 1:
                ftemp += 2
                break
            continue
        elif ('comedone' in ffCount_nose and ffCount_nose['comedone'] >= 1) or ('comedones' in ffCount_nose and ffCount_nose['comedones'] >= 1) or ('3' in ffCount_nose and ffCount_nose['3'] >= 1) :
            if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3') and count >= 1:
                ftemp += 1
                break
            continue

    for pred_class, count in ffCount_chin.items():
        if ('nodule' in ffCount_chin and ffCount_chin['nodule'] >= 1) or ('nodules' in ffCount_chin and ffCount_chin['nodules'] >= 1) or ('0' in ffCount_chin and ffCount_chin['0'] >= 1) :
            if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0') and count >= 1:
                ftemp += 4
                break
            continue
        elif ('pustule' in ffCount_chin and ffCount_chin['pustule'] >= 1) or ('pustules' in ffCount_chin and ffCount_chin['pustules'] >= 1) or ('2' in ffCount_chin and ffCount_chin['2'] >= 1) :
            if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2') and count >= 1:
                ftemp += 3
                break
            continue
        elif ('papule' in ffCount_chin and ffCount_chin['papule'] >= 1) or ('papules' in ffCount_chin and ffCount_chin['papules'] >= 1) or ('1' in ffCount_chin and ffCount_chin['1'] >= 1) :
            if (pred_class == "papule" or pred_class == "papules" or pred_class == '1') and count >= 1:
                ftemp += 2
                break
            continue
        elif ('comedone' in ffCount_chin and ffCount_chin['comedone'] >= 1) or ('comedones' in ffCount_chin and ffCount_chin['comedones'] >= 1) or ('3' in ffCount_chin and ffCount_chin['3'] >= 1) :
            if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3') and count >= 1:
                ftemp += 1
                break
            continue
        
    for pred_class, count in lcCount.items():
        if ('nodule' in lcCount and lcCount['nodule'] >= 1) or ('nodules' in lcCount and lcCount['nodules'] >= 1) or ('0' in lcCount and lcCount['0'] >= 1) :
            if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0') and count >= 1:
                ltemp += 8
                break
            continue
        elif ('pustule' in lcCount and lcCount['pustule'] >= 1) or ('pustules' in lcCount and lcCount['pustules'] >= 1) or ('2' in lcCount and lcCount['2'] >= 1) :
            if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2') and count >= 1:
                ltemp += 6
                break
            continue
        elif ('papule' in lcCount and lcCount['papule'] >= 1) or ('papules' in lcCount and lcCount['papules'] >= 1) or ('1' in lcCount and lcCount['1'] >= 1) :
            if (pred_class == "papule" or pred_class == "papules" or pred_class == '1') and count >= 1:
                ltemp += 4
                break
            continue
        elif ('comedone' in lcCount and lcCount['comedone'] >= 1) or ('comedones' in lcCount and lcCount['comedones'] >= 1) or ('3' in lcCount and lcCount['3'] >= 1) :
            if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3') and count >= 1:
                ltemp += 2
                break
            continue

    for pred_class, count in rcCount.items():
        if ('nodule' in rcCount and rcCount['nodule'] >= 1) or ('nodules' in rcCount and rcCount['nodules'] >= 1) or ('0' in rcCount and rcCount['0'] >= 1) :
            if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0') and count >= 1:
                rtemp += 8
                break
            continue
        elif ('pustule' in rcCount and rcCount['pustule'] >= 1) or ('pustules' in rcCount and rcCount['pustules'] >= 1) or ('2' in rcCount and rcCount['2'] >= 1) :
            if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2') and count >= 1:
                rtemp += 6
                break
            continue
        elif ('papule' in rcCount and rcCount['papule'] >= 1) or ('papules' in rcCount and rcCount['papules'] >= 1) or ('1' in rcCount and rcCount['1'] >= 1) :
            if (pred_class == "papule" or pred_class == "papules" or pred_class == '1') and count >= 1:
                rtemp += 4
                break
            continue
        elif ('comedone' in rcCount and rcCount['comedone'] >= 1) or ('comedones' in rcCount and rcCount['comedones'] >= 1) or ('3' in rcCount and rcCount['3'] >= 1) :
            if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3') and count >= 1:
                rtemp += 2
                break
            continue
    
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
    fn, fpu, fpa, nn, npu, npa, nc, cn, cpu, cpa, cc, fc, ln, lpu, lpa, lc, rn, rpu, rpa, rc = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
    for pred_class, count in ffCount_forehead.items():
        if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0'): fn = count;
        if (pred_class == "papule" or pred_class == "papules" or pred_class == '1'): fpa = count;
        if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2'): fpu = count;
        if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3'): fc = count;
    for pred_class, count in ffCount_nose.items():
        if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0'): nn = count;
        if (pred_class == "papule" or pred_class == "papules" or pred_class == '1'): npa = count;
        if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2'): npu = count;
        if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3'): nc = count;
    for pred_class, count in ffCount_chin.items():
        if (pred_class == "nodule" or pred_class == "nodules" or pred_class == '0'): cn = count;
        if (pred_class == "papule" or pred_class == "papules" or pred_class == '1'): cpa = count;
        if (pred_class == "pustule" or pred_class == "pustules" or pred_class == '2'): cpu = count;
        if (pred_class == "comedone" or pred_class == "comedones" or pred_class == '3'): cc = count;
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

    generateReport(res, totalScore, ftemp, ltemp, rtemp,fn, fpu, fpa, fc,nn, npu, npa, nc, cn, cpu, cpa, cc, ln, lpu, lpa, lc, rn, rpu, rpa, rc)
    css_file = url_for('static', filename='css/style.css')
    return render_template('result.html', css_file=css_file, res = res, globalScore=totalScore, lsff=ftemp, lslc=ltemp, lsrc=rtemp, fn=fn+nn+cn, fpu=fpu+npu+cpu, fpa=fpa+npa+cpa, fc=fc+nc+cc, ln=ln, lpu=lpu, lpa=lpa, lc=lc, rn=rn, rpu=rpu, rpa=rpa, rc=rc)

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
