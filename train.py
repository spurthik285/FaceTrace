import tkinter as tk
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

window = tk.Tk()

# helv36 = tk.Font(family='Helvetica', size=36, weight='bold')

window.title('Face_Recogniser')
window.minsize(1500, 900)

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'

# answer = messagebox.askquestion(dialog_title, dialog_text)

# window.geometry('1280x720')

window.configure(background='#97CAEF')

# window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
# Check if the StudentsDetails.csv file already exists or not and if not then create one and append respective data
if not os.path.isfile('StudentDetails\\StudentDetails.csv'):
    col = ['Id', 'Name']
    with open('StudentDetails\\StudentDetails.csv', 'a+', newline=''
              ) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(col)

# Tkinter GUI part

lbl = tk.Label(
    window,
    text='Enter ID',
    width=20,
    height=2,
    fg='white',
    bg='#2a1b3d',
    font=('times', 15, ' bold '),
    )
lbl.place(x=400, y=200)

txt = tk.Entry(window, width=20, bg='#e1fcff', fg='black', font=('times'
               , 15, ' bold '))
txt.place(x=700, y=200, width=200, height=50)

lbl2 = tk.Label(
    window,
    text='Enter Name',
    width=20,
    fg='white',
    bg='#2a1b3d',
    height=2,
    font=('times', 15, ' bold '),
    )
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window, width=20, bg='#e1fcff', fg='black',
                font=('times', 15, ' bold '))
txt2.place(x=700, y=300, width=200, height=50)

lbl3 = tk.Label(
    window,
    text='Notification : ',
    width=20,
    fg='white',
    bg='#2a1b3d',
    height=2,
    font=('times', 15, ' bold '),
    )
lbl3.place(x=400, y=400)

message = tk.Label(
    window,
    text='',
    bg='#e1fcff',
    fg='black',
    width=30,
    height=2,
    activebackground='yellow',
    font=('times', 15, ' bold '),
    )
message.place(x=700, y=400)

lbl3 = tk.Label(
    window,
    text='Attendance : ',
    width=20,
    fg='white',
    bg='#004043',
    height=2,
    font=('times', 15, ' bold  '),
    )
lbl3.place(x=400, y=650)

message2 = tk.Label(
    window,
    text='',
    bg='#e1fcff',
    fg='black',
    activeforeground='green',
    width=30,
    height=2,
    font=('times', 15, ' bold '),
    )
message2.place(x=700, y=650)

# function to clear texts in the inputbox
def clear():
    txt.delete(0, 'end')
    res = ''
    message.configure(text=res)


def clear2():
    txt2.delete(0, 'end')
    res = ''
    message.configure(text=res)

# check the validity of the input the user going to give
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def TakeImages():
    Id = txt.get()  # id taken from user
    name = txt2.get() # name taken from user
    flag = False
    search_file = open('StudentDetails\\StudentDetails.csv', 'rt') # path of the StudentDetails.csv file
    reader = csv.reader(search_file, delimiter=',') # read the csv file

    if len(name) == 0 and len(Id) == 0: # Checking whether Data given by user is valid or not and if not show warning
        res = 'Enter ID and Name'
        message.configure(text=res)
    if is_number(Id) and name.isalpha() and len(name) != 0:
        # parse through csv file row by row to check if ID given by user is already present or not
        for row in reader:
            if Id == row[0]:
                clear()
                clear2()
                res = 'Id ' + Id + ' already Present'
                message.configure(text=res)
                flag = True
                return
        # if ID is not present insert record and do next procedure
        if not flag:
            row = [Id, name]
            # append the data into StudentDeatils.csv file
            with open('StudentDetails\\StudentDetails.csv', 'a+',
                      newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)

            csvFile.close()
            res = 'Images Saved for ID : ' + Id + ' Name : ' + name
            message.configure(text=res) # Display the notification

        cam = cv2.VideoCapture(0)   # Start video capturing from camera
        harcascadePath = 'haarcascade_frontalface_default.xml' # path for the haar cascade classifiers file
        detector = cv2.CascadeClassifier(harcascadePath) # Loads a classifier from a file
        sampleNum = 0 # count for the Dataset of the faces
        while True:
            (ret, img) = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the images captured into grayscale
            faces = detector.detectMultiScale(gray, 1.12, 5) # Detects objects of different sizes in the input image.
            # The detected objects are returned as a list of rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0xFF, 0, 0), 2) # draw the rectangle on the facial portion
                sampleNum = sampleNum + 1
                cv2.imwrite("TrainingImage\ " + name + '.' + Id + '.'
                            + str(sampleNum) + '.jpg', gray[y:y + h, x:
                            x + w]) # Write the dataset images into TrainingUmage folder

                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'): # condtion to come out of the capturing procedure at will
                break
            elif sampleNum > 200:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = 'Images Saved for ID : ' + Id + ' Name : ' + name # show notification
    else:
        if is_number(Id):
            res = 'Enter Alphabetical Name'
            message.configure(text=res)
        if name.isalpha():
            res = 'Enter Numeric Id'
            message.configure(text=res)


def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create() # function call to the LBPHFaceRecognizer_create()
    harcascadePath = 'haarcascade_frontalface_default.xml'
    (faces, Id) = getImagesAndLabels('TrainingImage') # Load the training images from dataSet folder
    recognizer.train(faces, np.array(Id)) # train the dataset
    recognizer.save("TrainingImageLabel\Trainner.yml") # save the trainer file
    res = 'Image Trained'  # +",".join(str(f) for f in Id)
    message.configure(text=res) # show repective notification


def getImagesAndLabels(path):

    # get the path of all the files in the folder

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    # create empth face list

    faces = []

    # create empty ID list

    Ids = []

    # now looping through all the image paths and loading the Ids and the images

    for imagePath in imagePaths:

        # loading the image and converting it to gray scale

        pilImage = Image.open(imagePath).convert('L')

        # Now we are converting the PIL image into numpy array

        imageNp = np.array(pilImage, 'uint8')

        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split('.')[1])

        # extract the face from the training image sample

        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # function call to the LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml") # read the trainer.yml file
    harcascadePath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(harcascadePath) # Loads a classifier from a file
    df = pd.read_csv('StudentDetails\\StudentDetails.csv')
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX # set the font of the text to be shown on edges of the rectangle
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        (ret, im) = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # convert the images captured into grayscale
        faces = faceCascade.detectMultiScale(gray, 1.2, 5) # Detects objects of different sizes in the input image.
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            (Id, conf) = recognizer.predict(gray[y:y + h, x:x + w]) # Recognize the Id from the available data in trainer.yml file.
            if conf < 50: # 50 is threshold value so anything less than the threshold is successful recognition
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id) + '-' + aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
            else:
                # otherwise it is unknown to system
                Id = 'Unknown'
                tt = str(Id)
            if conf > 75:
                noOfFile = len(os.listdir('ImagesUnknown')) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile)
                            + '.jpg', im[y:y + h, x:x + w])
            cv2.putText(
                im,
                str(tt),
                (x, y + h),
                font,
                1,
                (0xFF, 0xFF, 0xFF),
                2,
                )
        attendance = attendance.drop_duplicates(subset=['Id'],
                keep='first')
        cv2.imshow('im', im)
        if cv2.waitKey(1) == ord('q'):
            break
    ts = time.time()
    cdate = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    (Hour, Minute, Second) = timeStamp.split(':')

    # row = attendance.values.tolist()

    today = datetime.date.today() # take todays datetime

    if str(today) == str(cdate):  # check if the recogntion date is the todays dat or not
        rec_id = []
        fileName = "Attendance\Attendance_" + cdate + '.csv' # path of the Attendace file

        if not os.path.isfile(fileName):  # 1st time creation of file
            attendance.to_csv(fileName, index=False)
        else:
            # Append data
            flag = False
            search_file = open(fileName, 'rt')
            reader = csv.reader(search_file, delimiter=',')
            for row in reader:
                if Id == row[0]:
                    flag = True
                    break
            if not flag:
                attendance.to_csv(fileName, mode='a', header=False,
                                  index=False)
    cam.release()
    cv2.destroyAllWindows()
    res = attendance
    message2.configure(text=res)

# fucntion to check the work hour of the person
def checkWork():
    Id = txt.get()
    ts = time.time()
    cdate = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') # take the current datetime
    fileName = "Attendance\Attendance_" + cdate + '.csv' # path of the file to be searched in
    search_file = open(fileName, 'rt')
    listtime = [] # list of the timimg of every recognition occurance of the given ID
    reader = csv.reader(search_file, delimiter=',')
    noWorkFlag = True # flag variable to check its work-time
    for row in reader:
        if Id == row[0]:
            listtime.append(row[3]) # append the timimgs of the matched the data
            noWorkFlag = False
    if noWorkFlag: # if noWorkFalg is false means no work data is present
        message2.configure(text='No Work Record for today')
    if not noWorkFlag: # if the noWorkFlag is true then notify their work-hour data
        datetime1 = datetime.datetime.strptime(listtime[0], '%H:%M:%S')
        datetime2 = datetime.datetime.strptime(listtime[-1], '%H:%M:%S')
        str1 = str(datetime2 - datetime1)
        message2.configure(text='Hr:Min:Sec = ' + str1)


# Tkinter part where actions are assigned to the respective buttons

clearButton = tk.Button(
    window,
    text='Clear',
    command=clear,
    fg='white',
    bg='#25274d',
    activebackground='Red',
    font=('times', 15, ' bold '),
    )
clearButton.place(x=950, y=215, height=25, width=100)
clearButton2 = tk.Button(
    window,
    text='Clear',
    command=clear2,
    fg='white',
    bg='#25274d',
    width=20,
    height=2,
    activebackground='Red',
    font=('times', 15, ' bold '),
    )
clearButton2.place(x=950, y=312, height=25, width=100)
takeImg = tk.Button(
    window,
    text='Capture Faces',
    command=TakeImages,
    fg='white',
    bg='#660066',
    width=20,
    height=3,
    activebackground='Red',
    font=('times', 15, ' bold '),
    )
takeImg.place(x=200, y=500)
trainImg = tk.Button(
    window,
    text='Train Faces',
    command=TrainImages,
    fg='white',
    bg='#660066',
    width=20,
    height=3,
    activebackground='Red',
    font=('times', 15, ' bold '),
    )
trainImg.place(x=500, y=500)
trackImg = tk.Button(
    window,
    text='Recognize Faces',
    command=TrackImages,
    fg='white',
    bg='#660066',
    width=20,
    height=3,
    activebackground='Red',
    font=('times', 15, ' bold '),
    )
trackImg.place(x=800, y=500)
check_work = tk.Button(
    window,
    text='Check Work Hour',
    command=checkWork,
    fg='white',
    bg='#660066',
    width=20,
    height=3,
    activebackground='Red',
    font=('times', 15, ' bold '),
    )
check_work.place(x=1100, y=500)

window.mainloop()
