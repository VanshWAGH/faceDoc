import subprocess
from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return render(request, 'index.html')

def capture_images(request):
    subprocess.Popen(['/usr/bin/python3', 'data_set.py'], cwd='/home/vanshwagh/Desktop/mysite/test1/pythonFiles')
    return HttpResponse("Capture Images script is running...")

def recognize_faces(request):
    subprocess.Popen(['/usr/bin/python3', 'detection.py'], cwd='/home/vanshwagh/Desktop/mysite/test1/pythonFiles')
    return HttpResponse("Recognize Faces script is running...")

def train_model(request):
    subprocess.Popen(['/usr/bin/python3', 'training.py'], cwd='/home/vanshwagh/Desktop/mysite/test1/pythonFiles')
    return HttpResponse("Train Model script is running...")
