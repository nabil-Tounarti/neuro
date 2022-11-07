from django.shortcuts import render ,redirect
from django.http import HttpResponse
from .utils import get_plot
from .utils import X_Y

# Create your views here.
def connexion(request):
    return render(request,'connexion.html')
def index(request):
    return render(request,'index.html')

def contact(request):
    return render(request,'contact.html')

def service(request):
    return render(request,'service.html')

def graph(request):
    x,y =X_Y()
    chart = get_plot([48,49,50,51,52,53,54,55,56,57,58,59],x,y)
    rest=-y[59]+x[11]
    return render(request,'graph.html',{'chart': chart,'rest':rest})

def send(request):
    return render(request,'send.html')

def prod(request):
    return render(request,'prod.html')

def collecte(request):
    return render(request, 'collecte.html')

def map(request):
    return render(request, 'map.html')
def con(request):
    return render(request,'index.html')

def click(request):
    x,y =X_Y()
    chart = get_plot([48,49,50,51,52,53,54,55,56,57,58,59],x,y)
    rest=-y[59]+x[11]
    return render(request,'click.html',{'chart': chart,'rest':rest})

