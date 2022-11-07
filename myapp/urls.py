from django.urls import path
from . import views
urlpatterns =[
    path('' , views.index, name='index'),
     path('contact.html' , views.contact, name='contact'),
     path('service.html' , views.service, name='service'),
     path('index.html' , views.index, name='index'),
     path('graph.html' , views.graph, name='graph'),
     path('send.html' , views.send, name='send'),
     path('prod.html' , views.prod, name='prod'),
     path('collecte.html' , views.collecte, name='collecte'),
     path('map.html' , views.map, name='map'),
     path('connexion.html' , views.map, name='connexion'),
     path('click.html' , views.click, name='click'),


]