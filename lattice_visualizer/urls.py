from django.urls import path
from . import views

urlpatterns = [
    path('', views.matrix_input, name='matrix_input'),
]