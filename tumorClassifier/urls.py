from django.urls import path
from tumorClassifier import views

urlpatterns = [
    path("", views.home, name="home"),
    path('start_classification', views.start_classification, name='start_classification'),
]