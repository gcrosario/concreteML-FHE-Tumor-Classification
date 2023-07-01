from django.urls import path
from tumorClassifier import views

urlpatterns = [
    path("", views.index, name="home"),
    path("index", views.index, name="index"),
    path("start_classification", views.start_classification, name="start_classification"),
]