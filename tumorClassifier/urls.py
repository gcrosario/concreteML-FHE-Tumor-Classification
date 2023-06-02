from django.urls import path
from tumorClassifier import views

urlpatterns = [
    path("", views.home, name="home"),
]